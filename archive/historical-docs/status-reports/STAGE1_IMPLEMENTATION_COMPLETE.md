# 🚀 **Stage-1 cp.async Implementation COMPLETE**

**Date**: October 20, 2025  
**Commit**: `6c79e3b`  
**Branch**: `feat/stage1-cp-async`  
**Status**: ✅ **Ready for GPU Validation**

---

## 📋 **Implementation Summary**

Successfully implemented **cp.async double-buffering** for K/V tile prefetching in the FP8 SDPA Stage-C WMMA kernel, controlled by `USE_CP_ASYNC` compile-time toggle.

### **Key Features**

| Feature | Description |
|---------|-------------|
| **Toggle** | `USE_CP_ASYNC` environment variable (0=baseline, 1=cp.async) |
| **Pipeline** | 2-stage ping-pong with `__pipeline_memcpy_async` |
| **Alignment** | 16B chunks for cp.async, scalar tail fallback |
| **Overlap** | Prefetch tile t+1 while processing tile t |
| **Numerics** | **100% preserved** (WMMA, softmax, P·V unchanged) |
| **SMEM** | +16 KB (sK_u8[2], sV_u8[2]) → ~38.5 KB total |
| **NVTX** | Optional profiling ranges for NCU |

---

## 🧬 **Implementation Details**

### **1. Headers & Toggles**

```cpp
#include <cuda_pipeline_primitives.h>

#ifndef USE_CP_ASYNC
#define USE_CP_ASYNC 0
#endif

// NVTX profiling ranges (optional)
#ifdef ENABLE_NVTX
#include <nvToolsExt.h>
#define NVTX_RANGE(name) nvtxRangePushA(name)
#define NVTX_POP() nvtxRangePop()
#else
#define NVTX_RANGE(name)
#define NVTX_POP()
#endif
```

### **2. Shared Memory Layout**

```cpp
// Working buffers (both paths)
__shared__ alignas(16) half sQ[TILE_M][D_PAD];   // 4 KB
__shared__ alignas(16) half sKT[TILE_N][D_PAD];  // 4 KB
__shared__ alignas(16) half sV[TILE_N][D_PAD];   // 4 KB

#if USE_CP_ASYNC
// Double-buffering for cp.async prefetch (uint8 staging)
__shared__ alignas(16) uint8_t sK_u8[2][TILE_N][D_PAD];  // 8 KB
__shared__ alignas(16) uint8_t sV_u8[2][TILE_N][D_PAD];  // 8 KB
#endif

// Per-row state + accumulator
__shared__ alignas(16) half sS[TILE_M][TILE_N];    // 2 KB
__shared__ float m_smem[TILE_M];                   // 128 B
__shared__ float l_smem[TILE_M];                   // 128 B
__shared__ alignas(16) float U_smem[TILE_M][D_PAD]; // 8 KB

// Total SMEM:
//   USE_CP_ASYNC=0: ~22.5 KB (baseline)
//   USE_CP_ASYNC=1: ~38.5 KB (cp.async)
```

### **3. cp.async Pipeline (USE_CP_ASYNC=1)**

```cpp
const int nTiles = (S + TILE_N - 1) / TILE_N;

// Helper: async copy one tile of K/V (uint8) from gmem to smem staging buffer
auto cp_async_tile_u8 = [&](int tile_idx, int stage) {
    if (tile_idx >= nTiles) return;
    
    const int kv_start = tile_idx * TILE_N;
    const int kv_len = min(TILE_N, S - kv_start);
    
    constexpr int BYTES = 16;  // 16B chunks for cp.async
    const size_t elems = (size_t)kv_len * D;
    const size_t bytes = elems * sizeof(uint8_t);
    
    uint8_t* __restrict__ dstK = &sK_u8[stage][0][0];
    uint8_t* __restrict__ dstV = &sV_u8[stage][0][0];
    const uint8_t* __restrict__ srcK = Kbh + (size_t)kv_start * D;
    const uint8_t* __restrict__ srcV = Vbh + (size_t)kv_start * D;
    
    // Copy in 16B chunks (safe for cp.async alignment)
    for (size_t off = threadIdx.x * BYTES; off + BYTES <= bytes; off += blockDim.x * BYTES) {
        __pipeline_memcpy_async(dstK + off, srcK + off, BYTES);
        __pipeline_memcpy_async(dstV + off, srcV + off, BYTES);
    }
    
    // Handle tail bytes with scalar copy (fallback for unaligned remainder)
    size_t tail = bytes % BYTES;
    if (tail && threadIdx.x == 0) {
        size_t off_tail = bytes - tail;
        for (size_t i = 0; i < tail; ++i) {
            dstK[off_tail + i] = srcK[off_tail + i];
            dstV[off_tail + i] = srcV[off_tail + i];
        }
    }
    __pipeline_commit();
};

// Prefetch tile 0 into stage 0
cp_async_tile_u8(0, 0);

for (int t = 0; t < nTiles; ++t) {
    const int read_stage  = t & 1;
    const int write_stage = (t + 1) & 1;
    
    // Prefetch next tile (overlaps with compute below)
    if (t + 1 < nTiles) {
        cp_async_tile_u8(t + 1, write_stage);
    }
    
    // Wait for current tile data (read_stage) to be visible
    __pipeline_wait_prior(1);
    __syncthreads();
    
    // Dequantize from u8 staging buffer → half working buffers (sKT, sV)
    const int kv_start = t * TILE_N;
    const int kv_len   = min(TILE_N, S - kv_start);
    
    for (int idx = tid; idx < kv_len * D; idx += blockDim.x) {
        int n = idx / D;
        int d = idx % D;
        uint8_t ku = sK_u8[read_stage][n][d];
        uint8_t vu = sV_u8[read_stage][n][d];
        float kf = dequant_sim_fp8(ku, k_s);
        float vf = dequant_sim_fp8(vu, v_s);
        sKT[n][d] = __float2half(kf);
        sV[n][d]  = __float2half(vf);
    }
    
    // Zero-pad for partial tiles
    for (int idx = tid + kv_len * D; idx < TILE_N * D; idx += blockDim.x) {
        int n = idx / D;
        int d = idx % D;
        sKT[n][d] = __float2half(0.f);
        sV[n][d]  = __float2half(0.f);
    }
    __syncthreads();
    
    // *** WMMA Q@K^T → sS (unchanged) ***
    // *** Online Softmax + P·V → U_smem (unchanged) ***
}

// Ensure all outstanding cp.async operations are complete
__pipeline_wait_prior(0);
__syncthreads();
```

### **4. Baseline Path (USE_CP_ASYNC=0)**

```cpp
// Preserved exact original implementation:
//   - Direct load K/V → dequant → sKT/sV
//   - WMMA compute (unchanged numerics)
//   - Online softmax (unchanged numerics)
//   - P·V accumulation (unchanged numerics)
```

---

## 🧪 **Validation Plan (GPU Required)**

### **Prerequisites**

```bash
# Ensure you're on the correct branch
git checkout feat/stage1-cp-async
git pull origin feat/stage1-cp-async

# Verify commit
git log --oneline -1
# Expected: 6c79e3b feat(fp8-wmma): Stage-1 cp.async double-buffering for K/V
```

### **Step 1: Correctness Validation (Both Paths)**

```bash
# Baseline (USE_CP_ASYNC=0)
USE_CP_ASYNC=0 python -m tasks.fp8_sdpa_stage_c_wmma.runner --shapes small,mission --seeds 0,1,2

# Expected: 6/6 PASS (same as infrastructure validation)

# Candidate (USE_CP_ASYNC=1)
USE_CP_ASYNC=1 python -m tasks.fp8_sdpa_stage_c_wmma.runner --shapes small,mission --seeds 0,1,2

# Expected: 6/6 PASS (numerics unchanged)
```

**Acceptance Criterion**: Both paths pass all correctness gates.

---

### **Step 2: Establish Baseline Performance**

```bash
# Baseline (USE_CP_ASYNC=0)
USE_CP_ASYNC=0 python -m tasks.fp8_sdpa_stage_c_wmma.runner \
    --shapes mission --seeds 0 --iters 500

# Note the output directory (e.g., results/fp8_wmma_baseline/20251020-143022/)
export BASE_DIR=$(ls -dt results/fp8_wmma_baseline/* | head -n1)
echo "Baseline: $BASE_DIR"
```

**Expected Output**:
```
[mission] seed=0: p50=XXX.XXμs, p90=XXX.XXμs, std=X.XXμs
```

---

### **Step 3: Measure Candidate Performance**

```bash
# Candidate (USE_CP_ASYNC=1)
USE_CP_ASYNC=1 python -m tasks.fp8_sdpa_stage_c_wmma.runner \
    --shapes mission --seeds 0 --iters 500

export CAND_DIR=$(ls -dt results/fp8_wmma_baseline/* | head -n1)
echo "Candidate: $CAND_DIR"
```

**Expected Output**:
```
[mission] seed=0: p50=YYY.YYμs, p90=YYY.YYμs, std=Y.YYμs
```

---

### **Step 4: Compare Performance**

```bash
python scripts/compare_results.py "$BASE_DIR/perf_baseline.json" "$CAND_DIR/perf_baseline.json"
cat results/COMPARE.md
```

**Expected Output**:
```markdown
# Performance Comparison

## mission (1, 8, 512, 64)

| Metric       | Baseline (μs) | Candidate (μs) | Speedup  | Status |
|--------------|---------------|----------------|----------|--------|
| p50          | XXX.XX        | YYY.YY         | +ZZ.Z%   | ✅      |
| p90          | XXX.XX        | YYY.YY         | +ZZ.Z%   | ✅      |
| mean         | XXX.XX        | YYY.YY         | +ZZ.Z%   | ✅      |

**Overall**: +ZZ.Z% speedup (p50)
```

**Acceptance Criterion**: Speedup ≥ +10% (p50)

---

### **Step 5: NCU Profiling (Optional)**

```bash
# Profile candidate with NCU
scripts/profile_ncu.sh mission 0 3

# Inspect NCU report
# Expected signals:
#   - smsp__pipe_tensor_cycles_active: ↑ (more tensor core activity)
#   - sm__throughput: ↑ (higher SM utilization)
#   - dram__bytes.sum: ~flat (same data, better overlap)
```

---

## 📊 **Expected Outcomes**

| Metric | Target | Rationale |
|--------|--------|-----------|
| **Correctness** | 6/6 PASS | Numerics unchanged |
| **Speedup (p50)** | ≥ +10% | cp.async hides gmem latency |
| **PTXAS Regs** | ≤128/thread | Lambda adds ~4 regs |
| **SMEM** | ≤64 KiB | 38.5 KB (safe for 2 CTAs/SM) |
| **Occupancy** | ≥50% | 2 CTAs/SM target |
| **NCU: Tensor Cycles** | ↑ | More TC time relative to mem stalls |
| **NCU: SM Throughput** | ↑ | Better instruction throughput |
| **NCU: DRAM Bytes** | ~flat | Same data moved, better overlap |

---

## 🔧 **Troubleshooting**

### **If Correctness Fails (USE_CP_ASYNC=1)**

**Symptom**: `max_abs_err` > 0.06 or `%bad` > 1.0%

**Likely Causes**:
1. **Stage Index Bug**: Verify `read_stage = t & 1` and `write_stage = (t + 1) & 1`
2. **Missing Wait**: Ensure `__pipeline_wait_prior(1)` before dequant
3. **sKT Layout**: Verify `sKT[n][d]` addressing for WMMA col-major

**Debug**:
```bash
# Enable debug prints
USE_CP_ASYNC=1 DEBUG_PRINT=1 python -m tasks.fp8_sdpa_stage_c_wmma.runner --shapes small --seeds 0
```

---

### **If Performance Flat (<10% Speedup)**

**Symptom**: `COMPARE.md` shows +0-5% speedup

**Likely Causes**:
1. **No Overlap**: Missing `cp_async_tile_u8(t + 1, write_stage)` before wait
2. **Excessive Syncs**: Extra `__syncthreads()` in wrong place
3. **Register Pressure**: PTXAS shows >128 regs/thread → spills to local mem

**Debug**:
```bash
# Check PTXAS output during build
# Look for "registers per thread" and "bytes spill stores" (should be 0)

# Run NCU to confirm overlap
scripts/profile_ncu.sh mission 0 3
# Inspect: smsp__pipe_tensor_cycles_active (should increase)
```

**Fixes**:
- Reduce unroll: `#pragma unroll 1` on dequant loops
- Hoist constants: Move `k_s`, `v_s` to registers outside loop
- Profile with `ENABLE_NVTX=1` to isolate bottleneck phase

---

### **If SMEM Exceeds Limit**

**Symptom**: Kernel launch fails with "too much shared memory"

**Likely Cause**: SMEM > 64 KiB (L4 dynamic SMEM limit)

**Fix**:
```cpp
// In launcher (sdpa_fp8_stage_c_wmma.cu, line ~552):
size_t smem_bytes = compute_smem_bytes();  // Add helper
if (smem_bytes > 64 * 1024) {
    // Fallback: reduce TILE_N or disable cp.async for this shape
    fprintf(stderr, "Warning: SMEM %zu KB exceeds 64 KB limit\n", smem_bytes / 1024);
    return cudaErrorInvalidConfiguration;
}
```

---

## 📚 **File Modifications**

### **Modified Files**

- `cudadent42/bench/kernels/sdpa_fp8_stage_c_wmma.cu` (+296 lines, -24 lines)

### **Unchanged Files (Build System Already Ready)**

- ✅ `tasks/fp8_sdpa_stage_c_wmma/build.py` (USE_CP_ASYNC toggle already implemented)
- ✅ `tasks/fp8_sdpa_stage_c_wmma/README.md` (USE_CP_ASYNC docs already present)
- ✅ `tasks/fp8_sdpa_stage_c_wmma/config_forward.json` (gates tuned)
- ✅ `tasks/fp8_sdpa_stage_c_wmma/runner.py` (robust validation framework)
- ✅ `scripts/compare_results.py` (performance comparison utility)

---

## 🎯 **Success Criteria Checklist**

Before considering Stage-1 **COMPLETE**, verify:

- [ ] **Correctness (Baseline)**: `USE_CP_ASYNC=0` passes 6/6 tests
- [ ] **Correctness (Candidate)**: `USE_CP_ASYNC=1` passes 6/6 tests
- [ ] **Performance**: `COMPARE.md` shows ≥ +10% speedup (p50)
- [ ] **PTXAS**: Regs ≤128/thread, SMEM ≤64 KiB, no spills
- [ ] **NCU**: Tensor cycles ↑, SM throughput ↑, DRAM bytes ~flat
- [ ] **Reproducibility**: `build_meta.json` saved with git SHA, flags, device info

---

## 🚀 **Next Steps (After Stage-1 Validation)**

Once Stage-1 is validated on GPU:

1. **Merge to Main**: `git push origin feat/stage1-cp-async` → PR → merge
2. **Stage-2 Prep**: See `docs/PERF_PLAN.md` for next optimization (Faster P·V or WMMA for P·V)
3. **Baseline Update**: Update `config_forward.json` with Stage-1 perf as new baseline
4. **NCU Deep Dive**: Identify next bottleneck (memory-bound vs compute-bound)
5. **EvoEngineer Loop**: Setup elite-of-3 loop for systematic iteration

---

## 📖 **References**

- **CUDA Pipeline Primitives**: `cuda_pipeline_primitives.h`
- **cp.async Alignment**: 4/8/16 bytes (NVIDIA Programming Guide)
- **EvoEngineer Framework**: "GREEN before FAST" gating strategy
- **FlashAttention-2**: Online softmax + tiling inspiration
- **WMMA Programming Guide**: Tensor Core usage patterns

---

**Status**: ✅ **Implementation Complete**  
**Next**: 🧪 **GPU Validation Required**  
**Target**: ≥ +10% speedup on mission shape (1,8,512,64)

---

**Questions or Issues?**  
Contact: `@engineer` with NCU reports + `correctness_summary.json`


