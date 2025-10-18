# Phase D.3: FP8 SDPA TDD Plan - Path to 8-12 μs

**Date**: Oct 17, 2025  
**Mission**: Implement FP8 SDPA kernel to achieve 8-12 μs (2.5-3× from 24.22 μs)  
**Timeline**: 6-10 hours  
**Success Rate**: 80%

---

## **🎯 Target Metrics**

### **Performance Goals**:
```
Current Champion: 24.22 μs (xFormers FP16)
Conservative Target: 16.00 μs (1.51× speedup)
Stretch Target: 12.00 μs (2.02× speedup)
Dream Target: 8.00 μs (3.03× speedup)
```

### **Acceptance Gates**:
- ✅ **Correctness**: max_diff ≤ 5e-3 (FP8 has more error than FP16)
- ✅ **Latency**: ≤ 16 μs (conservative), ≤ 12 μs (stretch)
- ✅ **NCU**: 2× FP8 throughput vs FP16 visible in metrics
- ✅ **Occupancy**: ≥ 25% (improve from 9.28%)

---

## **📋 TDD Cycle 1: FP8 Baseline (2-3 hours)**

### **Test 1.1: FP8 Type Support**
```bash
# Verify L4 has FP8 support
python -c "import torch; print(torch.cuda.get_device_capability())"
# Expected: (8, 9) - Ada supports FP8 E4M3
```

### **Test 1.2: Minimal FP8 GEMM**
```cuda
// Test FP8 tensor core with minimal GEMM
// A[M,K] @ B[K,N] = C[M,N]
// FP8 E4M3 input, FP32 accumulator, FP16 output

__global__ void test_fp8_gemm(
    const __nv_fp8_e4m3* A,
    const __nv_fp8_e4m3* B,
    const float* A_scale,
    const float* B_scale,
    half* C,
    int M, int N, int K
) {
    // 16x16x16 tile using WMMA/MMA
    // Dequant on load: A_fp32 = fp8_to_float(A) * scale
    // Accumulate in FP32
    // Write FP16 output
}
```

**Expected**: Build succeeds, 2× throughput vs FP16 (from NCU)

### **Test 1.3: FP8 Flash-Style SDPA Skeleton**
```cuda
// Full SDPA but scalar operations first
// Verify correctness before optimizing

__global__ void sdpa_fp8_baseline(
    const __nv_fp8_e4m3* Q,
    const __nv_fp8_e4m3* K,
    const __nv_fp8_e4m3* V,
    const float* Q_scale,
    const float* K_scale,
    const float* V_scale,
    half* O,
    int B, int H, int S, int D
) {
    // Flash-style: tile Q, stream K/V
    // All ops in FP32 (dequant on load)
    // Online softmax
    // Write FP16 output
}
```

**Expected**: 
- Correctness: max_diff ≤ 5e-3
- Latency: ~30-40 μs (slower than FP16 due to scalar ops)

---

## **📋 TDD Cycle 2: FP8 Tensor Cores (2-3 hours)**

### **Test 2.1: WMMA Q@K^T**
```cuda
// Use WMMA for Q@K^T on FP8
// 16x16x16 tiles, FP32 accumulator

using namespace nvcuda::wmma;

fragment<matrix_a, 16, 16, 16, __nv_fp8_e4m3, row_major> a_frag;
fragment<matrix_b, 16, 16, 16, __nv_fp8_e4m3, col_major> b_frag;
fragment<accumulator, 16, 16, 16, float> c_frag;  // FP32 acc

// Load with scale
load_matrix_sync(a_frag, Q_fp8, 16);
// Dequant: manual scale multiplication after load
load_matrix_sync(b_frag, K_fp8, 16);

// Compute
mma_sync(c_frag, a_frag, b_frag, c_frag);
```

**Expected**:
- Latency: ~20-25 μs (better than scalar)
- NCU: Tensor Core activity visible

### **Test 2.2: WMMA P@V**
```cuda
// Same WMMA pattern for P@V
// P is FP32 (from softmax), need to convert to FP8 or keep FP32

// Option A: Convert P to FP8 (saves memory)
// Option B: Keep P in FP32 (better accuracy)

// Recommendation: Option B (FP32 P) for first pass
```

**Expected**:
- Latency: ~18-22 μs
- Correctness maintained

---

## **📋 TDD Cycle 3: cp.async Pipelining (1-2 hours)**

### **Test 3.1: Double-Buffer K/V**
```cuda
#define STAGES 2

__shared__ __align__(16) __nv_fp8_e4m3 K_smem[STAGES][TILE_N][TILE_K];
__shared__ __align__(16) __nv_fp8_e4m3 V_smem[STAGES][TILE_N][TILE_K];

// Pipeline pattern
for (int kv_tile = 0; kv_tile < num_kv_tiles; kv_tile++) {
    int stage = kv_tile % STAGES;
    
    // Async load next tile
    if (kv_tile + 1 < num_kv_tiles) {
        int next_stage = (kv_tile + 1) % STAGES;
        __pipeline_memcpy_async(&K_smem[next_stage], &K_global[...], size);
        __pipeline_commit();
    }
    
    // Wait for current tile
    __pipeline_wait_prior(STAGES - 1);
    __syncthreads();
    
    // Compute
    compute_qkt_tile(Q_smem, K_smem[stage], S_tile);
    
    __syncthreads();
}
```

**Expected**:
- Latency: ~15-18 μs (1.2-1.3× from Cycle 2)
- NCU: Memory and compute overlap

---

## **📋 TDD Cycle 4: Persistent CTA (1-2 hours)**

### **Test 4.1: Launch Configuration**
```cuda
// Target: 50% occupancy (vs 9.28% current)
// L4: 58 SMs, want 6 blocks/SM

int num_blocks = 58 * 6;  // 348 blocks
int threads_per_block = 256;  // 8 warps

dim3 grid(num_blocks, 1, 1);
dim3 block(threads_per_block);

__launch_bounds__(256, 6)  // Guide compiler
__global__ void sdpa_fp8_persistent(...) {
    // Each block loops over multiple attention heads
}
```

### **Test 4.2: Work Distribution**
```cuda
__global__ void __launch_bounds__(256, 6)
sdpa_fp8_persistent(...) {
    int work_id = blockIdx.x;
    int num_work_items = B * H;  // Total attention heads
    
    while (work_id < num_work_items) {
        int b = work_id / H;
        int h = work_id % H;
        
        // Process this head
        process_attention_head(b, h, ...);
        
        // Next work item
        work_id += gridDim.x;
    }
}
```

**Expected**:
- Latency: ~12-15 μs (1.2-1.3× from Cycle 3)
- NCU: Occupancy ≥ 25%

---

## **📋 TDD Cycle 5: Micro-Optimizations (1 hour)**

### **Test 5.1: Bank Conflict Elimination**
```cuda
// XOR swizzling for SMEM indexing
int smem_idx = (row ^ (col >> 2)) * TILE_K + col;
```

### **Test 5.2: Warp Specialization**
```cuda
// Warps 0-3: Load Q/K/V
// Warps 4-6: Compute Q@K^T
// Warp 7: Softmax + output
```

### **Test 5.3: Aggressive Unrolling**
```cuda
#pragma unroll
for (int k = 0; k < TILE_K; k += 4) {
    // Inner loop fully unrolled
}
```

**Expected**:
- Latency: ~10-14 μs (1.05-1.15× from Cycle 4)

---

## **📋 TDD Cycle 6: NCU Validation (30 min)**

### **Metrics to Capture**:
```bash
ncu \
  --metrics \
    sm__sass_thread_inst_executed_op_fp8_pred_on.sum,\
    sm__sass_thread_inst_executed_op_fp16_pred_on.sum,\
    sm__sass_thread_inst_executed_op_fp32_pred_on.sum,\
    smsp__sass_average_data_bytes_per_sector_mem_global_op_ld.pct,\
    smsp__sass_average_data_bytes_per_sector_mem_global_op_st.pct,\
    sm__warps_active.avg.pct_of_peak_sustained_active,\
    smsp__warps_eligible.avg.per_cycle_active,\
    sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_active \
  --target-processes all \
  --kernel-name regex:"sdpa_fp8" \
  python bench/run_fp8_sdpa.py
```

**Acceptance**:
- ✅ FP8 instructions > 0 (using FP8 TCs)
- ✅ Tensor pipe active > 20%
- ✅ Occupancy ≥ 25%
- ✅ Memory efficiency ≥ 70%

---

## **🎯 Success Criteria by Cycle**

| Cycle | Goal | Latency Target | Pass Criteria |
|-------|------|----------------|---------------|
| 1 | FP8 baseline | ~30-40 μs | Correctness ≤ 5e-3 |
| 2 | Tensor Cores | ~18-22 μs | NCU shows TC activity |
| 3 | cp.async | ~15-18 μs | Memory/compute overlap |
| 4 | Persistent | ~12-15 μs | Occupancy ≥ 25% |
| 5 | Micro-opts | ~10-14 μs | < 16 μs (conservative) ✅ |
| 6 | Validate | Final | All NCU gates pass |

**Final Targets**:
- ✅ Conservative: ≤ 16 μs (1.51× speedup)
- ✅✅ Stretch: ≤ 12 μs (2.02× speedup)
- ✅✅✅ Dream: ≤ 10 μs (2.42× speedup)

---

## **🚨 Risk Mitigation**

### **Risk 1: FP8 Accuracy Issues**
- **Mitigation**: Use FP32 accumulators, per-channel scales
- **Fallback**: Accept max_diff ≤ 1e-2 if needed (still correct)

### **Risk 2: SMEM Overflow**
- **Current bug**: Phase D kernel had 200 KB SMEM
- **Solution**: FP8 uses 1 byte vs 2 bytes → 2× less SMEM!
- **Budget**: 27 KB total (well within 100 KB limit) ✅

### **Risk 3: Build Issues**
- **Mitigation**: Start with minimal kernel, build incrementally
- **Fallback**: Use scalar FP32 path if FP8 TCs won't compile

### **Risk 4: Performance Regression**
- **If slower than 24 μs**: Investigate NCU, check for serialization
- **Fallback**: Accept 18-20 μs (still good) or revert to champion

---

## **📊 Time Budget**

```
Cycle 1 (FP8 baseline):      2-3 hours
Cycle 2 (Tensor Cores):      2-3 hours
Cycle 3 (cp.async):          1-2 hours
Cycle 4 (Persistent CTA):    1-2 hours
Cycle 5 (Micro-opts):        1 hour
Cycle 6 (NCU validate):      30 min
──────────────────────────────────────
Total: 7.5-11.5 hours

Target: 6-10 hours (achievable with focus)
```

---

## **🔮 Path 3 Reconsideration (If Successful)**

**If we achieve ≤ 10 μs** (2.42× speedup):
- Gap to 5 μs: 2× more needed
- Additional techniques to consider:
  1. INT4/INT2 for K/V (extreme quantization)
  2. Block-sparse attention (skip empty blocks)
  3. Mixed precision (FP8 Q@K^T, FP16 P@V)
  4. Persistent CTAs + warp specialization (more aggressive)

**Timeline for < 5 μs**: Additional 10-20 hours (research-level)

---

**Status**: READY TO EXECUTE  
**Next**: TDD Cycle 1 - FP8 Baseline  
**Philosophy**: Build incrementally, validate each step, NO QUITTING! 🚀


