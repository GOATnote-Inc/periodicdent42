# Stage-3 Full Fusion — Progress Checkpoint

**Date**: October 20, 2025  
**Branch**: `feat/stage3-fusion-full`  
**Session**: Initial implementation session  
**Status**: 🔄 IN PROGRESS

---

## Completed Tasks ✅

### 1. **Comprehensive Implementation Plan** (commit `62ca01e`)
- Documented full architecture in `STAGE3_FUSION_FULL_PLAN.md`
- Estimated timeline: 8-10 hours
- Target: +15-25% speedup (≤557-525 μs from 656 μs baseline)

### 2. **Feature Flags Infrastructure** (commit `742da83`)

**File**: `tasks/fp8_sdpa_stage_c_wmma/build.py`

```python
USE_FUSED_SOFTMAX = 1     # Stage-3: Fused softmax in registers (eliminate sS)
USE_SMEM_SWIZZLE_XOR = 1  # Stage-3: XOR swizzle for bank conflicts
USE_CP_ASYNC_3STAGE = 0   # Stage-3: 3-stage pipeline (long seq only)
USE_WMMA_PV = 1           # Stage-2: WMMA P·V (baseline, default ON)
```

**Changes**:
- ✅ Added 3 new environment variables
- ✅ Added preprocessor defines (`-DUSE_FUSED_SOFTMAX`, etc.)
- ✅ Updated build summary printout
- ✅ Updated metadata capture for reproducibility
- ✅ Changed `USE_WMMA_PV` default to 1 (Stage-2 merged)

---

## Remaining Tasks 🔄

### Step 2: XOR Swizzle (1 hour)

**File**: `cudadent42/bench/kernels/sdpa_fp8_stage_c_wmma.cu`

**Goal**: Mitigate SMEM bank conflicts in K^T/V accesses.

```cuda
#ifdef USE_SMEM_SWIZZLE_XOR
  #define SWIZZLE_COL(n, d) ((d) ^ (((n) & 0x1) * 8))
#else
  #define SWIZZLE_COL(n, d) (d)
#endif

// In dequant loop:
const int d_swz = SWIZZLE_COL(n, d);
sKT_h[n][d_swz] = dequant_sim_fp8(k_u8, k_s);
sV_h[n][d_swz] = dequant_sim_fp8(v_u8, v_s);
```

**Expected**: -10 μs from bank conflict reduction (NCU: `l1tex__data_bank_conflicts`)

---

### Step 3: Fused Softmax in Registers (3-4 hours) ⚠️ COMPLEX

**Core Challenge**: Keep WMMA Q@K^T `c_frag` (FP32) in registers, compute softmax without materializing to `sS`.

**Algorithm**:

```cuda
#ifdef USE_FUSED_SOFTMAX
// 1. After WMMA Q@K^T → c_frag (FP32 16x16)
//    Extract scores & apply scale
float scores[16];
for (int i = 0; i < c_frag.num_elements; i++) {
    scores[i] = c_frag.x[i] * softmax_scale;
}

// 2. Warp-reduce row-wise max
float m_tile[16];
for (int r = 0; r < 16; r++) {
    float val = (thread owns row r) ? scores[...] : -INFINITY;
    m_tile[r] = warp_reduce_max(val);
}

// 3. Update global m/l, rescale U
for (int r = warp_m + 0..15; r++) {
    float m_old = m_smem[r];
    float m_new = max(m_old, m_tile[r]);
    float rescale = exp(m_old - m_new);
    
    // Rescale U
    for (int d = lane; d < D; d += 32) {
        U_smem[r][d] *= rescale;
    }
    
    // Update l
    float l_tile = sum over tile { exp(scores - m_new) };
    l_smem[r] = l_old * rescale + l_tile;
    m_smem[r] = m_new;
}

// 4. Store P = exp(score - m_new) to sP for WMMA
for (i, j in 16x16) {
    sP[warp_m + i][tile_n + j] = exp(scores[i*16+j] - m_new);
}

// 5. WMMA P @ V (unchanged from Stage-2)
wmma::load_matrix_sync(a_frag, &sP[...]);
wmma::mma_sync(...);
#endif
```

**Challenges**:
- WMMA fragment layout (each thread owns 8 non-contiguous FP32 elements)
- Warp reduction pattern (16 rows × 16 cols)
- Register pressure (c_frag + scores[] + m_tile[] + ...)
- Correctness (online softmax numerical stability)

**Expected**: -60 μs (eliminate sS write + read)

---

### Step 4: 3-Stage cp.async Scaffold (1 hour)

**Trigger**: Auto-enable when `S >= 2048`

```cuda
#ifdef USE_CP_ASYNC_3STAGE
    #define NUM_STAGES 3
    __shared__ uint8_t sK_u8[3][TILE_N][D_PAD];
    __shared__ uint8_t sV_u8[3][TILE_N][D_PAD];
    // ...
    __pipeline_wait_prior(1);  // Wait for t (not t+1)
#else
    #define NUM_STAGES 2
    // ...
    __pipeline_wait_prior(0);
#endif
```

**Expected**: +5% on long sequences (hides more latency)

---

## Validation Gates

Once Steps 2-4 are implemented:

1. **PTXAS**: regs ≤ 128, smem ≤ 64 KB, spills = 0
2. **Correctness**: 6/6 tests (small/mission × seeds 0,1,2)
3. **Performance**: p50 ≥ +15% vs Stage-2 baseline (656 μs → ≤557 μs)

**Only if all gates pass**: Merge to `main` and tag `v3.0-stage3-fusion`

---

## Estimated Remaining Time

| Task | Time | Difficulty |
|------|------|------------|
| Step 2: XOR swizzle | 1 hour | Easy |
| Step 3: Fused softmax | 3-4 hours | **HARD** |
| Step 4: 3-stage scaffold | 1 hour | Medium |
| PTXAS validation | 30 min | Easy |
| Correctness tests | 1 hour | Medium |
| Performance benchmarks | 1 hour | Easy |
| Reports & merge | 1 hour | Easy |
| **Total** | **8-10 hours** | - |

---

## Next Steps

**Immediate**: Implement Step 2 (XOR swizzle) on L4 GPU

**Command to continue**:
```bash
# SSH to L4
gcloud compute ssh cudadent42-l4-dev --zone=us-west1-c

# Activate environment
cd ~/periodicdent42
source venv/bin/activate
export PATH=/usr/local/cuda-12.2/bin:$PATH  # Or CUDA 12.8

# Fetch latest branch
git fetch origin feat/stage3-fusion-full
git checkout feat/stage3-fusion-full
git pull

# Begin kernel editing
vim cudadent42/bench/kernels/sdpa_fp8_stage_c_wmma.cu
# (Or use Cursor's remote editing)
```

---

## Notes for Continuation

- **Baseline**: Stage-2 at `v2.0-stage2-wmma-pv` (656 μs)
- **Previous Stage-3A** (+0.2%): Deferred, different approach (reused sS for P)
- **This Stage-3**: Full fusion (eliminate sS entirely)
- **Risk**: Step 3 (fused softmax) is complex, may need debugging iterations
- **Fallback**: Set `USE_FUSED_SOFTMAX=0` to revert to Stage-2

---

**Updated**: October 20, 2025 (Session 2)  
**Status**: Step 1 ✅, Step 2 ❌ (valid negative), Step 3 ready

---

## Session 2 Summary (October 20, 2025)

### Completed ✅

**1. Safety Guardrail**
- Set `USE_FUSED_SOFTMAX=0` by default (Stage-2 behavior until Step-3 lands)

**2. Step-2 Implementation**
- Vectorized u8→half dequantization with lane-group scatter
- Goal: Reduce SMEM bank conflicts via uint4 loads + XOR pattern

**3. Validation (all on L4 GPU)**
- PTXAS: 96 regs, 37 KB SMEM, 0 spills ✅
- Correctness: 6/6 PASS (all errors ≤ 0.06) ✅
- Performance: 696 μs (+6.1% regression) ❌

### Key Finding: Valid Negative Result

**Hypothesis**: Bank conflicts limited K/V dequantization performance.

**Reality**: Vectorization added overhead (84→96 regs) without performance gain.

**Metrics**:
- Baseline (Stage-2): ~656 μs (from earlier runs)
- Step-2 (XOR swizzle): 696 μs
- **Regression: +40 μs (+6.1%)**

**Root Cause**:
1. Original scalar loop already well-optimized by NVCC
2. Bank conflicts not a bottleneck (cp.async prefetch hides latency)
3. Vectorization complexity dominated any conflict reduction

**Recommendation**: **Revert Step-2** or set `USE_SMEM_SWIZZLE_XOR=0` by default.

### Artifacts Created

- `STAGE3_STEP2_VALID_NEGATIVE.md`: Comprehensive analysis
- `STAGE3_FUSED_SOFTMAX_NOTES.md`: Step-3 design (introspection LUT, register-level softmax)
- `scripts/validate_step2*.sh`: Build, correctness, perf validation scripts
- `results/2025-Stage3-Fusion-Full/step2-xor/`: All logs and summaries

### Next Session: Step-3 (Fused Softmax)

**Goal**: Eliminate sS buffer by computing softmax from WMMA `c_frag` in registers.

**Expected Savings**: -60 μs (sS write + read elimination)

**Approach**:
1. Generate WMMA accumulator layout LUT (introspection kernel)
2. Implement register-level softmax (row-wise max/sum, online update)
3. Validate incrementally (compare each phase with Stage-2)
4. Full validation (PTXAS, 6/6 correctness, performance)

**Estimated Time**: 3-4 hours

**Target**: ≤557 μs (+15% from 656 μs baseline)

---

**Session 2 Duration**: ~2 hours  
**Commits**: 8 (safety, impl, scripts, docs, artifacts)

---

## Session 3 Summary (October 20, 2025) 🔄 IN PROGRESS

### Completed ✅

**1. Safety Defaults**
- Set `USE_SMEM_SWIZZLE_XOR=0` by default (Step-2 regressed +6.1%)

**2. WMMA Accumulator LUT Generation**
- Created introspection kernel (`wmma16x16_accum_lut_gen.cu`)
- Added PyBind11 bindings for proper module export
- Generated LUT header on L4: 32 lanes × 8 (row, col) pairs
- Pattern verified: Each lane owns distributed 2×2 sub-blocks

**3. Implementation Specification**
- Comprehensive guide for fused softmax kernel changes
- Register budget: ~112 regs (target ≤128) ✅
- Conditional compilation strategy (USE_FUSED_SOFTMAX)
- Validation plan (build, correctness, perf)

### In Progress 🔄

**4. Kernel Implementation** (Next: ~2-3 hours)
- Modify WMMA Q@K^T section (keep c_frag in registers)
- Implement LUT-based register-level softmax
- Update softmax section (skip when fused)
- Fix P materialization for WMMA P·V
- Add conditional branches for Stage-2 fallback

**Estimated LOC**: ~200 lines of kernel modifications

### Artifacts

- `cudadent42/bench/kernels/wmma16x16_accum_lut.h`: Auto-generated LUT (37 lines)
- `cudadent42/bench/kernels/wmma16x16_accum_lut_gen.cu`: Introspection kernel
- `cudadent42/bench/kernels/wmma16x16_accum_lut_bindings.cpp`: PyBind11 wrapper
- `scripts/generate_wmma_lut.py`: Python LUT generator
- `scripts/generate_lut_on_l4.sh`: L4 execution wrapper
- `STAGE3B_IMPLEMENTATION_SPEC.md`: 248-line implementation guide

### Next Session: Complete Kernel Implementation

**Tasks**:
1. Add `#include "wmma16x16_accum_lut.h"` with USE_FUSED_SOFTMAX guard
2. Modify WMMA Q@K^T section (~80 LOC)
3. Wrap softmax section with `#if !USE_FUSED_SOFTMAX` (~10 LOC)
4. Fix P materialization conditional (~20 LOC)
5. Build & PTXAS validation on L4
6. Correctness: 6/6 tests (Stage-2 vs Stage-3B)
7. Performance: Mission shape, 500 iters (target ≤557 μs)
8. Merge to main if all gates pass

**Estimated Time**: 2-3 hours  
**Complexity**: High (multi-section kernel modifications, warp reductions)  
**Risk**: Medium (well-specified, incremental validation possible)

---

**Session 3 Duration**: ~4 hours (infrastructure + implementation)  
**Commits**: 13 (safety, LUT gen, spec, kernel impl, validation)  
**Status**: Implementation complete, ❌ correctness gate failed, debugging needed

---

## Session 3 Final Status

### What Was Accomplished ✅

**1. Infrastructure** (2 hours)
- Safety defaults (XOR swizzle OFF)
- WMMA accumulator LUT generation (introspection kernel + PyBind11)
- LUT verified on L4 (32 lanes × 8 elements)
- Comprehensive implementation spec (248 lines)

**2. Kernel Implementation** (2 hours)
- Added wmma16x16_accum_lut.h include with conditional compilation
- Modified 2 WMMA Q@K^T sections (192 LOC changes)
- Wrapped 2 scalar softmax sections with #if !USE_FUSED_SOFTMAX
- Stage-2 fallback path intact (USE_FUSED_SOFTMAX=0 by default)

**3. Validation Infrastructure**
- Comprehensive validation script (PTXAS, correctness, perf)
- Full validation run on L4

### Validation Results 🔬

**✅ Gate 1: PTXAS PASSED** (surprisingly good!)
- Stage-2: 96 regs, 37.1 KB SMEM, 0 spills
- Stage-3B: **73 regs** (↓23!), 35.1 KB SMEM (↓2 KB), 0 spills
- **Verdict**: Resource usage improved

**❌ Gate 2: Correctness FAILED** (fundamental bug)
- Stage-2: 6/6 tests PASS ✅
- Stage-3B: **0/6 tests PASS** ❌
- Errors: max_err=1.2-3.6 (100× tolerance), 37-85% bad elements
- **Verdict**: Implementation has critical bug

**⏸️ Gate 3: Performance** (not run due to correctness failure)

### Root Cause Hypotheses 🔍

1. **Missing __syncthreads()**: After P write, before WMMA P·V load
2. **Warp reduction broadcast**: Lane 0 may not have final reduced value
3. **Fragment size assumption**: Hardcoded `scores[8]` may be wrong
4. **LUT indexing**: Off-by-one or incorrect mapping
5. **Online softmax math**: `l_add` reduction incomplete

### Next Session: Debugging (2-3 hours) 🛠️

**Priority Fixes** (ranked by likelihood):
1. Add `__syncthreads()` after P materialization (easiest, high impact)
2. Fix warp reduction broadcast (ensure lane 0 has reduced value)
3. Verify `c_frag.num_elements` at runtime (likely 8, but confirm)

**Debugging Steps**:
1. Add debug prints (compare scores, m_row, l_add with Stage-2)
2. Verify LUT values (print sample, check mapping)
3. Test with single warp/tile first (simplify)
4. Apply fixes based on debug output
5. Revalidate (full 6-test suite)

**Expected**: Fix likely trivial (missing sync, wrong broadcast), 1-2 hours to resolve

---

**Session 3 Metrics**:
- Duration: 4 hours
- Commits: 13
- Files Changed: 7 (kernel, LUT, scripts, docs)
- LOC: 192 kernel, 248 spec, 141 validation script
- Status: ❌ **Blocked on correctness bug**
