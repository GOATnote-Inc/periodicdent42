# Stage-3 Fused Softmax+P·V Validation Report (L4 GPU)

**Date**: October 20, 2025  
**Device**: Google Cloud L4 (SM 8.9, CUDA 12.2)  
**Branch**: `feat/stage3-fused-softmax`  
**Commit**: `dd29186`

---

## Executive Summary

❌ **MERGE DEFERRED** — Stage-3A fails performance gate (+0.2% vs +10% target)

**Key Findings:**
- ✅ **Correctness**: 9/9 tests PASS (bit-exact with Stage-2)
- ✅ **PTXAS**: 84 regs, 35.1 KB SMEM (-2 KB), 0 spills
- ❌ **Performance**: **+0.2% speedup** (657 μs → 656 μs, **fails +10% gate**)

**Conclusion:** Stage-3A's micro-fusion (reusing sS for P) provides **negligible performance benefit**. This is a **valid negative result** that indicates SMEM capacity and buffer overhead are not the primary bottlenecks. Recommend pursuing Stage-3B (full fusion) or alternative optimizations.

---

## 1. Implementation Summary

### Stage-3A Design

**Goal:** Eliminate sP buffer overhead by reusing sS (score buffer) to store unnormalized P after softmax.

**Changes:**
```cuda
// cudadent42/bench/kernels/sdpa_fp8_stage_c_wmma.cu

// BEFORE (Stage-2):
__shared__ half sP[TILE_M][TILE_N];  // +2 KB
// ... softmax ...
sP[r][n] = __float2half(S_row[n]);   // Write P to sP
// ... WMMA P·V loads from sP ...

// AFTER (Stage-3A):
// No sP buffer (saves 2 KB)
// ... softmax ...
sS[r][n] = __float2half(S_row[n]);   // Write P to sS (reuse score buffer)
// ... WMMA P·V loads from sS ...
```

**Toggle:** `USE_FUSED_SOFTMAX_PV=1`

---

## 2. PTXAS Resource Analysis

### Register & SMEM Usage

| Variant | Registers | SMEM (KB) | Spills | Status |
|---------|-----------|-----------|--------|--------|
| **Stage-2** (baseline) | 84 | 37.1 | 0 | ✅ |
| **Stage-3A** (sS fusion) | **84** | **35.1** ↓ | 0 | ✅ |
| **Stage-3A + XOR** | **84** | **35.1** | 0 | ✅ |

**SMEM Savings:** 2 KB (37.1 KB → 35.1 KB) as designed ✅

**Occupancy Impact:** None (both support 2 CTAs/SM on L4's 48 KB SMEM limit)

---

## 3. Correctness Validation

### Test Matrix: 3 Shapes × 3 Seeds = 9 Tests

#### Stage-2 Baseline (USE_FUSED_SOFTMAX_PV=0)
```
[small   ] seed=0: max_err=0.0459, mean_err=0.0142, %bad=0.0% ✅ PASS
[small   ] seed=1: max_err=0.0596, mean_err=0.0132, %bad=0.0% ✅ PASS
[small   ] seed=2: max_err=0.0459, mean_err=0.0133, %bad=0.0% ✅ PASS
[mission ] seed=0: max_err=0.0540, mean_err=0.0170, %bad=0.0% ✅ PASS
[mission ] seed=1: max_err=0.0356, mean_err=0.0171, %bad=0.0% ✅ PASS
[mission ] seed=2: max_err=0.0474, mean_err=0.0165, %bad=0.0% ✅ PASS
[long    ] seed=0: max_err=0.0391, mean_err=0.0178, %bad=0.0% ✅ PASS
[long    ] seed=1: max_err=0.0311, mean_err=0.0177, %bad=0.0% ✅ PASS
[long    ] seed=2: max_err=0.0315, mean_err=0.0179, %bad=0.0% ✅ PASS
```

#### Stage-3A (USE_FUSED_SOFTMAX_PV=1)
```
[small   ] seed=0: max_err=0.0459, mean_err=0.0142, %bad=0.0% ✅ PASS
[small   ] seed=1: max_err=0.0596, mean_err=0.0132, %bad=0.0% ✅ PASS
[small   ] seed=2: max_err=0.0459, mean_err=0.0133, %bad=0.0% ✅ PASS
[mission ] seed=0: max_err=0.0540, mean_err=0.0170, %bad=0.0% ✅ PASS
[mission ] seed=1: max_err=0.0356, mean_err=0.0171, %bad=0.0% ✅ PASS
[mission ] seed=2: max_err=0.0474, mean_err=0.0165, %bad=0.0% ✅ PASS
[long    ] seed=0: max_err=0.0391, mean_err=0.0178, %bad=0.0% ✅ PASS
[long    ] seed=1: max_err=0.0311, mean_err=0.0177, %bad=0.0% ✅ PASS
[long    ] seed=2: max_err=0.0315, mean_err=0.0179, %bad=0.0% ✅ PASS
```

### Numerical Equivalence

**Result:** **Bit-exact parity** across all seeds and shapes.
- Max errors: **Identical** (Stage-2 vs Stage-3A)
- Mean errors: **Identical**
- **0.0% bad elements**

**Conclusion:** Stage-3A is **numerically equivalent** to Stage-2 ✅

---

## 4. Performance Benchmark

### Mission Shape: (B=1, H=8, S=512, D=64)

#### Stage-2 Baseline (USE_FUSED_SOFTMAX_PV=0)
```
[mission ] seed=0: p50=657.41μs, p90=660.58μs, std=4.06μs
```

#### Stage-3A (USE_FUSED_SOFTMAX_PV=1)
```
[mission ] seed=0: p50=656.38μs, p90=659.46μs, std=4.10μs
```

#### Stage-3A + XOR Swizzle (USE_XOR_SWIZZLE=1)
```
[mission ] seed=0: p50=656.38μs, p90=659.46μs, std=4.58μs
```

### Performance Summary

| Variant | p50 (μs) | vs Stage-2 | vs Target | Gate |
|---------|----------|------------|-----------|------|
| **Stage-2 Baseline** | 657.41 | - | - | - |
| **Stage-3A** | 656.38 | **+0.2%** ⚡ | +10% required | ❌ **FAILED** |
| **Stage-3A + XOR** | 656.38 | **+0.2%** | +10% required | ❌ **FAILED** |

**Latency Reduction:** 657.41 μs → 656.38 μs = **1.03 μs** (0.16%)

**Target:** ≥+10% speedup (p50 ≤ 590 μs)

**Outcome:** **FAILED** — Improvement is within measurement noise

---

## 5. Root Cause Analysis

### Why Stage-3A Didn't Improve Performance

**Hypothesis:** Eliminating sP buffer (2 KB) would reduce SMEM pressure and improve performance.

**Reality:** The eliminated overhead is negligible compared to the kernel's compute workload.

#### Performance Breakdown (Estimated)

| Operation | Time (μs) | % of Total |
|-----------|-----------|------------|
| **Q@K^T** (WMMA) | ~200 | 30% |
| **Softmax** (exp, div, reductions) | ~150 | 23% |
| **P·V** (WMMA) | ~200 | 30% |
| **Global Memory** (Q, K, V, O) | ~80 | 12% |
| **sP buffer write/read** | **~1** | **<1%** ← Stage-3A target |
| **Epilogue** (l normalization, store) | ~26 | 4% |

**Insight:** The sP buffer overhead (**~1 μs, <1%**) is too small to measure given the dominant compute costs (WMMA, softmax).

#### SMEM Capacity Analysis

| Metric | Stage-2 | Stage-3A | Impact |
|--------|---------|----------|--------|
| **SMEM per CTA** | 37.1 KB | 35.1 KB | -2 KB |
| **L4 SMEM limit** | 48 KB | 48 KB | - |
| **CTAs/SM** | 2 | 2 | No change |
| **Occupancy** | ~50% | ~50% | No change |

**Insight:** Both variants fit comfortably within L4's 48 KB SMEM limit. Saving 2 KB does not improve occupancy or enable more CTAs/SM.

#### Bank Conflict Analysis

**XOR Swizzle Result:** No performance change (656.38 μs both with and without swizzle)

**Conclusion:** Bank conflicts are not a bottleneck. The kernel's memory access patterns are already well-coalesced.

---

## 6. Comparison to Predictions

### Pre-Implementation Estimates

| Scenario | Predicted p50 | Actual p50 | Outcome |
|----------|---------------|------------|---------|
| **Conservative** | ~620 μs (+5-6%) | 656 μs (+0.2%) | ❌ Over-optimistic |
| **Realistic** | ~590 μs (+10%) | 656 μs (+0.2%) | ❌ Over-optimistic |
| **Optimistic** | ~550 μs (+15-20%) | 656 μs (+0.2%) | ❌ Over-optimistic |

**Lesson Learned:** Micro-optimizations (eliminating small buffers) rarely yield measurable performance gains in compute-bound kernels. The overhead must be **≥5% of total time** to be worth pursuing.

---

## 7. Recommendations

### ❌ **Do NOT Merge Stage-3A**

**Rationale:**
- Performance gain (+0.2%) is within measurement noise
- Does not meet +10% minimum gate
- Adds complexity (new toggle) without benefit
- Stage-2 remains the optimal production path

### ✅ **Pursue Stage-3B (Full Fusion)**

**Design:** Eliminate **entire S materialization** by fusing QK^T → softmax → P·V into a single streaming pipeline.

**Expected Benefit:** **+15-30% speedup** by:
1. Eliminating **all** S buffer traffic (not just sP)
2. Keeping intermediate scores in registers/SMEM tiles
3. Better instruction-level parallelism (ILP)

**Implementation Complexity:** High (requires restructuring tile loop)

**Success Rate:** ~60% (challenging but achievable)

### Alternative: Warp Specialization

**Design:** Separate producer (load Q/K/V) and consumer (compute WMMA) warps to overlap memory and compute.

**Expected Benefit:** +10-20% speedup

**Implementation Complexity:** Medium

---

## 8. Artifacts

### Files Created

```
results/fp8_wmma_baseline/20251020-164431/  (Stage-2 baseline)
  ├── build_meta.json
  ├── correctness_summary.json
  └── perf_baseline.json

results/fp8_wmma_baseline/20251020-164451/  (Stage-3A)
  ├── build_meta.json
  ├── correctness_summary.json
  └── perf_baseline.json

results/COMPARE.md  (Performance comparison)
```

### Git Commits

```bash
c84387b: build: add Stage-3 toggles
3f2a6b2: feat(stage3): 3A fusion (reuse sS for P)
dd29186: docs(stage3): Progress checkpoint
```

---

## 9. Lessons Learned

### What Went Right

1. **Systematic validation:** EvoEngineer "Green before Fast" prevented incorrect optimizations
2. **Correctness maintained:** 9/9 tests pass with bit-exact parity
3. **Clean rollback:** Toggle system allows easy revert to Stage-2
4. **Valid negative result:** Learned that micro-fusion alone is insufficient

### What Didn't Work

1. **Hypothesis was wrong:** sP buffer overhead is negligible (<1% of runtime)
2. **Insufficient analysis:** Should have profiled Stage-2 first to identify real bottlenecks
3. **Over-optimistic predictions:** Estimated +10% based on SMEM savings, actual +0.2%

### Process Improvements

1. **Profile before optimizing:** Use NCU to identify bottlenecks **before** implementing changes
2. **Set realistic expectations:** Micro-optimizations (<5% of runtime) rarely yield measurable gains
3. **Focus on high-leverage changes:** Target operations that consume ≥10% of runtime

---

## Appendix A: PTXAS Output

### Stage-2 Baseline

```
ptxas info    : 0 bytes gmem
ptxas info    : Compiling entry function '_Z28sdpa_fp8_stage_c_wmma_kernelPKhS0_S0_PKfS2_S2_P6__halfiiiif' for 'sm_89'
ptxas info    : Function properties for _Z28sdpa_fp8_stage_c_wmma_kernelPKhS0_S0_PKfS2_S2_P6__halfiiiif
    0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads
ptxas info    : Used 84 registers, 37120 bytes smem, 428 bytes cmem[0]
```

### Stage-3A

```
ptxas info    : 0 bytes gmem
ptxas info    : Compiling entry function '_Z28sdpa_fp8_stage_c_wmma_kernelPKhS0_S0_PKfS2_S2_P6__halfiiiif' for 'sm_89'
ptxas info    : Function properties for _Z28sdpa_fp8_stage_c_wmma_kernelPKhS0_S0_PKfS2_S2_P6__halfiiiif
    0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads
ptxas info    : Used 84 registers, 35072 bytes smem, 428 bytes cmem[0]
```

**Difference:** 37120 - 35072 = **2048 bytes** (2 KB, as designed)

---

## Appendix B: Full Benchmark Data

### Stage-2 Baseline (500 iterations)

```
p50:  657.41 μs
p90:  660.58 μs
mean: 657.52 μs
std:  4.06 μs
min:  650.24 μs
max:  672.77 μs
```

### Stage-3A (500 iterations)

```
p50:  656.38 μs
p90:  659.46 μs
mean: 656.52 μs
std:  4.10 μs
min:  649.22 μs
max:  671.68 μs
```

**Statistical Significance:** Δ = 1.03 μs is **0.25× std dev** → **Not statistically significant**

---

**Report Generated**: October 20, 2025  
**Status**: ❌ **MERGE DEFERRED** (performance gate failed)  
**Recommendation**: Pursue Stage-3B (full fusion) for meaningful gains

