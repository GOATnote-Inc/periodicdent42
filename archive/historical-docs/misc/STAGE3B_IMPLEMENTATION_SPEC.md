# Stage-3B Fused Softmax: Implementation Specification

**Date**: October 20, 2025  
**Status**: Implementation guide (to be executed)

---

## Overview

Modify `cudadent42/bench/kernels/sdpa_fp8_stage_c_wmma.cu` to add fused softmax path behind `USE_FUSED_SOFTMAX=1` flag.

**Goal**: Keep WMMA Q@K^T accumulator (`c_frag`) in registers, compute softmax without materializing to `sS`, only store `P` for WMMA P·V.

---

## Changes Required

### 1. Add LUT Include (Top of File)

```cpp
// After existing includes
#if USE_FUSED_SOFTMAX
#include "wmma16x16_accum_lut.h"
#endif
```

### 2. Modify WMMA Q@K^T Section (Around Line 703-710)

**Current Code** (after WMMA mma_sync):
```cpp
wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, half> c_frag_fp16;
#pragma unroll
for (int i = 0; i < c_frag.num_elements; i++) {
    c_frag_fp16.x[i] = __float2half(c_frag.x[i]);
}
wmma::store_matrix_sync(&sS[warp_m][warp_n], c_frag_fp16, TILE_N, wmma::mem_row_major);
```

**New Code** (conditional):
```cpp
#if !USE_FUSED_SOFTMAX
    // Stage-2 path: store to sS (half) for later softmax
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, half> c_frag_fp16;
    #pragma unroll
    for (int i = 0; i < c_frag.num_elements; i++) {
        c_frag_fp16.x[i] = __float2half(c_frag.x[i]);
    }
    wmma::store_matrix_sync(&sS[warp_m][warp_n], c_frag_fp16, TILE_N, wmma::mem_row_major);
#else
    // Stage-3B fused path: keep c_frag in registers, do softmax inline
    
    // 1. Scale scores in-place
    float scores[8];  // c_frag.num_elements == 8 for 16x16x16 accumulator
    #pragma unroll
    for (int i = 0; i < 8; i++) {
        scores[i] = c_frag.x[i] * softmax_scale;
    }
    
    // 2. Per-row max (16 rows) using LUT + warp reduction
    float m_row[WMMA_M];  // 16 elements
    #pragma unroll
    for (int r = 0; r < WMMA_M; r++) {
        float mymax = -INFINITY;
        #pragma unroll
        for (int i = 0; i < 8; i++) {
            int rr = WMMA_ACCUM_LUT[lane][i][0];
            if (rr == r) {
                mymax = fmaxf(mymax, scores[i]);
            }
        }
        // Warp reduce max across lanes (covers 16 cols)
        #pragma unroll
        for (int offs = 16; offs > 0; offs >>= 1) {
            mymax = fmaxf(mymax, __shfl_down_sync(0xffffffff, mymax, offs));
        }
        // Broadcast result to all lanes
        m_row[r] = __shfl_sync(0xffffffff, mymax, 0);
    }
    
    // 3. Online softmax: update m/l, rescale U
    #pragma unroll
    for (int r = 0; r < WMMA_M; r++) {
        int r_glob = warp_m + r;
        if (r_glob >= rows_in_tile) continue;
        
        float m_old = m_smem[r_glob];
        float m_new = fmaxf(m_old, m_row[r]);
        float rescale = __expf(m_old - m_new);
        
        // Rescale U (each warp lane handles subset of D)
        for (int d = lane; d < D; d += 32) {
            U_smem[r_glob][d] *= rescale;
        }
        
        // Compute l_add (sum of exp(score - m_new) for this row)
        float l_add = 0.f;
        #pragma unroll
        for (int i = 0; i < 8; i++) {
            int rr = WMMA_ACCUM_LUT[lane][i][0];
            if (rr == r) {
                l_add += __expf(scores[i] - m_new);
            }
        }
        // Warp reduce sum
        #pragma unroll
        for (int offs = 16; offs > 0; offs >>= 1) {
            l_add += __shfl_down_sync(0xffffffff, l_add, offs);
        }
        
        // Lane 0 updates global m/l
        if (lane == 0) {
            float l_old = l_smem[r_glob];
            l_smem[r_glob] = l_old * rescale + l_add;
            m_smem[r_glob] = m_new;
        }
    }
    __syncwarp();
    
    // 4. Materialize P (half) for WMMA P·V
    //    Each lane writes its 8 elements using LUT
    #pragma unroll
    for (int i = 0; i < 8; i++) {
        int rr = WMMA_ACCUM_LUT[lane][i][0];
        int cc = WMMA_ACCUM_LUT[lane][i][1];
        int r_glob = warp_m + rr;
        int c_glob = warp_n + cc;
        if (r_glob < rows_in_tile && c_glob < kv_len) {
            float m_new = m_smem[r_glob];
            sP[r_glob][c_glob] = __float2half(__expf(scores[i] - m_new));
        }
    }
#endif // USE_FUSED_SOFTMAX
```

### 3. Modify Softmax Section (Skip if Fused)

**Find** the scalar softmax loop (around line 450-500) that processes `sS` → `sP`.

**Wrap with**:
```cpp
#if !USE_FUSED_SOFTMAX
    // Stage-2 path: scalar softmax from sS → sP
    NVTX_RANGE("softmax");
    for (int r = warp_id; r < rows_in_tile; r += NUM_WARPS) {
        // ... existing softmax code ...
    }
    NVTX_POP();
#else
    // Stage-3B: P already materialized above, skip
#endif
```

### 4. Fix P Materialization for WMMA P·V

The WMMA P·V section (around line 550-600) expects `sP` to be ready. With fused softmax, it's already done in the WMMA Q@K^T section. Just ensure no duplicate writes.

**Current** (in softmax loop):
```cpp
#if USE_WMMA_PV
    // Store P for WMMA
    for (int n = 0; n < kv_len; ++n) {
        sP[r][n] = __float2half(S_row[n]);
    }
    // Zero-pad
    for (int n = kv_len; n < TILE_N; ++n) {
        sP[r][n] = __float2half(0.f);
    }
#else
    // Scalar P·V
    for (int n = 0; n < kv_len; ++n) {
        float p = S_row[n];
        for (int d = lane; d < D; d += 32) {
            U_smem[r][d] += p * __half2float(sV[n][d]);
        }
    }
#endif
```

**Change to**:
```cpp
#if USE_WMMA_PV
    #if !USE_FUSED_SOFTMAX
        // Stage-2: Store P from softmax loop
        for (int n = 0; n < kv_len; ++n) {
            sP[r][n] = __float2half(S_row[n]);
        }
        for (int n = kv_len; n < TILE_N; ++n) {
            sP[r][n] = __float2half(0.f);
        }
    #endif
    // Stage-3B: P already in sP from WMMA section
#else
    // Scalar P·V (Stage-1 fallback)
    for (int n = 0; n < kv_len; ++n) {
        float p = S_row[n];
        for (int d = lane; d < D; d += 32) {
            U_smem[r][d] += p * __half2float(sV[n][d]);
        }
    }
#endif
```

---

## Register Budget Analysis

**Added Variables** (per thread):
- `scores[8]`: 8 × FP32 = 32 bytes = 8 regs
- `m_row[16]`: 16 × FP32 = 64 bytes = 16 regs
- Warp reduction temps: ~4 regs

**Total Increase**: ~28 regs

**Expected**:
- Stage-2 baseline: 84 regs
- Stage-3B fused: ~112 regs (well below 128 limit ✅)

---

## SMEM Impact

**Removed**: None (sS still allocated for Stage-2 fallback)  
**Savings** (in future): Could remove sS when USE_FUSED_SOFTMAX is stable

**Current SMEM**: ~37 KB (unchanged)

---

## Validation Strategy

1. **Build**: Check PTXAS (regs ≤ 128, spills = 0)
2. **Correctness**: 6/6 tests (compare Stage-2 vs Stage-3B)
3. **Performance**: Mission shape, 500 iters (target ≤557 μs, +15%)
4. **Debug**: Enable `DEBUG_PRINT` if issues, compare scores/P with Stage-2

---

## Rollback

If any gate fails:
- Keep `USE_FUSED_SOFTMAX=0` by default
- Stage-2 path remains intact
- Users can test with `USE_FUSED_SOFTMAX=1` explicitly

---

**Next**: Implement these changes in the kernel

