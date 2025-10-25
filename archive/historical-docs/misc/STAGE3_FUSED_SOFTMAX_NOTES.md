# Stage-3: Fused Softmax from WMMA Accumulator (Registers)

**Date**: October 20, 2025  
**Branch**: `feat/stage3-fusion-full`  
**Status**: 📝 Design notes (not yet implemented)

---

## Goal

**Eliminate sS buffer** by computing softmax directly from WMMA Q@K^T accumulator fragments (`c_frag`) in registers, only materializing P for WMMA P·V.

**Expected Savings**: **-60 μs**
- sS write: ~30 μs
- sS read: ~30 μs
- Better register locality: +10 μs from ILP

**Target**: +15-25% speedup (656 μs → ≤557-525 μs)

---

## Current Flow (Stage-2)

```cuda
WMMA Q@K^T → c_frag (FP32 16×16)
Store c_frag → sS (half)              ← ELIMINATE
Load sS → softmax → sP (half)         ← ELIMINATE
WMMA sP @ V → U_smem (FP32)
```

**Problem**: Two SMEM round-trips (write + read) for a short-lived intermediate value.

---

## Target Flow (Stage-3)

```cuda
WMMA Q@K^T → c_frag (FP32 16×16)      ← Keep in registers
Softmax in registers:
  - Apply softmax_scale
  - Warp-reduce row-wise max/sum
  - Update global m/l in SMEM
  - Rescale U_smem
  - Store exp(score - m) → sP (half)  ← Only store P
WMMA sP @ V → U_smem (FP32)           ← Unchanged
```

**Savings**: No sS buffer, c_frag stays hot in registers.

---

## Implementation Plan

### Phase 1: WMMA Accumulator Layout Introspection

**Goal**: Generate a lookup table (LUT) mapping each lane's `c_frag.x[i]` to `(row, col)` in the 16×16 tile.

**Approach**: Write a tiny introspection kernel:

```cuda
__global__ void wmma_accum_introspect_kernel(half* out) {
    using namespace nvcuda::wmma;
    fragment<accumulator, 16, 16, 16, float> c_frag;
    
    // Fill with lane*8 + i to identify ownership
    for (int i = 0; i < c_frag.num_elements; i++) {
        c_frag.x[i] = threadIdx.x * 8 + i;
    }
    
    // Store to 16×16 tile
    wmma::store_matrix_sync(out, c_frag, 16, wmma::mem_row_major);
}
```

**Output**: A 16×16 matrix where `out[r][c] = lane*8 + slot`. Invert this on CPU to get `(row, col) = LUT[lane][slot]`.

**Result**: Header file `wmma16x16_accum_lut.h`:

```cpp
// Per-lane ownership of 16×16 accumulator (8 elements per lane)
constexpr int WMMA_ACCUM_LUT[32][8][2] = {
    // lane 0: owns 8 (row,col) pairs
    {{0,0}, {0,1}, {1,0}, {1,1}, {8,0}, {8,1}, {9,0}, {9,1}},
    // lane 1: ...
    // ...
};
```

### Phase 2: Register-Level Softmax

**Algorithm**:

```cuda
#ifdef USE_FUSED_SOFTMAX

// 1. Extract scores from c_frag and apply scale
float scores[8];  // Each thread owns 8 elements
#pragma unroll
for (int i = 0; i < 8; i++) {
    scores[i] = c_frag.x[i] * softmax_scale;
}

// 2. Compute per-row max using LUT + warp reduction
float m_tile[16];  // Per-row max for this 16×16 tile
#pragma unroll
for (int r = 0; r < 16; r++) {
    float my_max = -INFINITY;
    #pragma unroll
    for (int i = 0; i < 8; i++) {
        int row = WMMA_ACCUM_LUT[lane][i][0];
        if (row == r) {
            my_max = fmaxf(my_max, scores[i]);
        }
    }
    // Warp-reduce across columns
    my_max = warp_reduce_max(my_max);  
    m_tile[r] = my_max;  // Broadcast
}

// 3. Update global m/l and rescale U
for (int r = warp_m; r < warp_m + 16; r++) {
    float m_old = m_smem[r];
    float m_new = fmaxf(m_old, m_tile[r - warp_m]);
    float rescale = __expf(m_old - m_new);
    
    // Rescale U (distribute across warp lanes)
    for (int d = lane; d < D; d += 32) {
        U_smem[r][d] *= rescale;
    }
    
    // Compute l_tile (sum of exp(score - m_new) for this row)
    float l_add = 0.f;
    #pragma unroll
    for (int i = 0; i < 8; i++) {
        int row = WMMA_ACCUM_LUT[lane][i][0];
        if (row == (r - warp_m)) {
            l_add += __expf(scores[i] - m_new);
        }
    }
    l_add = warp_reduce_sum(l_add);  // Sum across columns
    
    // Update global l
    float l_old = l_smem[r];
    l_smem[r] = l_old * rescale + l_add;
    m_smem[r] = m_new;
}

// 4. Store P = exp(score - m_new) to sP for WMMA
//    (Each lane writes its 8 elements using the LUT)
#pragma unroll
for (int i = 0; i < 8; i++) {
    int row = WMMA_ACCUM_LUT[lane][i][0];
    int col = WMMA_ACCUM_LUT[lane][i][1];
    float m_new = m_smem[warp_m + row];
    sP[warp_m + row][tile_n + col] = __float2half(__expf(scores[i] - m_new));
}
__syncthreads();

// 5. WMMA P @ V (unchanged from Stage-2)
wmma::load_matrix_sync(a_frag, &sP[warp_m][tile_n], TILE_N);
wmma::load_matrix_sync(b_frag, &sV[tile_n][d_tile*16], D_PAD);
wmma::mma_sync(c_frag_pv, a_frag, b_frag, c_frag_pv);

#else
// Stage-2 path: Store c_frag → sS, then softmax from sS
#endif
```

---

## Register Budget Analysis

**New Variables**:
- `scores[8]`: 8 × FP32 = 32 bytes = 8 regs
- `m_tile[16]`: 16 × FP32 = 64 bytes = 16 regs (can be computed incrementally)
- Warp reduction temps: ~4 regs

**Total increase**: ~20-30 regs

**Budget Check**:
- Stage-2: 84 regs
- Step-2 (XOR): 96 regs
- Step-3 (fused): ~110-120 regs (still ≪ 128 limit ✅)

---

## SMEM Savings

**Current (Stage-2)**:
- sS: `[TILE_M][TILE_N] = [32][64] × 2 bytes = 4 KB`
- sP: `[TILE_M][TILE_N] = 4 KB`
- Total: 8 KB

**After Step-3**:
- sS: **ELIMINATED** (-4 KB)
- sP: 4 KB (still needed for WMMA)
- **Net savings**: 4 KB

**New SMEM total**: ~33 KB (down from 37 KB)

---

## Challenges

### 1. WMMA Fragment Layout Complexity
- Each thread owns 8 **non-contiguous** elements
- Need LUT to map `c_frag.x[i]` → `(row, col)`
- Warp reductions must aggregate across correct rows

### 2. Warp Reduction Pattern
- Need to reduce **16 rows** independently
- Each row's max/sum spans multiple lanes
- Use `__shfl_sync` for intra-warp communication

### 3. Numerical Stability
- Online softmax requires careful handling of `m_old - m_new` → `exp(...)`
- FP32 accumulation helps (already using FP32 `c_frag`)

### 4. Debugging
- Register-level operations are hard to debug
- Use `DEBUG_PRINT` to dump `scores[]` for small cases
- Validate against Stage-2 sS output before proceeding

---

## Validation Strategy

### Step 3A: Introspection Kernel
1. Write `wmma_accum_introspect_kernel`
2. Run on GPU, generate `wmma16x16_accum_lut.h`
3. Validate LUT correctness (check symmetry, coverage)

### Step 3B: Incremental Implementation
1. **Phase 1**: Extract scores, apply scale → compare with sS
2. **Phase 2**: Compute m_tile → compare with warp reduction
3. **Phase 3**: Update global m/l → validate online softmax math
4. **Phase 4**: Store P → compare with Stage-2 sP
5. **Phase 5**: Full flow → run 6/6 correctness tests

### Step 3C: Full Validation
- **PTXAS**: ≤128 regs, ≤64 KB SMEM, 0 spills
- **Correctness**: 6/6 tests (small/mission × seeds 0,1,2)
- **Performance**: p50 ≤557 μs (+15% from 656 μs baseline)

---

## Expected Outcomes

### Conservative (+15%)
**Target**: 557 μs (from 656 μs)
- sS eliminated: -60 μs
- Register locality: +10 μs ILP
- Net: -50 μs ≈ -8%... wait, that's only -8%, not -15%

**Revised**: Need -99 μs for -15%. Might need **Step 4 (3-stage cp.async)** to hit this.

### Optimistic (+25%)
**Target**: 525 μs
- Requires Step 3 + Step 4 combined
- Or additional micro-optimizations (warp specialization, etc.)

---

## Next Session Action Items

1. **Implement introspection kernel** → generate LUT
2. **Stub out fused softmax path** with `USE_FUSED_SOFTMAX=1`
3. **Incremental validation** (compare each phase with Stage-2)
4. **Full validation** (PTXAS, correctness, performance)
5. **Decision**: Merge if +15% met, otherwise iterate

---

**Status**: Ready to implement. Estimated time: **3-4 hours** (complex but well-defined).

