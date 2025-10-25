# Stage-3 Full Fusion Implementation Plan

**Date**: October 20, 2025  
**Branch**: `feat/stage3-fusion-full`  
**Baseline**: `main` at `v2.0-stage2-wmma-pv` (656 μs)  
**Target**: +15-25% speedup (≤557-525 μs)

---

## Objectives

Implement **full QK^T→softmax→P·V fusion** by:
1. **Eliminate sS buffer**: Compute softmax in registers from WMMA fragments
2. **Add XOR swizzle**: Mitigate SMEM bank conflicts for K^T/V loads
3. **Scaffold 3-stage cp.async**: For long sequences (S ≥ 2048)
4. **Keep WMMA P·V**: Maintain tensor core acceleration
5. **Preserve Stage-2 fallback**: Via `USE_FUSED_SOFTMAX` flag

---

## Architecture Changes

### Current (Stage-2)

```
Load Q tile → SMEM
For each KV tile:
  Load K/V with cp.async (2-stage)
  WMMA Q@K^T → c_frag (FP32)
  Store c_frag → sS (half)              ← ELIMINATE THIS
  Load sS → softmax → sP (half)         ← ELIMINATE THIS  
  WMMA sP @ sV → U_smem (FP32)
Normalize U_smem / l → O
```

### Target (Stage-3)

```
Load Q tile → SMEM (with XOR swizzle)
For each KV tile:
  Load K/V with cp.async (2 or 3-stage, XOR swizzled)
  WMMA Q@K^T → c_frag (FP32)           ← Keep in registers
  Softmax in registers:
    - Warp-reduce max/sum over c_frag
    - Update global m/l
    - Rescale U_smem
    - Store exp(score-m) → sP (half)   ← Only store P
  WMMA sP @ sV → U_smem (FP32)
Normalize U_smem / l → O
```

**Key Savings:**
- Eliminate sS write (2 KB, ~32 cycles)
- Eliminate sS read for softmax (~32 cycles)
- Better register reuse (c_frag stays hot)

---

## Implementation Steps

### Step 1: Add Feature Flags

**File**: `tasks/fp8_sdpa_stage_c_wmma/build.py`

```python
USE_FUSED_SOFTMAX = int(os.environ.get("USE_FUSED_SOFTMAX", "1"))  # NEW: default ON
USE_SMEM_SWIZZLE_XOR = int(os.environ.get("USE_SMEM_SWIZZLE_XOR", "1"))  # NEW: default ON
USE_CP_ASYNC_3STAGE = int(os.environ.get("USE_CP_ASYNC_3STAGE", "0"))  # NEW: default OFF

if USE_FUSED_SOFTMAX:
    extra_cuda_cflags.append("-DUSE_FUSED_SOFTMAX=1")
if USE_SMEM_SWIZZLE_XOR:
    extra_cuda_cflags.append("-DUSE_SMEM_SWIZZLE_XOR=1")
if USE_CP_ASYNC_3STAGE:
    extra_cuda_cflags.append("-DUSE_CP_ASYNC_3STAGE=1")
```

**Commit**: `build: add Stage-3 fusion flags (FUSED_SOFTMAX, SMEM_SWIZZLE_XOR, CP_ASYNC_3STAGE)`

### Step 2: XOR Swizzle for K^T/V SMEM

**File**: `cudadent42/bench/kernels/sdpa_fp8_stage_c_wmma.cu`

**Pattern**: XOR the column index with a function of row to spread accesses across banks.

```cuda
#ifdef USE_SMEM_SWIZZLE_XOR
  #define SWIZZLE_COL(n, d) ((d) ^ (((n) & 0x1) * 8))  // Toggle 8 on odd rows
#else
  #define SWIZZLE_COL(n, d) (d)
#endif

// In K/V dequantization loop:
const int d_swz = SWIZZLE_COL(n, d);
sKT_h[n][d_swz] = dequant_sim_fp8(k_u8, k_s);
sV_h[n][d_swz] = dequant_sim_fp8(v_u8, v_s);

// Update WMMA loads to account for swizzle:
// For row-major sV, leading dim stays D_PAD
// For col-major sKT, need to handle swizzled layout
```

**Commit**: `feat(stage3): add XOR swizzle for K^T/V SMEM (bank conflict mitigation)`

### Step 3: Fused Softmax (Eliminate sS)

**Core Algorithm**:

```cuda
#ifdef USE_FUSED_SOFTMAX
// After WMMA Q@K^T (c_frag is FP32 16x16):

// 1. Extract scores from c_frag to per-thread array
float scores[16];  // Each thread owns 8 elements of c_frag
for (int i = 0; i < c_frag.num_elements; i++) {
    scores[i] = c_frag.x[i] * softmax_scale;
}

// 2. Warp-reduce to get row-wise max (for each of 16 rows)
//    Use __shfl_sync to gather max across columns
float m_tile[16];  // Per-row max for this 16x16 tile
for (int r = 0; r < 16; r++) {
    float val = (thread owns row r) ? scores[...] : -INFINITY;
    m_tile[r] = warp_reduce_max(val);  // Broadcast to all lanes
}

// 3. Update global m/l and rescale U
for (int r = warp_m + 0..15) {
    float m_old = m_smem[r];
    float m_new = max(m_old, m_tile[r]);
    float rescale = exp(m_old - m_new);
    
    // Rescale U
    for (int d = lane; d < D; d += 32) {
        U_smem[r][d] *= rescale;
    }
    
    // Update l
    float l_old = l_smem[r];
    float l_tile = 0.f;
    for (int col in this tile) {
        l_tile += exp(scores[col] - m_new);
    }
    l_smem[r] = l_old * rescale + l_tile;
    m_smem[r] = m_new;
}

// 4. Store P = exp(score - m_new) to sP for WMMA
for (int i = 0; i < 16; i++) {
    for (int j = 0; j < 16; j++) {
        sP[warp_m + i][tile_n + j] = __float2half(exp(scores[i*16+j] - m_new));
    }
}

// 5. WMMA P @ V (unchanged)
wmma::load_matrix_sync(a_frag, &sP[warp_m][kTile], TILE_N);
wmma::load_matrix_sync(b_frag, &sV[kTile][dTile*16], D_PAD);
wmma::mma_sync(c_frag_pv, a_frag, b_frag, c_frag_pv);

#else
// Stage-2 path: Store c_frag → sS, then softmax from sS
#endif
```

**Challenges**:
1. **WMMA fragment layout**: Each thread owns 8 FP32 elements, need to map to 16x16 tile
2. **Warp reduction**: Need to reduce across 16 columns for each row
3. **Register pressure**: Keep c_frag, scores[], m_tile[], etc. in registers

**Commit**: `feat(stage3): fused softmax in registers (eliminate sS buffer)`

### Step 4: 3-Stage cp.async Scaffold

**File**: `cudadent42/bench/kernels/sdpa_fp8_stage_c_wmma.cu`

```cuda
#ifdef USE_CP_ASYNC_3STAGE
    #define NUM_STAGES 3
    __shared__ alignas(16) uint8_t sK_u8[3][TILE_N][D_PAD];
    __shared__ alignas(16) uint8_t sV_u8[3][TILE_N][D_PAD];
#else
    #define NUM_STAGES 2
    __shared__ alignas(16) uint8_t sK_u8[2][TILE_N][D_PAD];
    __shared__ alignas(16) uint8_t sV_u8[2][TILE_N][D_PAD];
#endif

// In tile loop:
#ifdef USE_CP_ASYNC_3STAGE
    // Prefetch t+2
    if (t + 2 < nTiles) {
        cp_async_tile(..., write_stage);
        __pipeline_commit();
    }
    __pipeline_wait_prior(1);  // Wait for t (not t+1)
#else
    // 2-stage (current)
    if (t + 1 < nTiles) {
        cp_async_tile(..., write_stage);
        __pipeline_commit();
    }
    __pipeline_wait_prior(0);
#endif
```

**Heuristic**: Enable 3-stage only when `S >= 2048` (auto-detect in launcher)

**Commit**: `feat(stage3): scaffold 3-stage cp.async for long sequences`

### Step 5: Resource Budget

**Targets**:
- **Registers**: ≤128 per thread (allow slight increase from Stage-2's 84)
- **SMEM**: ≤64 KB per CTA (Stage-2 was 35-37 KB, we add ~4 KB for 3-stage)
- **Spills**: 0

**Strategies** if over budget:
1. `#pragma unroll 1` on deep loops
2. Reuse c_frag for both QK^T and P·V
3. Move large arrays to SMEM if necessary
4. Reduce TILE_M/TILE_N if desperate

---

## Validation Gates

### Gate 1: PTXAS

```bash
# Stage-2 baseline
USE_FUSED_SOFTMAX=0 python -m tasks.fp8_sdpa_stage_c_wmma.build | tee .build_s2.log

# Stage-3 (2-stage)
USE_FUSED_SOFTMAX=1 USE_SMEM_SWIZZLE_XOR=1 \
  python -m tasks.fp8_sdpa_stage_c_wmma.build | tee .build_s3_2stage.log

# Stage-3 (3-stage)
USE_FUSED_SOFTMAX=1 USE_SMEM_SWIZZLE_XOR=1 USE_CP_ASYNC_3STAGE=1 \
  python -m tasks.fp8_sdpa_stage_c_wmma.build | tee .build_s3_3stage.log

grep -E "Function properties|Used|spill" .build_s*.log
```

**Pass**: regs ≤128, smem ≤64 KB, spills=0

### Gate 2: Correctness (6/6)

```bash
# Stage-2 baseline
USE_FUSED_SOFTMAX=0 python -m tasks.fp8_sdpa_stage_c_wmma.runner \
  --shapes small,mission --seeds 0,1,2 | tee .corr_s2.log

# Stage-3
USE_FUSED_SOFTMAX=1 python -m tasks.fp8_sdpa_stage_c_wmma.runner \
  --shapes small,mission --seeds 0,1,2 | tee .corr_s3.log
```

**Pass**: 6/6 PASS, max_err matches Stage-2 (ideally bit-exact)

### Gate 3: Performance

```bash
# Stage-2 baseline
USE_FUSED_SOFTMAX=0 python -m tasks.fp8_sdpa_stage_c_wmma.runner \
  --shapes mission --seeds 0 --iters 500 | tee .perf_s2.log

# Stage-3 (2-stage)
USE_FUSED_SOFTMAX=1 python -m tasks.fp8_sdpa_stage_c_wmma.runner \
  --shapes mission --seeds 0 --iters 500 | tee .perf_s3_2stage.log

# Stage-3 (3-stage, if long seq)
USE_FUSED_SOFTMAX=1 USE_CP_ASYNC_3STAGE=1 python -m tasks.fp8_sdpa_stage_c_wmma.runner \
  --shapes long --seeds 0 --iters 500 | tee .perf_s3_3stage.log
```

**Pass**: p50 improvement ≥+15% on mission (≤557 μs from 656 μs)

---

## Expected Outcomes

### Conservative (+15%)

**Baseline (Stage-2)**: 656 μs  
**Target (Stage-3)**: 557 μs  
**Savings breakdown**:
- sS write eliminated: ~30 μs
- sS read eliminated: ~30 μs
- XOR swizzle (bank conflicts): ~10 μs
- Better register locality: ~29 μs

### Optimistic (+25%)

**Target**: 525 μs  
**Additional gains**:
- 3-stage cp.async on long seq: +5%
- Improved ILP from register fusion: +5%

---

## Risks & Mitigation

| Risk | Mitigation |
|------|------------|
| **Register spills** | `#pragma unroll 1`, reuse fragments |
| **SMEM overflow** | Remove sS buffer (saves 2 KB) |
| **Correctness issues** | Incremental dev, test after each change |
| **WMMA fragment layout confusion** | Use debug prints, validate mapping |
| **Warp reduction complexity** | Use proven patterns from FlashAttention |

---

## Rollback Plan

If any gate fails:
1. **Use `USE_FUSED_SOFTMAX=0`** to revert to Stage-2
2. **Debug with `DEBUG_PRINT=1`** to isolate issues
3. **Do NOT merge** until all gates pass

---

## Timeline

| Step | Estimated Time | Status |
|------|----------------|--------|
| 1. Feature flags | 30 min | ⏳ |
| 2. XOR swizzle | 1 hour | ⏳ |
| 3. Fused softmax | 3-4 hours | ⏳ |
| 4. 3-stage scaffold | 1 hour | ⏳ |
| 5. PTXAS gate | 30 min | ⏳ |
| 6. Correctness gate | 1 hour | ⏳ |
| 7. Performance gate | 1 hour | ⏳ |
| 8. Reports & merge | 1 hour | ⏳ |
| **Total** | **8-10 hours** | ⏳ |

---

**Created**: October 20, 2025  
**Next**: Begin Step 1 (feature flags)

