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

**Updated**: October 20, 2025  
**Status**: Step 1 complete ✅, Step 2 ready to implement
