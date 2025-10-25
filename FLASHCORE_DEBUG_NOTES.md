# FlashCore Fused Kernel - Debug Notes

**Date**: October 22, 2025  
**Status**: ⚠️ **Correctness bug - Performance is excellent**

---

## 🎯 Current Status

### Performance ✅
- **Latency**: 280 μs (vs 634 μs baseline = **4.5× speedup!**)
- **vs PyTorch SDPA**: 280 μs vs 44 μs = 6.4× slower (expected - no cp.async yet)
- **Resource usage**: 91 regs, 27 KB SMEM, 0 spills ✅

### Correctness ❌
- **max_err**: 7.87 (threshold: 0.06) - **FAIL**
- **mean_err**: 0.127

**Error is too large** - this suggests a fundamental bug, not numerical precision.

---

## 🐛 Debugging Trail

### Attempt 1: Direct Fragment Softmax (FAILED)
**Approach**: Try to do online softmax directly in WMMA fragments using LUT  
**Result**: max_err = 2.38, performance = 1874 μs  
**Issue**: Fragment layout too complex, per-row iteration error-prone  

### Attempt 2: Simplified SMEM Softmax (CURRENT)
**Approach**: Store WMMA result to sS, do softmax in shared memory, then WMMA P@V  
**Result**: max_err = 7.87, performance = 280 μs ✅  
**Issue**: Performance is excellent but results are wrong!

### Fixes Tried
1. ✅ Added correct WMMA_ACCUM_LUT from reference
2. ✅ Added sS zeroing to avoid garbage
3. ✅ Split softmax into two phases (update stats, then materialize P)
4. ❌ **Still failing!**

---

## 🔍 Hypotheses

### Hypothesis 1: Memory Access Pattern Bug
**Theory**: Reading/writing to wrong indices in sS, sP, or U_smem  
**Evidence**: Large error (7.87) suggests systematic wrongness  
**Check**: 
- sS is [TILE_M][TILE_N] = [32][32]
- Warps write to non-overlapping 16×16 regions
- wmma::store_matrix_sync uses stride TILE_N = 32 ✓
- Softmax reads entire row sS[m][0:kv_len] ✓

**Status**: Looks correct, but need to verify with debug output

### Hypothesis 2: Online Softmax Algorithm Bug
**Theory**: The online update formula is wrong or applied incorrectly  
**Formula** (from FlashAttention-2):
```
m_new = max(m_old, m_tile)
rescale = exp(m_old - m_new)
U_new = U_old * rescale + P_tile @ V_tile
l_new = l_old * rescale + sum(exp(s - m_new))
```

**Our implementation**:
```cuda
// Phase 1: Update stats
m_tile = max(sS[m][0:kv_len])
m_old = m_smem[m]
m_new = max(m_old, m_tile)
scale_old = exp(m_old - m_new)

for (int d = 0; d < HEAD_DIM; ++d) {
    U_smem[m][d] *= scale_old;  // Rescale U
}

l_add = sum(exp(sS[m][n] - m_new) for n in 0:kv_len)
l_old = l_smem[m]
l_new = l_old * scale_old + l_add

m_smem[m] = m_new
l_smem[m] = l_new

// Phase 2: Materialize P
m_new = m_smem[m]  // Read back
for (int n = 0; n < kv_len; ++n) {
    sP[m][n] = half(exp(sS[m][n] - m_new))
}

// Later: P@V accumulates into U_smem
// Final: O = U / l
```

**Potential issues**:
- ❓ Are we applying rescale correctly?
- ❓ Is l_add computed correctly?
- ❓ Do we handle first tile (m_old = -inf, l_old = 0) correctly?

**Status**: Formula matches reference, but need to trace through manually

### Hypothesis 3: Multi-Tile Accumulation Bug
**Theory**: Issue with how we accumulate across multiple KV tiles  
**Evidence**: Error is consistent (~7-8) across runs  
**Details**:
- We have 16 KV tiles (S=512, TILE_N=32)
- Each tile updates U_smem additively (P@V)
- Each tile updates m_smem and l_smem

**Potential issues**:
- ❓ Are we resetting sS between tiles? YES (we zero it)
- ❓ Are we resetting sP between tiles? Not explicitly, but we overwrite it
- ❓ Does WMMA P@V correctly accumulate? Using atomicAdd

**Status**: Need to verify tile-by-tile accumulation

### Hypothesis 4: WMMA Store/Load Mismatch
**Theory**: wmma::store uses different layout than we expect  
**Check**:
```cuda
wmma::store_matrix_sync(&sS[warp_m_start][warp_n_start], c_frag_fp16, TILE_N, wmma::mem_row_major);
```

**Expected**: Stores to sS[warp_m_start + r][warp_n_start + c] for r,c in 0:16  
**Actual**: Need to verify with NVIDIA docs

**Status**: Should be correct for row-major, but need confirmation

### Hypothesis 5: Softmax Scale Issue
**Theory**: Scale is wrong or applied at wrong place  
**Check**:
- Scale = 1/sqrt(64) = 0.125 ✓
- Applied to c_frag_qk after WMMA Q@K^T ✓
- Stored to sS (already scaled) ✓
- Softmax uses scaled values ✓

**Status**: Looks correct

---

## 🧪 Debugging Strategy

### Step 1: Verify Q@K^T is Correct
**Goal**: Check if WMMA Q@K^T produces correct scores before softmax  
**Method**:
1. Comment out softmax and P@V
2. Store sS directly to output
3. Compare with PyTorch: `S_ref = (Q @ K.T) * scale`
4. Check max_err

**Expected**: If this passes, bug is in softmax/PV  
**If fails**: Bug is in WMMA Q@K^T or data loading

### Step 2: Verify Online Softmax Math
**Goal**: Check if softmax updates are correct  
**Method**:
1. Add debug prints for first tile:
   - m_tile, m_old, m_new, scale_old
   - l_add, l_old, l_new
2. Manually compute expected values
3. Compare

**Expected**: Values should match FlashAttention-2 formula  

### Step 3: Verify P@V Accumulation
**Goal**: Check if WMMA P@V correctly accumulates to U  
**Method**:
1. After all tiles, check U_smem values
2. Manually compute expected: U = sum over tiles of (P_tile @ V_tile)
3. Compare

**Expected**: U should match manual calculation before final normalization

### Step 4: Compare with Reference Implementation
**Goal**: Find exact point of divergence  
**Method**:
1. Read reference implementation (sdpa_fp8_stage_c_wmma.cu) more carefully
2. Look for subtle differences in:
   - Warp/thread mapping
   - Synchronization points
   - Memory access patterns
3. Match our implementation exactly

**Expected**: Find the one small difference causing the bug

---

## 📊 Known Good Values (for debugging)

**Input**: B=1, H=8, S=512, D=64  
**PyTorch SDPA**: max_err should be 0  

**Expected intermediate values** (for first CTA, first tile):
- Q[0][0:4] = random FP16 values
- K[0][0:4] = random FP16 values  
- QK[0][0] ≈ sum(Q[0] * K[0]) * 0.125
- After softmax: sum(P[0]) ≈ (portion for this tile)
- After all tiles: sum(P[0]) = 1.0
- Final: O[0][0:4] = weighted sum of V

---

## 🚀 Next Steps

1. **Quick win**: Try Step 1 (verify Q@K^T) - 15 minutes
2. **If Step 1 passes**: Focus on softmax algorithm (Step 2) - 30 min
3. **If Step 1 fails**: Debug WMMA or data loading - 1 hour
4. **Last resort**: Implement scalar version for comparison - 2 hours

---

## 💡 Insights

### What's Working ✅
1. **Kernel compiles**: No errors, good resource usage
2. **Performance is excellent**: 280 μs = 4.5× speedup!
3. **No crashes**: Runs to completion every time
4. **WMMA infrastructure**: LUT, fragments, loads/stores all compile

### What's Broken ❌
1. **Correctness**: max_err = 7.87 (should be < 0.06)
2. **Results are systematically wrong**: Not random noise, consistent error

### Key Observation
**The fact that performance is so good suggests**:
- Code structure is correct
- WMMA is being used
- Memory access patterns are reasonable  
- **BUT**: Algorithm or indexing has a subtle bug

**Most likely**: Small off-by-one error or incorrect formula application

---

## 📝 Reference Code Snippets

### Reference Online Softmax (sdpa_fp8_stage_c_wmma.cu)
```cuda
// Lines 677-710
for (int r = 0; r < WMMA_M; ++r) {
    int r_glob = warp_m + r;
    
    float m_old = m_smem[r_glob];
    float m_new = fmaxf(m_old, m_row[r]);
    float rescale = __expf(m_old - m_new);
    
    // Rescale U
    for (int d = lane; d < D; d += 32) {
        U_smem[r_glob][d] *= rescale;
    }
    
    // Compute l_add
    float l_add = 0.f;
    for (int i = 0; i < FRAG_ELEMS; ++i) {
        int rr = WMMA_ACCUM_LUT[lane][i][0];
        int cc = WMMA_ACCUM_LUT[lane][i][1];
        if (rr == r && cc < kv_local) {
            l_add += __expf(scores_local[i] - m_new);
        }
    }
    l_add = warp_reduce_sum(l_add);
    
    if (lane == 0) {
        float l_old = l_smem[r_glob];
        l_smem[r_glob] = l_old * rescale + l_add;
        m_smem[r_glob] = m_new;
    }
}
__syncwarp();
__syncthreads();

// Materialize P (lines 714-725)
for (int i = 0; i < FRAG_ELEMS; ++i) {
    int r_glob = warp_m + rr;
    int c_glob = warp_n + cc;
    if (r_glob < rows_in_tile && cc < kv_local) {
        float m_new = m_smem[r_glob];  // Read back!
        sP[r_glob][c_glob] = __float2half(__expf(scores_local[i] - m_new));
    }
}
```

**Key differences from ours**:
- They use warp-per-fragment model (we use thread-per-row)
- They read m_new from m_smem when materializing P (we do too now)
- They use __expf (we use expf - should be same)

---

**Status**: Ready for systematic debugging  
**Next session**: Start with Step 1 (verify Q@K^T)  
**Time estimate**: 1-3 hours to find and fix bug

**We're close! Just need to find that one small bug!** 🐛🔍

