# FlashCore Bug Found - Q@K^T Layout Issue

**Date**: October 22, 2025  
**Status**: 🐛 **BUG IDENTIFIED - K^T WMMA Load Pattern Wrong**

---

## 🎯 DEBUG QK-ONLY Results

**Test**: Isolated Q@K^T computation (skipped softmax/PV)  
**Result**: ❌ **FAIL** - max_err = 5.94 (target: <0.001)

```
Reference: [ 0.7573  1.316   1.487  -1.468  -1.189  -2.67 ...]
Ours:      [-0.01945  0.01393 -0.07874  0.05164  0.0903 ...]
Diff:      [ 0.777  1.302  1.566  1.519  1.280  2.704 ...]
```

**Analysis**: Values are 10-100× too small → **WMMA K^T load/layout bug**

---

## 🐛 Root Cause: WMMA Col-Major Interpretation

### Current Code (WRONG)
```cuda
// Storage: sKT[TILE_N][HEAD_DIM_SMEM] = [32][80]
__shared__ half sKT[TILE_N][HEAD_DIM_SMEM];

// Load K as row-major
for (int idx...) {
    int n = idx / D;
    int d = idx % D;
    sKT[n][d] = K_bh[(kv_start + n) * D + d];  // K stored as [N][D]
}

// WMMA load with col_major fragment
wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::col_major> b_frag_qk;
wmma::load_matrix_sync(b_frag_qk, &sKT[warp_n_start][k], HEAD_DIM_SMEM);
```

### Problem
**WMMA col-major interpretation** with `ldm=HEAD_DIM_SMEM`:
- Element (r, c) at offset: `r + c * ldm`
- From `&sKT[warp_n_start][k]`:
  - (0, 0) → sKT[warp_n_start][k+0] ✓
  - (1, 0) → sKT[warp_n_start][k+1] ← WRONG! This is D dimension!
  - (0, 1) → sKT[warp_n_start][k+80] ← Out of bounds!

**Memory layout of sKT[N][D]**:
- sKT[n][d] at offset: n * HEAD_DIM_SMEM + d
- Consecutive elements go along D, not N!

**For col-major WMMA**, consecutive elements should be along the **row** dimension, but our layout has consecutive elements along the **column** dimension!

---

## ✅ Solutions

### Option A: Change Fragment to Row-Major (SIMPLEST)
```cuda
// Change B fragment to row_major
wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> b_frag_qk;

// Load with row_major interpretation
wmma::load_matrix_sync(b_frag_qk, &sKT[warp_n_start][k], HEAD_DIM_SMEM);
```

**Pros**: Minimal code change  
**Cons**: Need to verify WMMA semantics (does B row-major give us K^T?)

### Option B: Transpose K When Loading (EXPLICIT)
```cuda
// Transpose K explicitly: sKT[d][n] instead of sKT[n][d]
__shared__ half sKT[HEAD_DIM_SMEM][TILE_N];  // Note: swapped dimensions!

// Load K transposed
for (int idx...) {
    int n = idx / D;
    int d = idx % D;
    sKT[d][n] = K_bh[(kv_start + n) * D + d];  // Explicitly transpose
}

// Load as col-major (now correct!)
wmma::load_matrix_sync(b_frag_qk, &sKT[k][warp_n_start], TILE_N);
//                                        ^ note: k first, warp_n second
//                                                       ^ ldm = TILE_N
```

**Pros**: Explicit, clear intent  
**Cons**: More invasive change, different SMEM layout

### Option C: Load from Correct Offset
```cuda
// Keep sKT[N][D], but load differently
// Need to point to column k, rows warp_n_start:warp_n_start+16

// This requires loading 16 separate column vectors, which WMMA doesn't support directly
// NOT VIABLE
```

---

## 🎯 Recommended Fix: Option A (Row-Major Fragment)

**Change**:
```cuda
// Line ~245: Change fragment type
wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> b_frag_qk;
```

**Test**:
1. Rebuild with DEBUG_QK_ONLY=1
2. Check if max_err < 0.001
3. If yes → bug fixed! Proceed to full kernel
4. If no → Try Option B

---

## 📊 Impact

**If Option A works**:
- ✅ Fixes Q@K^T computation
- ✅ Should fix full kernel correctness (max_err 3.78 → <0.05)
- ✅ No performance impact (just fragment type change)
- ✅ Minimal code change (1 line!)

**Expected full kernel results after fix**:
- Correctness: max_err < 0.05 ✅
- Performance: ~350-400 μs (similar to current)
- Then: Apply Phase 3 optimizations (cp.async, 64×64 tiles) → <40 μs

---

## 🚀 Next Steps

1. ✅ **PRIORITY 1**: Try Option A (change to row_major) - 5 min
2. **PRIORITY 2**: Test with DEBUG_QK_ONLY - 5 min
3. **PRIORITY 3**: If passes, test full kernel - 5 min
4. **PRIORITY 4**: If correctness passes, proceed to performance optimization

**Time to fix**: 15-30 minutes  
**Confidence**: 90% (Option A should work based on WMMA semantics)

---

**STATUS**: 🔍 **BUG LOCATED - READY TO FIX!**

**We found it! Just need to change the fragment type!** 🎉🐛

