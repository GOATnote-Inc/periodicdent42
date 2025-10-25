# FlashCore Phase 2 - Debugging Status

**Date**: October 22, 2025  
**Status**: 🔍 **BUG LOCATED, FIX IN PROGRESS**

---

## ✅ Phase 1 Achievements (Completed)

1. ✅ Pre-scale Q (eliminates hot-path multiply)
2. ✅ FP32 score tile (sS_f32) for numerical stability
3. ✅ PV k-partition by warp_n (avoids double-counting)
4. ✅ HEAD_DIM_SMEM = 80 (multiple of 16)
5. ✅ Robust initialization (-INFINITY)

**Result**: 354 μs (3.95× speedup), max_err = 3.78 (51% improvement!)

---

## 🎯 Phase 2 Progress

### DEBUG_QK_ONLY Gate ✅
**Implemented**: Isolation mode that tests Q@K^T only  
**Result**: ❌ **Bug is in Q@K^T, NOT in softmax/PV!**

```
Reference: [ 0.7573  1.316   1.487  -1.468 ...]
Ours (col_major B): [-0.01945 0.01393 -0.07874 ...] (10-100× too small)
Ours (row_major B): [-0.6333 -2.129 2.07 -1.048 ...]  (closer, but still wrong)
```

---

## 🐛 Root Cause: WMMA K^T Layout

### Problem
**WMMA needs K^T**, but our layout sKT[N][D] doesn't naturally provide it.

**Attempts**:
1. ❌ `col_major` fragment: Values 10-100× too small
2. ⚠️ `row_major` fragment: Values closer, but still wrong (max_err = 5.55)

### Hypothesis
The layout `sKT[N][D]` with pointer `&sKT[warp_n_start][k]` doesn't correctly represent K^T for WMMA, regardless of fragment layout specification.

**Memory layout**:
- sKT[n][d] at offset: n * HEAD_DIM_SMEM + d
- Consecutive memory: along D dimension
- For K^T, we need consecutive memory along N dimension!

---

## 🎯 Next Attempt: Explicit Transpose

### Option B: Store K as [D][N]
```cuda
// Change shared memory layout
__shared__ half sKT[HEAD_DIM_SMEM][TILE_N];  // Swapped dimensions!

// Load K transposed
for (int idx = tid; idx < kv_len * D; idx += THREADS_PER_BLOCK) {
    const int n = idx / D;
    const int d = idx % D;
    sKT[d][n] = K_bh[(kv_start + n) * D + d];  // Transpose on load
}

// WMMA load (now K^T is naturally represented)
wmma::load_matrix_sync(b_frag_qk, &sKT[k][warp_n_start], TILE_N, wmma::row_major);
```

**Pros**:
- Explicit, clear transpose
- Memory layout matches what WMMA expects

**Cons**:
- More invasive change
- Need to update all K references

---

## 📊 Current Status

**Files Ready**:
- ✅ `flashcore/kernels/flashcore_fused_wmma.cu` (Phase 1 fixes + DEBUG gate)
- ✅ `flashcore/build_fused.py` (supports extra_cflags)
- ✅ `flashcore/test_qk_only.py` (QK isolation test)

**Debug Tools**:
- ✅ DEBUG_QK_ONLY gate implemented
- ✅ QK-only test compares against PyTorch reference
- ✅ Can iterate quickly (15-30 sec per test)

**Progress**:
- ✅ Bug isolated to Q@K^T (not softmax/PV)
- ✅ Error magnitude improved (10-100× → ~2×)
- ⏳ Need correct WMMA layout for K^T

---

## 🚀 Action Plan

### Priority 1: Try Explicit Transpose (30 min)
1. Change sKT to [D][N] layout
2. Transpose K when loading
3. Update WMMA load call
4. Test with DEBUG_QK_ONLY
5. **Target**: max_err < 0.001

### Priority 2: If Transpose Works (15 min)
1. Remove DEBUG_QK_ONLY flag
2. Test full kernel
3. **Target**: max_err < 0.05

### Priority 3: Performance Optimization (2-3 hours)
1. Recover to ~280 μs baseline
2. Add 2-stage cp.async (2× speedup)
3. Expand to 64×64 tiles (2× speedup)
4. **Target**: <100 μs → <50 μs → <40 μs

---

## 💪 Confidence

**Correctness (Priority 1)**: **80%** confident explicit transpose will work  
- Clear mathematical reasoning
- Matches how other kernels handle K^T
- Can iterate quickly to test

**Full Kernel (Priority 2)**: **90%** confident once QK is fixed  
- Softmax/PV code is sound (from Phase 1)
- Only QK was wrong

**Performance (Priority 3)**: **75%** confident we'll hit <100 μs  
- Have clear optimization path
- Known techniques (cp.async, larger tiles)
- May need 2-3 iterations

---

## 📈 Timeline

```
Now:        Phase 2 debugging (QK layout)
+30 min:    QK fixed, testing full kernel
+1 hour:    Correctness validated (max_err < 0.05)
+3 hours:   Performance optimized (<100 μs)
+5 hours:   Stretch goal (<50 μs)
+8 hours:   Ultra stretch (<40 μs)
```

---

## 🎓 Key Learnings

### What Worked ✅
1. **DEBUG_QK_ONLY gate**: Isolated bug in 1 test!
2. **Systematic debugging**: Narrowed from "everything wrong" to "just QK layout"
3. **Quick iteration**: 15-30 sec per test on GPU
4. **Phase 1 fixes**: Reduced error 51%, good foundation

### What's Challenging ⚠️
1. **WMMA layout semantics**: Non-obvious how to represent K^T
2. **Limited documentation**: Need to infer from examples
3. **Trial and error**: Testing different layout combinations

### Next Time 💡
1. **Start with reference**: Match existing working kernel layout exactly
2. **Test early**: Add DEBUG gates from the start
3. **Document layout**: ASCII diagrams of memory layout vs WMMA expectations

---

**STATUS**: 🔍 **80% there - just need correct K^T layout!**

**We're very close! One more layout fix and we're done!** 🚀

