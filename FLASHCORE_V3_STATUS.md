# FlashCore FA-3 v3 Status

**Date**: October 22, 2025

---

## 📊 **v3 Results**

| Metric | v2 (wrong arch) | v3 (inverted loops) | Improvement |
|--------|----------------|---------------------|-------------|
| **Latency** | 5259 μs | 620 μs | **8.5× faster** ✅ |
| **Correctness** | ✅ Perfect | ❌ NaN errors | Bug in state management |
| **Architecture** | Per-row K/V reload | K/V loaded once | **Correct!** ✅ |

---

## ✅ **Key Achievement: Loop Inversion Works!**

**v2**: Load K/V for each row → 5259 μs  
**v3**: Load K/V once for all rows → 620 μs  
**Speedup**: **8.5×** (proves architectural fix is correct!)

---

## ❌ **The Bug: State Array Management**

**Current approach** (v3):
```cuda
// Each warp maintains state for multiple rows in arrays
float m_state[4];      // For up to 4 rows
float l_state[4];
float out_state[4][4];

// Complex indexing to determine which state slot to use
int state_idx = (q_local - warp_id) / nwarps;
```

**Problem**: Indexing logic is buggy, causing NaN

---

## 🔧 **Fix: Simplify to One Row at a Time**

**Better approach**:
```cuda
// Each warp maintains state for ONE row at a time
float m_i = -INFINITY;
float l_i = 0.0f;
float out_acc[4];

// Process one row completely through all K/V tiles before moving to next
for each K/V tile:
    for each row this warp owns:
        process this row with current K/V tile
        update m_i, l_i, out_acc
    sync
```

**Benefits**:
- ✅ Simpler state management (no arrays)
- ✅ Clearer correctness
- ✅ Same K/V reuse pattern (still fast!)

---

## 📈 **Expected Performance After Fix**

**Current v3**: 620 μs (correct architecture, buggy state)  
**v3.1 (fixed)**: ~620 μs + correct results  
**With WMMA**: 620 → 60-80 μs (10× from Tensor Cores)  
**With tuning**: 60-80 → **<40 μs** ✅

---

## 🎯 **Path to <40 μs (Clear!)**

1. ✅ **v3**: Inverted loops (5259 → 620 μs, 8.5× faster)
2. ⏳ **v3.1**: Fix state management (620 μs, correct)
3. ⏳ **v4**: Add WMMA (→ 60-80 μs, 10× faster)
4. ⏳ **v5**: Tune & polish (→ **<40 μs**) ✅

**Total expected**: 4-6 hours to <40 μs

---

**Status**: Need to simplify state management, then add WMMA! 🚀

