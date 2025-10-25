# FlashCore FA-3: Correct Architecture

**Date**: October 22, 2025  
**Status**: Correctness ✅, Performance needs fix

---

## ❌ **Current (Wrong) Architecture**

```
Load Q tile once
for each warp:
    for each row assigned to this warp:
        Load K/V tile 0  ← EXPENSIVE!
        for each K/V tile:
            Process this row against tile
```

**Problem**: K/V tiles loaded **O(num_rows) times**!  
- For 64 query rows, we load each K/V tile 64 times
- This is **64× more memory traffic than needed**

---

## ✅ **Correct (FlashAttention) Architecture**

```
Load Q tile once
for each K/V tile (stream ONCE):
    Load K/V tile cooperatively
    for each warp:
        for each row assigned to warp:
            Process row against THIS tile
            Update online softmax state
```

**Benefit**: K/V tiles loaded **ONCE** and reused by all rows!  
- Same K/V tile used by all 64 query rows
- Only **1× memory traffic** (optimal)

---

## 🔧 **The Fix**

**Invert the loop nesting**:

**Before** (wrong):
```cuda
for (int q_local = warp_id; q_local < q_tile_len; q_local += nwarps) {
    // Each warp processes its assigned rows
    // Load K/V tiles for THIS row
    for (k_start = 0; k_start < S; k_start += N_TILE) {
        // Stream through K/V
    }
}
```

**After** (correct):
```cuda
for (k_start = 0; k_start < S; k_start += N_TILE) {
    // Load K/V tile ONCE for all rows
    Load K/V tile cooperatively
    __syncthreads();
    
    // All warps process their rows against THIS tile
    for (int q_local = warp_id; q_local < q_tile_len; q_local += nwarps) {
        // Process this row against current K/V tile
        // Update online softmax state
    }
}
```

**Key insight**: **Outer loop over K/V tiles, inner loop over query rows**

---

## 📊 **Expected Performance**

**Current**: 5259 μs (64× too much K/V loading)  
**After fix**: ~80-150 μs (optimal K/V reuse)

**Why**:
- K/V tiles: 128 × 64 × 2 bytes = 16 KB per tile
- Before: Load 16 KB × 64 rows × 4 tiles = 4 MB
- After: Load 16 KB × 4 tiles = 64 KB (64× reduction!)

---

## 🎯 **Implementation Plan**

1. **Outer loop**: Stream K/V tiles (load once)
2. **Inner loop**: All warps process all rows
3. **State management**: Each thread maintains (m_i, l_i, out_acc) per row
4. **Sync**: One `__syncthreads()` after K/V load, before processing

**Complexity**: State management is harder (need to maintain state across outer loop iterations)

---

**Status**: Ready to implement correct architecture! 🚀

