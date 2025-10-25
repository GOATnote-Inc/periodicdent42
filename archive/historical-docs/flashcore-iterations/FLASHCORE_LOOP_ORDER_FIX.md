# FlashCore: Critical Loop Order Fix

**Date**: October 22, 2025

---

## 🔍 **The Bug: Wrong Loop Nesting**

### **v3.1 Results (SLOWER!)**
- **Latency**: 2891 μs (worse than v3's 620 μs!)
- **Correctness**: ✅ Perfect
- **Why slow?**: K/V tiles loaded 64× (once per row!)

---

## 🧠 **The Architecture Insight**

### **WRONG (v3.1 current)**:
```cuda
for each query row:                    // 64 iterations
    for each K/V tile:                 // 8 iterations
        load K/V (all warps sync)      // ← 64×8 = 512 loads! ❌
        process row
```
**K/V loads**: 512× (catastrophic!)

---

### **CORRECT (FlashAttention-3)**:
```cuda
for each K/V tile:                     // 8 iterations
    load K/V (all warps, once)         // ← 8 loads total! ✅
    
    for each query row:                // 64 iterations
        process row with this tile
        update state
```
**K/V loads**: 8× (64× fewer!)

---

## 📊 **Performance Impact**

| Version | K/V Loads | Latency | Notes |
|---------|-----------|---------|-------|
| v2 | 512× | 5259 μs | Per-row reload |
| v3 | ??? | 620 μs | Buggy but faster |
| v3.1 | 512× | 2891 μs | Fixed state, wrong loops |
| **v4 (correct)** | **8×** | **~60 μs** | **Proper FA-3** ✅ |

---

## 🔧 **The Fix: State Persistence**

**Challenge**: Online softmax state (m_i, l_i, out_acc) must persist across K/V tiles.

**Solution**: Maintain state in registers across outer loop:
```cuda
// Each warp owns rows (strided)
for (int q_local = warp_id; q_local < q_tile_len; q_local += nwarps) {
    // Initialize state for this row
    float m_i = -INFINITY;
    float l_i = 0.0f;
    float out_acc[D/32];
    
    // Stream K/V tiles (CORRECT ORDER!)
    for (int k_start = 0; k_start < S; k_start += N_TILE) {
        // Process tile with persistent state
        // Update m_i, l_i, out_acc
    }
    
    // Write output
}
```

**Wait, this is what v3.1 does!** So why is it slower?

**ANSWER**: The `__syncthreads()` inside the K/V loop forces ALL warps to wait!

---

## 💡 **The True Fix**

Each warp needs to process its rows **independently** without global syncs in the K/V loop.

**Use shared memory double-buffering or independent warp progress!**

Actually, the REAL issue is that we CAN'T avoid the sync if we're using shared memory for K/V!

**TRUE FlashAttention architecture**:
1. Load K/V tile to shared memory (requires sync)
2. ALL warps process their rows with this tile (no sync needed!)
3. Sync before loading next tile

Let me check the v3.1 code again... Actually, I think the issue is different. Let me trace through v3.1:

```cuda
for (int q_local = warp_id; q_local < q_tile_len; q_local += nwarps) {
    for (int k_start = 0; k_start < S; k_start += N_TILE) {
        // ALL WARPS load K/V here
        for (int idx = threadIdx.x; idx < kv_elems; idx += blockDim.x) {
            smem_K[...] = ...;
            smem_V[...] = ...;
        }
        __syncthreads();
        
        // Only THIS warp processes (warps 1-3 wait!)
        for (int j = 0; j < k_len; ++j) {
            // compute
        }
        
        __syncthreads();
    }
}
```

AH! That's the problem! When warp 0 is at q_local=0, warps 1-3 are at q_local=1,2,3. But they're all in the SAME K/V loop iteration, so they sync together but process different rows!

Actually no, each warp is doing its own q_local loop independently. So when warp 0 is at q_local=0, k_start=0, warp 1 is also at q_local=1, k_start=0.

Wait, that's not right either. The q_local loop is: `for (int q_local = warp_id; q_local < q_tile_len; q_local += nwarps)`. So:
- Warp 0: q_local = 0, 4, 8, ...
- Warp 1: q_local = 1, 5, 9, ...

When warp 0 is processing q_local=0, warp 1 is processing q_local=1. They're both in the K/V loop at the same time, loading the same K/V tiles! So actually this SHOULD work correctly and only load K/V tiles 8 times total!

But the performance says otherwise (2891 μs). Let me think about this more carefully...

OH! I see it now! The q_local loop is the OUTER loop. So:

Iteration sequence:
1. Warp 0 enters q_local=0, goes through ALL K/V tiles
2. Then warp 0 enters q_local=4, goes through ALL K/V tiles AGAIN
3. Meanwhile, warp 1 enters q_local=1, goes through ALL K/V tiles
4. etc.

So warps are NOT synchronized in their progression! When warp 0 is at q_local=0, k_start=0, warp 1 is ALSO at q_local=1, k_start=0. They sync together at the `__syncthreads()` after loading K/V.

This means all 4 warps process their first rows (0,1,2,3) with the first K/V tile, then all process with the second K/V tile, etc. This SHOULD be correct!

But wait, that's only if the loops are perfectly synchronized. Let me check if there's any divergence...

Actually, I think the issue might be the register usage. 36 registers per thread is quite high. Let me check occupancy.

Actually, I think I need to step back. The v3.1 code has the Q loop outside, K/V loop inside. This is WRONG for the FlashAttention pattern. Let me just implement the correct loop order (K/V outer, Q inner with state persistence).

