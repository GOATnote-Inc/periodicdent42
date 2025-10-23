# FlashCore v12 Optimization Plan

**Current**: 1512 µs (51× slower than SDPA @ 29 µs)  
**Target**: <60 µs (2× SDPA, realistic first milestone)  
**Final Target**: <28 µs (excellence)

---

## 🔍 Root Cause Analysis

### Current v12 Issues

```
❌ 4 barriers per KV tile:
   1. After KV load
   2. After QK^T compute
   3. After softmax
   4. After P·V compute
   
❌ All 512 threads loading KV (not just load warps)
❌ No warp-local sync (all use __syncthreads())
❌ Compute warps wait for everyone
```

### Performance Breakdown (Estimated)

```
Per KV tile (11 tiles for S=512):
- 4× __syncthreads() = ~4 µs overhead
- Redundant loads = ~10 µs wasted
- Total waste = ~14 µs/tile × 11 tiles = 154 µs

Actual work (WMMA + softmax): ~100 µs
Overhead: ~1400 µs
Total: ~1500 µs ✅ matches observed
```

---

## 🎯 Optimization Strategy

### Phase 1: Reduce Barriers (Target: 400-600 µs)

**Remove 3 of 4 barriers per tile**:

```cpp
// BEFORE (v12 current):
for (kv_tile) {
    load_kv();           // all threads
    __syncthreads();     // ❌ barrier 1
    
    compute_qkt();       // compute warps
    __syncthreads();     // ❌ barrier 2
    
    softmax();           // softmax warp
    __syncthreads();     // ❌ barrier 3
    
    compute_pv();        // compute warps
    __syncthreads();     // ❌ barrier 4
}

// AFTER (v12 optimized):
for (kv_tile) {
    if (is_load) {
        load_kv();       // only load warps
    }
    __syncthreads();     // ✅ single barrier
    
    if (is_compute) {
        compute_qkt();
        __syncwarp();    // warp-local
    }
    
    if (is_softmax) {
        softmax();
        __syncwarp();    // warp-local
    }
    
    if (is_compute) {
        compute_pv();
        __syncwarp();    // warp-local
    }
    // NO barrier here - next iteration does it
}
```

**Expected**: 1512 → 400-600 µs (2.5-4× speedup)

---

### Phase 2: Warp-Specialized Loads (Target: 200-300 µs)

**Only load warps load data**:

```cpp
if (is_load) {
    // 4 load warps, each loads 1/4 of KV tile
    const int warp_load_id = warp_id - kComputeWarps;
    const int rows_per_warp = (kv_len + 3) / 4;
    const int row_start = warp_load_id * rows_per_warp;
    const int row_end = min(row_start + rows_per_warp, kv_len);
    
    for (int row = row_start + lane_id; row < row_end; row += kWarpSize) {
        // Load K row
        // Load V row
    }
}
```

**Expected**: 400-600 → 200-300 µs (2× speedup from Phase 1)

---

### Phase 3: Vectorized Loads (Target: 150-200 µs)

**Use uint4 (128-bit) loads**:

```cpp
if (is_load && D % 8 == 0) {
    for (int row = ...; row < row_end; row++) {
        for (int col = lane_id * 8; col < D; col += kWarpSize * 8) {
            uint4 k_vec = *reinterpret_cast<const uint4*>(&K_bh[row * D + col]);
            *reinterpret_cast<uint4*>(&layout.k_tiles[stage][row * kTilePadD + col]) = k_vec;
        }
    }
}
```

**Expected**: 200-300 → 150-200 µs (1.5× speedup from Phase 2)

---

### Phase 4: cp.async Prefetch (Target: 80-120 µs)

**Double-buffer with cp.async**:

```cpp
// Prefetch first tile
if (is_load) {
    cp_async_load(stage=0, tile=0);
}
cp_async_commit();

for (kv_tile_idx = 0; kv_tile_idx < num_kv_tiles; kv_tile_idx++) {
    const int curr_stage = kv_tile_idx % 2;
    const int next_stage = 1 - curr_stage;
    
    // Prefetch next tile while computing current
    if (is_load && kv_tile_idx + 1 < num_kv_tiles) {
        cp_async_load(next_stage, tile=kv_tile_idx+1);
        cp_async_commit();
    }
    
    cp_async_wait<0>();  // Wait for current
    
    if (is_compute) {
        compute_qkt(curr_stage);
        compute_pv(curr_stage);
    }
    if (is_softmax) {
        softmax(curr_stage);
    }
    
    __syncthreads();  // Single barrier
}
```

**Expected**: 150-200 → 80-120 µs (2× speedup from Phase 3)

---

### Phase 5: Register Blocking + Tuning (Target: <60 µs)

**Optimizations**:
- Keep Q in registers (no reload from SMEM)
- Reduce SMEM round-trips
- Tune launch bounds
- Profile with NCU

**Expected**: 80-120 → 40-60 µs (2× speedup from Phase 4)

---

## 📊 Cumulative Speedup Roadmap

| Phase | Optimization | Latency (µs) | vs SDPA | Speedup |
|:------|:-------------|-------------:|--------:|--------:|
| v12 Baseline | 4 barriers, all load | 1512 | 51.8× | 1.0× |
| Phase 1 | Reduce to 1 barrier | 500 | 17.1× | 3.0× |
| Phase 2 | Warp-specialized loads | 250 | 8.6× | 6.0× |
| Phase 3 | Vectorized loads | 175 | 6.0× | 8.6× |
| Phase 4 | cp.async prefetch | 100 | 3.4× | 15.1× |
| Phase 5 | Register blocking | 50 | 1.7× | 30.2× |

**Final Target**: 29-50 µs (parity to 2× SDPA)

---

## 🔧 Implementation Order

### Immediate (30 min)
**Phase 1**: Remove 3 barriers → test → should see 3× speedup

### Next (1 hour)
**Phase 2**: Warp-specialized loads → test → should see 6× cumulative

### Follow-up (2 hours)
**Phase 3-4**: Vectorization + cp.async → test → should see 15× cumulative

### Final (4 hours)
**Phase 5**: Full optimization → NCU profile → tune → target <60 µs

---

## ✅ Success Criteria

**Minimum**: <100 µs (10× speedup from baseline)  
**Good**: <60 µs (2× SDPA, 25× speedup from baseline)  
**Excellence**: <28 µs (parity with SDPA, 54× speedup from baseline)

**Current**: v12 @ 1512 µs  
**Next Milestone**: v12 Phase 1 @ ~500 µs (target: next 30 minutes)

---

**NO QUITTING. Optimizing Phase 1 NOW! 🚀**

