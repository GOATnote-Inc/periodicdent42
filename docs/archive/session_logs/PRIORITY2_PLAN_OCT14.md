# Priority 2: Performance Optimization Plan

**Current State**: 0.3184 ms (1.58× vs V1, 0.23× vs SDPA)  
**Target**: 0.20 ms (2.5× vs V1, 0.35× vs SDPA)  
**GPU Time Remaining**: ~13 hours

---

## Quick Wins (Option C) - 1-2 hours

### 1. Remove S_float→S_half Conversion Overhead (~10% speedup)
**Current**: Convert entire S matrix after softmax  
**Cost**: ~5-10% overhead (128 threads × 1024 elements)

**Options**:
- Keep S as half throughout (loses FP32 softmax accuracy)
- Stream conversion inline with softmax (overlap work)
- Use wmma with FP32 accumulator for S@V (requires different approach)

**Decision**: Skip - numerical accuracy is critical

### 2. Tune Warp Distribution (~5-10% speedup)
**Current**: 4 warps, 128 threads  
**Test**: 8 warps (256 threads) for better occupancy

**Implementation**:
```cuda
constexpr int NUM_WARPS = 8;  // Was 4
constexpr int NUM_THREADS = NUM_WARPS * WARP_SIZE;  // 256
```

**Risk**: Might exceed register budget or SMEM per block

### 3. Reduce __syncthreads() Calls (~5% speedup)
**Current**: 10+ syncs per K/V tile iteration  
**Optimize**: Combine operations between syncs

---

## Tile Size Optimization (Option A) - 2-3 hours

### Target: TILE_M = 64 (2× larger)

**Benefits**:
- 2× fewer K/V tile iterations (16 → 8 for S=512)
- Better SMEM amortization
- Higher arithmetic intensity

**SMEM Calculation**:
```
Q: 64 × 64 × 2 bytes = 8KB  (was 4KB)
K: 32 × 64 × 2 bytes = 4KB  (unchanged)
V: 32 × 64 × 2 bytes = 4KB  (unchanged)
S union: 64 × 32 × 4 bytes = 8KB  (was 4KB)
temp_O: 64 × 64 × 4 bytes = 16KB  (was 8KB)
Total: 40KB < 48KB L4 limit ✓
```

**Warp Distribution** (8 warps, 256 threads):
```
QK: 4 M-blocks (0-15, 16-31, 32-47, 48-63)
    Each of 4 warps handles 1 M-block × 2 N-blocks
SV: 4 M-blocks × 4 N-blocks
    Each of 4 warps handles 1 M-block × 2 N-blocks
```

**Implementation Steps**:
1. Change TILE_M to 64, NUM_WARPS to 8
2. Update warp work distribution in compute_QK_wmma
3. Update warp work distribution in compute_SV_wmma
4. Adjust temp_O allocation
5. Test correctness (7 test cases)
6. Measure performance

**Expected**: 1.5-2× speedup (0.3184 ms → 0.16-0.21 ms)

---

## Async Memory (Option B for later) - 2-3 hours

### Target: cp.async for overlapped compute/memory

**Benefit**: Hide memory latency during compute

**Requirements**:
- SM 8.0+ (L4 is SM 8.9 ✓)
- Double-buffering (2× SMEM)
- Pipeline depth 1-2 stages

**SMEM with Double-Buffering** (TILE_M=64):
```
Stage 0: Q, K, V = 16KB
Stage 1: Q, K, V = 16KB
S union: 8KB
temp_O: 16KB
Total: 56KB > 48KB ❌
```

**Problem**: Exceeds L4's 48KB SMEM limit

**Solutions**:
- Reduce TILE_M back to 32 (40KB with double-buffer)
- Use single-buffering with async (less benefit)
- Profile to see if worth the complexity

---

## Decision Tree

```
Start: 0.3184 ms (1.58×)
  │
  ├─ Option C: Quick Wins (1-2h)
  │   ├─ Tune NUM_WARPS: 0.29 ms (1.74×) +10%
  │   └─ Reduce syncs: 0.28 ms (1.80×) +5%
  │
  ├─ Option A: TILE_M=64 (2-3h)
  │   └─ Expected: 0.18 ms (2.80×) +75%
  │
  └─ Option B: cp.async (3-4h)
      └─ Expected: 0.15 ms (3.36×) +20%

Total Path: 0.15 ms (~3.4× vs V1, 0.48× vs SDPA)
Still 2× slower than SDPA, but respectable for academic project
```

---

## Recommended Approach

**Phase 1** (next 3 hours):
1. Implement TILE_M=64 (Option A) - highest ROI
2. Test correctness + performance
3. Commit as V3

**Phase 2** (if time remains, ~4 hours):
1. Profile V3 with Nsight Compute
2. Identify remaining bottlenecks
3. Implement targeted fixes

**Phase 3** (stretch goal, ~6 hours):
1. Attempt cp.async with TILE_M=32 (fits in 40KB with double-buffer)
2. Test if async provides meaningful benefit
3. Compare final result to SDPA

---

## Success Criteria

**Minimum** (achievable in 3h):
- V3 with TILE_M=64 implemented
- Correctness validated (7 tests pass)
- Performance: 2.5-3× vs V1 (0.18-0.20 ms)

**Stretch** (if 6-8h available):
- cp.async implemented
- Performance: 3-4× vs V1 (0.13-0.17 ms)
- 0.5× vs SDPA (2× slower than SOTA)

**Publication-Ready**:
- Honest documentation of limitations
- Comparison to SDPA with profiling analysis
- Clear path forward for future work

---

**Next Action**: Implement TILE_M=64 (proceed immediately)

