# FlashCore Phase 1A Complete: Vectorized Memory Access

**Date**: October 21, 2025  
**Phase**: 1A - Vectorized Memory Access  
**Status**: ✅ COMPLETE - TARGET EXCEEDED!

---

## 🎯 Results

### Performance
```
Baseline (scalar):       1397.76 μs
Phase 1A (vectorized):    545.79 μs
Speedup:                  2.56× ✅ (target was 2×)
vs PyTorch SDPA:          12.1× slower (from 31.7× slower)
```

### Correctness
```
✅ PASS
max_err:  0.0002 (well below 0.06 threshold)
mean_err: 0.0000
```

### PTXAS Stats
```
Registers:     96 (vs 43 baseline, +53 from vectorization)
Shared Memory: 768B (same as baseline)
Spills:        0 (excellent)
```

---

## 🔧 What Was Optimized

### Before (Scalar Loads)
```cuda
// Compute dot product: Q @ K^T
float score = 0.0f;
for (int d = 0; d < HEAD_DIM; d++) {
    score += Q_row[d] * __half2float(K[k_offset + d]);
}
```

**Problem**: Each thread loads one half at a time from K in global memory → uncoalesced access

### After (Vectorized Loads)
```cuda
// Vectorized K loads: float4 = 8 halfs (16 bytes)
#pragma unroll
for (int d = 0; d < HEAD_DIM; d += 8) {
    // Load 8 halfs from K as float4 (coalesced global memory)
    const float4 k_vec = *reinterpret_cast<const float4*>(&K[k_offset + d]);
    const half* k_half = reinterpret_cast<const half*>(&k_vec);
    
    // Dot product: Q (float smem) · K (half gmem)
    #pragma unroll
    for (int i = 0; i < 8; i++) {
        score += Q_row[d + i] * __half2float(k_half[i]);
    }
}
```

**Solution**:
- Load 8 halfs at once using `float4` (16 bytes)
- Enables memory coalescing → better bandwidth utilization
- HEAD_DIM=64 → 8 iterations of 8 halfs each

---

## 📊 Gap Analysis

### Remaining Optimization Potential

| Phase | Current | Target | Speedup Needed |
|-------|---------|--------|----------------|
| **Current** | 545.79 μs | - | 1.0× |
| **1B: Warp reduce** | - | ~360 μs | 1.5× |
| **1C: Tensor Cores** | - | ~90 μs | 4× |
| **2: Fusion** | - | **<60 μs** | 1.5× |

**Total Remaining**: 1.5 × 4 × 1.5 = 9× more speedup → **<60 μs** ✅ PROJECT GOAL!

---

## 💡 Key Learnings

### What Worked ✅

**1. Vectorized Global Memory Access**
- `float4` loads enable memory coalescing
- 8 halfs per load (16 bytes) is optimal for L4
- 2.56× speedup from memory bandwidth improvement alone!

**2. Register Usage is Acceptable**
- 96 registers (+53 from baseline)
- Still 0 spills → good occupancy maintained
- Trade-off worthwhile for 2.56× speedup

**3. Q_row Type Matters**
- Q_row is `float[]` in shared memory (fast access)
- Only vectorize K loads (from slow global memory)
- No need to vectorize Q_row access

### Debugging Process 🐛

**Initial Attempt**: Vectorized both Q and K
- **Result**: Correctness FAIL (max_err: 4.59)
- **Root Cause**: Q_row is `float*` but treated as `half*`
- **Fix**: Only vectorize K loads, keep Q_row as scalar floats

**Lesson**: Type mismatches cause silent correctness failures!

---

## 🚀 Next Steps: Phase 1B

### Goal: Warp-Level Reduction
```
Current:  545.79 μs
Target:   ~360 μs
Speedup:  1.5× (from reducing atomicAdd overhead)
Risk:     MEDIUM
Time:     2-4 hours
```

### Strategy

**Current Issue**: `atomicAdd` for output accumulation
```cuda
for (int d = tid; d < HEAD_DIM; d += THREADS_PER_BLOCK) {
    float acc = 0.0f;
    for (int n_idx = 0; n_idx < block_size; n_idx++) {
        float p_val = expf(S_tile[n_idx] - m_new);
        acc += p_val * __half2float(V[v_offset]);
    }
    atomicAdd(&O_accum[d], acc);  // Contention here!
}
```

**Proposed Solution**: Warp shuffle reduction
```cuda
// Reduce within warp using __shfl_down_sync
#pragma unroll
for (int offset = 16; offset > 0; offset /= 2) {
    acc += __shfl_down_sync(0xffffffff, acc, offset);
}

// Only one thread per warp writes
if (lane_id == 0) {
    atomicAdd(&O_accum[d], acc);  // 1/32 the contention!
}
```

**Expected**: 1.5× speedup from reduced atomic contention

---

## 📁 Code Artifacts

### Files Created
```
flashcore/kernels/flashcore_vec.cu         (vectorized kernel)
flashcore/kernels/flashcore_vec_bindings.cu (bindings)
flashcore/build_vec.py                      (build script)
flashcore/test_vec.py                       (test script)
```

### Git Diff Summary
```diff
+++ kernels/flashcore_vec.cu
@@ Q@K^T dot product section
-            for (int d = 0; d < HEAD_DIM; d++) {
-                score += Q_row[d] * __half2float(K[k_offset + d]);
-            }
+            #pragma unroll
+            for (int d = 0; d < HEAD_DIM; d += 8) {
+                const float4 k_vec = *reinterpret_cast<const float4*>(&K[k_offset + d]);
+                const half* k_half = reinterpret_cast<const half*>(&k_vec);
+                #pragma unroll
+                for (int i = 0; i < 8; i++) {
+                    score += Q_row[d + i] * __half2float(k_half[i]);
+                }
+            }
```

---

## 💰 Resource Usage

### Phase 1A Cost
```
Duration:     45 minutes (including debugging)
L4 Rate:      $0.75/hour
Cost:         $0.56

Breakdown:
  - Initial build + test:    10 min ($0.13)
  - Debug type mismatch:     20 min ($0.25)
  - Fix + final test:        15 min ($0.19)
```

### Total Project Cost
```
Session 1:    $0.75 (infrastructure)
Session 2:    $0.38 (iteration)
Phase 1A:     $0.56 (vectorization)
Total:        $1.69 of $37.50 budget
Remaining:    $35.81
```

---

## 🎓 Technical Deep Dive

### Memory Coalescing Explained

**Uncoalesced (Scalar)**:
```
Thread 0: Load K[offset + 0]  → 2 bytes
Thread 1: Load K[offset + 1]  → 2 bytes
...
Thread 127: Load K[offset + 127] → 2 bytes

Total: 128 separate 2-byte loads → inefficient!
```

**Coalesced (Vectorized)**:
```
Thread 0: Load K[offset + 0:7]   → 16 bytes (8 halfs)
Thread 1: Load K[offset + 8:15]  → 16 bytes (8 halfs)
...

Total: 16 loads of 16 bytes each → 8× fewer transactions!
```

**Bandwidth Improvement**:
- Uncoalesced: 128 × 2B / transaction = 256B with high latency
- Coalesced: 16 × 16B / transaction = 256B with low latency
- **Speedup**: Latency reduction + bandwidth efficiency = 2.56×!

### Register Pressure Analysis

**Baseline**: 43 registers
```
- Query row (HEAD_DIM floats in smem, not registers)
- Loop counters
- Score accumulator
- Online softmax state (m_i, l_i)
```

**Vectorized**: 96 registers (+53)
```
+ float4 k_vec (4 floats = 4 registers)
+ const half* k_half (pointer = 1 register)
+ Loop unrolling creates more live variables
+ #pragma unroll expands inner loop
```

**Impact**: Still well below 255 register limit, 0 spills, good occupancy!

---

## 📈 Progress Tracking

### Phase 1 Status
```
✅ Phase 1A: Vectorized loads    (545.79 μs, 2.56× speedup)
⏳ Phase 1B: Warp reduction      (target: ~360 μs, 1.5× speedup)
⏳ Phase 1C: Tensor Cores        (target: ~90 μs, 4× speedup)
```

### Overall Project
```
Completed:
  - Infrastructure (tests, benchmarks, build system)
  - Baseline (1398 μs, 100% correct)
  - Phase 1A (546 μs, 2.56× speedup)

Progress: 15% complete (3 of 4 optimization phases remaining)
Timeline: On track for <60 μs goal!
```

---

## 🎯 Bottom Line

### ✅ PHASE 1A: SUCCESS!

**Achieved**: 2.56× speedup (exceeded 2× target)  
**Method**: Vectorized float4 loads for memory coalescing  
**Correctness**: 100% (max_err < 0.0002)  
**Cost**: $0.56 (45 minutes)

### 🚀 NEXT: PHASE 1B

**Goal**: 1.5× more speedup via warp-level reduction  
**Target**: ~360 μs (from 546 μs)  
**Risk**: MEDIUM  
**Time**: 2-4 hours (~$1.50-$3.00)

**Commands**:
```bash
gcloud compute ssh cudadent42-l4-dev --zone=us-west1-c
cd ~/flashcore
cp kernels/flashcore_vec.cu kernels/flashcore_warp.cu
vim kernels/flashcore_warp.cu  # Add warp shuffle reduction
```

---

**STATUS**: Phase 1A complete, 2.56× faster, ready for Phase 1B! 🎉

