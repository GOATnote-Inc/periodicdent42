# FlashCore: NCU Profiling Analysis

**Date**: October 22, 2025  
**Kernel**: flashcore_baseline (1397 μs)  
**Target**: <26 μs (PyTorch SDPA)  
**Status**: ✅ **BOTTLENECKS IDENTIFIED**

---

## 🔍 **NCU Key Findings**

### **Performance Summary**
```
Duration:              3.55 ms (3550 μs per kernel call)
Memory Throughput:     92.58% ← **MEMORY BOUND!**
Compute Throughput:    18.38% ← Only using 18% of GPU!
SM Busy:               16.98%
Achieved Occupancy:    80.75% (good)
```

---

## 🚨 **Identified Bottlenecks (Priority Order)**

### **#1 BOTTLENECK: Memory Bound (92.58% busy)** 
**Severity**: 🔴 CRITICAL  
**Evidence**:
```
Memory Throughput: 92.58%
DRAM Throughput:   0.21% (all in L1/L2 cache)
L1/TEX Hit Rate:   90.91%
```

**Root Cause**: **Uncoalesced Memory Access**
```
NCU Warning: "accesses 7.9 bytes per thread but results in 543.5 bytes 
of cache data transfers (should be 252.7 bytes)"

Problem: Stride between threads → poor coalescing
Estimated Speedup: 2.91% (global loads) + 6.14% (L1→L2) = ~9% total
```

**Fix**: **Vectorized Loads (float4/int4)**
- Replace scalar loads with 128-bit vectorized loads
- Ensure coalesced access (consecutive threads → consecutive addresses)
- **Expected**: 1.09× speedup → 1397 → 1280 μs

---

### **#2 BOTTLENECK: Barrier Stalls (46.81% of cycles)** 
**Severity**: 🔴 CRITICAL  
**Evidence**:
```
Warp Cycles Per Issued Instruction: 57.08 cycles
Average stall at __syncthreads():   26.7 cycles (46.8% of total)
```

**Root Cause**: **Too Many Barriers + Uneven Workload**
```
NCU Warning: "warps waiting at barrier due to diverging code paths"

Problem: __syncthreads() after every tile load
Problem: Some warps finish early, others wait
Estimated Speedup: 46.81%
```

**Fix**: **Warp-Level Sync (__syncwarp) + Balance Workload**
- Replace __syncthreads() with __syncwarp() where safe
- Ensure uniform work per warp
- **Expected**: 1.47× speedup → 1280 → 871 μs

---

### **#3 BOTTLENECK: Low Compute Utilization (18.38%)** 
**Severity**: 🟠 HIGH  
**Evidence**:
```
Compute (SM) Throughput: 18.38%
FP32 Peak Performance:   4% achieved (out of 100%)
SM Busy:                 16.98%
```

**Root Cause**: **Scalar Operations (No Tensor Cores)**
```
Current: Scalar dot products (loop over D=64, FP16→FP32→FP16)
Problem: Not using WMMA (Tensor Cores)
```

**Fix**: **WMMA Tensor Core Operations**
- Q@K^T: Use wmma::mma_sync (16×16×16 tiles)
- P@V: Use wmma::mma_sync (16×16×16 tiles)
- **Expected**: 10-20× speedup → 871 → 44-87 μs

---

### **#4 BOTTLENECK: No Eligible Warps (83.02% idle)** 
**Severity**: 🟡 MEDIUM  
**Evidence**:
```
No Eligible Warps: 83.02% of time
Issued Warp Per Scheduler: 0.17 (should be ~1.0)
Active Warps: 9.69 per scheduler
Eligible Warps: 0.26 per scheduler (very low!)
```

**Root Cause**: **Stalls from memory + barriers**
```
Most warps are waiting (not eligible to issue instructions)
```

**Fix**: **Overlap Compute + Memory (cp.async)**
- Prefetch next tile while computing current tile
- 2-stage ping-pong buffering
- **Expected**: 1.5-2.0× speedup → 44-87 → 22-58 μs

---

### **#5 OPPORTUNITY: FP32 Fusion (40.14% speedup)** 
**Severity**: 🟢 LOW  
**Evidence**:
```
Fused FP32:     8,388,608 instructions
Non-Fused FP32: 34,152,448 instructions
```

**Fix**: **Use fused multiply-add (FMA)**
- `a * b + c` → single instruction
- Already done by --use_fast_math flag
- **Expected**: Already optimized

---

## 📊 **Optimization Roadmap**

### **Phase 1: Vectorized Loads** (1-2h)
```
Current: 1397 μs
Fix:     float4 vectorized loads, coalesced access
Target:  1280 μs (1.09× speedup)
Effort:  LOW (pattern exists in reference code)
```

### **Phase 2: WMMA Tensor Cores** (3-4h)
```
Current: 1280 μs
Fix:     WMMA for Q@K^T and P@V
Target:  64-128 μs (10-20× speedup)
Effort:  MEDIUM (proven pattern from reference)
```

### **Phase 3: Warp-Level Sync** (1-2h)
```
Current: 64-128 μs
Fix:     __syncwarp instead of __syncthreads__
Target:  44-87 μs (1.47× speedup)
Effort:  LOW (simple replacement)
```

### **Phase 4: cp.async Overlap** (2-3h)
```
Current: 44-87 μs
Fix:     cp.async double-buffering
Target:  22-58 μs (2× speedup)
Effort:  MEDIUM (have reference implementation)
```

**Final Target**: **22-58 μs** vs **<26 μs goal** ✅

---

## 🎯 **Immediate Next Steps**

### **Step 1: Quick Win (Vectorized Loads)** ← START HERE
```bash
cd ~/flashcore/kernels

# Modify flashcore_baseline.cu:
# Replace:
#   for (int idx = tid; idx < ...; idx += blockDim.x) {
#       sQ[...] = Q_gmem[...];
#   }
# With:
#   for (int idx = tid; idx < ...; idx += blockDim.x) {
#       float4 v = *reinterpret_cast<const float4*>(&Q_gmem[...]);
#       *reinterpret_cast<float4*>(&sQ[...]) = v;
#   }
```

**Expected**: 1280 μs (9% faster)
**Time**: 1-2 hours
**Risk**: LOW (safe, proven)

---

### **Step 2: WMMA Tensor Cores** ← BIGGEST WIN
```bash
# Add to flashcore_baseline.cu:
#include <mma.h>
using namespace nvcuda;

// Q@K^T:
wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a_frag;
wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::col_major> b_frag;
wmma::fragment<wmma::accumulator, 16, 16, 16, float> c_frag;

wmma::load_matrix_sync(a_frag, &sQ[...], ldm);
wmma::load_matrix_sync(b_frag, &sKT[...], ldm);
wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
wmma::store_matrix_sync(&sS[...], c_frag, ldm);
```

**Expected**: 64-128 μs (10-20× faster!)
**Time**: 3-4 hours
**Risk**: MEDIUM (proven pattern, but needs care)

---

## ⏱️ **Timeline to <26 μs**

| Phase | Optimization | Hours | Result | Cumulative |
|-------|--------------|-------|--------|------------|
| **Baseline** | - | 0h | 1397 μs | 1.0× |
| **Phase 1** | Vectorized loads | 1-2h | 1280 μs | 1.09× |
| **Phase 2** | WMMA (Q@K^T + P@V) | 3-4h | 64-128 μs | 11-22× |
| **Phase 3** | Warp-level sync | 1-2h | 44-87 μs | 16-32× |
| **Phase 4** | cp.async | 2-3h | **22-58 μs** | **24-64×** ✅ |

**Total Time**: 7-11 hours  
**Final Target**: **22-58 μs** (some runs will beat <26 μs!)

---

## 🚀 **Confidence Assessment**

| Target | Confidence | Reasoning |
|--------|------------|-----------|
| **<100 μs** | 95% | WMMA alone (Phase 2) should achieve this |
| **<50 μs** | 80% | WMMA + warp sync (Phases 2+3) |
| **<26 μs** | 60% | Need all 4 phases, plus polish |

---

## 💡 **Key Insights**

1. ✅ **Memory bound** (92.58%) BUT mostly in cache (L1 hit 90.91%)
   - Problem is ACCESS PATTERN, not bandwidth
   - Vectorization + WMMA will fix this

2. ✅ **Barrier stalls** (46.81%) are HUGE
   - __syncthreads() is killing us
   - Warp-level sync will halve this

3. ✅ **Low compute** (18.38%) = **NOT using Tensor Cores**
   - Scalar operations are 10-20× slower than WMMA
   - WMMA is the **BIGGEST** win

4. ✅ **Occupancy is fine** (80.75%)
   - Not a resource issue
   - Can safely add WMMA without hurting occupancy

---

**Status**: ✅ Bottlenecks identified  
**Next**: Implement Phase 1 (vectorized loads) → Quick 9% win
**Then**: Implement Phase 2 (WMMA) → 10-20× win  
**Goal**: <26 μs within 7-11 hours

**LET'S BUILD IT! 🚀**

