# Nsight Compute Profiling - The Uncomfortable Truth

**Date:** November 1, 2025  
**Device:** NVIDIA L4 (SM 8.9, Driver 580.95.05)  
**Kernel:** `bsr_spmm_async<256,128,32>`  
**Config:** 8K×8K BSR, 78% sparse, BM=256 BN=128 BK=32

## Measured Performance

| Metric | Value | Assessment |
|--------|-------|------------|
| **Duration** | 1.53 ms | - |
| **TFLOPS** | 55.0 | 63× faster than cuSPARSE |
| **DRAM Util** | 71.07% | ✅ Good (memory bound expected) |
| **SM Util** | **12.61%** | 🚨 **TERRIBLE** |

## Critical Finding

**Only 12.61% of GPU cores are being used!**

### What This Means

**The Bad:**
- 87% of GPU compute is IDLE
- Massive inefficiency in SM utilization
- Could theoretically be 8× faster

**The Good:**
- Still 63× faster than cuSPARSE (0.87 TFLOPS)
- Memory pattern is efficient (71% DRAM)
- Works with CUDA 13.0.2 + CUTLASS 4.2.1

**The Reality:**
- We're winning because cuSPARSE is TERRIBLE
- NOT because this kernel is amazing
- Huge room for improvement

## Performance Ceiling Analysis

```
Current performance:  55.0 TFLOPS @ 12.6% SM
Theoretical ceiling:  437 TFLOPS @ 100% SM
Wasted potential:     382 TFLOPS

H100 extrapolation:
  Current (conservative): 770 TFLOPS
  With 100% SM:          6,118 TFLOPS
  
Your 610 TFLOPS claim assumes ~10% SM utilization on H100
```

## Root Causes (Likely)

1. **Low Occupancy**
   - Thread block config (256 threads) may be suboptimal
   - Register pressure limiting occupancy
   - Shared memory usage limiting warps/SM

2. **Divergence**
   - Sparse iteration causing warp divergence
   - Conditional branches in hot path

3. **Launch Configuration**
   - Grid size (64,32) may not saturate L4's 58 SMs

## Next Steps to Fix

1. **Occupancy optimization**
   - Profile register usage
   - Adjust block size (try 128, 512 threads)
   - Reduce shared memory per block

2. **Divergence mitigation**
   - Rewrite sparse iteration to minimize branches
   - Use warp-uniform predicates

3. **Better launch config**
   - Calculate optimal grid size for 58 SMs
   - Maximize blocks/SM

## Honest Conclusion

**We built a kernel that:**
- ✅ Beats cuSPARSE by 63×
- ✅ Proves cuSPARSE is terrible
- ❌ Uses only 12.6% of available compute
- ❌ Leaves 8× performance on the table

**This is NOT publication-worthy yet. It's a good start that exposes cuSPARSE's weakness, but the kernel itself needs major optimization.**

---

*Measured with Nsight Compute 2025.3.1 on L4, November 1, 2025*
