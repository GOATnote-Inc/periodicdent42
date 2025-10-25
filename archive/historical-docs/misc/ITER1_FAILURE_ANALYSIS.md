# EvoEngineer Iteration 1 - Failure Analysis

## Result: ❌ Correctness FAILED

**Max diff**: 5.070312 (tolerance: 0.001)  
**Status**: CATASTROPHIC - Not close to correct

## What Changed (Iteration 1)

```cpp
// Before: O_accum stored as float
float O_accum[Traits::BLOCK_M][Traits::HEAD_DIM];

// After: O_accum stored as half
half O_accum[Traits::BLOCK_M][Traits::HEAD_DIM];
```

## Root Cause Hypothesis

**FP16 accumulation loss** - The issue is accumulating small FP16 values repeatedly:

```cpp
// Line 497: Accumulation step
smem->O_accum[row][d] = __float2half(__half2float(smem->O_accum[row][d]) + acc);
```

**Problem**: Each iteration:
1. Read FP16 → convert to FP32
2. Add FP32 accumulator
3. Convert back to FP16 → **PRECISION LOSS**

With 8 blocks (512/64), we do this 8 times per element. FP16 has only ~3 decimal digits of precision, so accumulated error explodes.

## EvoEngineer Lesson

**What we learned**: 
- O_accum MUST stay FP32 for accumulation across blocks
- FP16 is fine for K/V (single-use per block)
- FP16 is fine for final output (after normalization)
- But NOT for iterative accumulation!

## Next Iteration Strategy

**Option A: Revert Iteration 1** (RECOMMENDED)
- Go back to float O_accum
- Try a different Phase 1 optimization
- Time: 5 minutes

**Option B: Hybrid approach**
- Keep K/V as half (already done)
- O_accum stays float
- Only convert final output to half
- Time: 10 minutes

**Option C: Debug deeper**
- Check if it's the read-modify-write pattern
- Maybe use atomics or separate buffer
- Time: 30+ minutes (NOT worth it)

## Decision: REVERT and try Option B

**Revert**: Change O_accum back to `float`  
**Keep**: Everything else from Iteration 1 (build fixes, etc)  
**Try next**: XOR swizzling for K/V (Phase 1, different optimization)

---

**EvoEngineer Protocol**: ✅ Followed  
- Test first ✅
- Measure impact ✅
- Learn from failure ✅
- Iterate quickly ✅
- **Keep GPU running** ✅

