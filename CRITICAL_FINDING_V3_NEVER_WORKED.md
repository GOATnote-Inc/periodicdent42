# CRITICAL FINDING: V3 Baseline Never Worked

## Discovery

**ALL V3 configs fail correctness**: max_diff=5.07 (configs 1-2) or NaN (configs 3-4)

## Git History Confirms

```
36faade - "comprehensive revert status - no working V3 baseline exists"
02a0fb8 - "Evidence: Scalar baseline + root cause analysis (WMMA local memory)"
```

**V3 was never correct from the beginning!**

## Impact on Optimization Work

**What we thought**: V3 @ 38μs was a working baseline to optimize  
**Reality**: V3 produces wrong outputs, 38μs measurement was invalid

**Iteration 1 lesson**: O_accum float→half failed because BASELINE was already broken!

## EvoEngineer Protocol Violation

❌ **Missed Step**: Verify baseline correctness BEFORE optimizing  
✅ **Lesson**: Always run correctness test on baseline first

## Available Kernels (Non-V3)

```
fa_inverted.cu
fa_inverted_prod.cu  
fa_inverted_v2_tensor_cores.cu
fa_s512.cu (broken per session logs - misaligned address)
fa_s512_inverted.cu
fa_tc_s512.cu
```

## Options (User wants efficient iteration, GPU stays running)

### Option A: Test Existing Kernels (FASTEST - 15 min)
1. Try `fa_inverted_prod.cu` or `fa_tc_s512.cu`
2. If one passes correctness → use as baseline
3. Apply EvoEngineer optimizations

**Pros**: Fast, might have working code  
**Cons**: Unknown if any work

### Option B: Minimal Correct Kernel from Scratch (SAFER - 2 hours)
1. Implement simple FlashAttention (no optimizations)
2. Verify correctness
3. Then optimize systematically

**Pros**: Clean slate, we control correctness  
**Cons**: Takes longer

### Option C: Check Session Logs for Known-Good Kernel (5 min)
1. Read archived session logs
2. Find which kernel actually worked
3. Start from there

**Pros**: Fastest if docs are clear  
**Cons**: Session logs might not have working kernel

## Recommendation: Option A → C → B

1. **Try Option A** (15 min): Test `fa_tc_s512.cu` and `fa_inverted_prod.cu`
2. If none work, **try Option C** (5 min): Check `docs/archive/session_logs/` for working kernel
3. If still nothing, **do Option B** (2 hrs): Build minimal correct kernel

## Current GPU Status

**Instance**: cudadent42-l4-dev @ us-west1-c  
**Status**: ✅ RUNNING (as requested - never stop GPU)  
**Cost**: ~$0.70/hr GPU vs $100+/hr engineering time  
**Action**: Keep iterating!

---

**Next**: Test fa_tc_s512.cu for correctness (5 min)

