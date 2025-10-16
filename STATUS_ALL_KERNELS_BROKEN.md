# STATUS: All Tested Kernels Fail Correctness

## GPU: Running (us-west1-c) ✅

## Tested Kernels

| Kernel | Status | Max Diff | Notes |
|--------|--------|----------|-------|
| `fa_s512_v3.cu` | ❌ FAIL | 5.070 | All configs fail |
| `fa_inverted_prod.cu` | ❌ FAIL | 0.267 | Builds, but wrong output |
| `fa_tc_s512.cu` | ❌ BUILD FAIL | - | Needs CUTLASS |
| `fa_inverted.cu` | ❌ BUILD FAIL | - | Build error |

## Next Actions (EvoEngineer: Adapt & Iterate)

**Option A: Check Session Logs** (5 min - TRY FIRST)  
Look in `docs/archive/session_logs/` for any documented working kernel or correctness evidence.

**Option B: Minimal Correct Kernel from Scratch** (2 hrs - RECOMMENDED IF A FAILS)  
1. Implement basic Flash Attention (scalar, no optimizations)
2. Verify correctness FIRST
3. Then apply EvoEngineer optimizations systematically

**Option C: Fix One of the Existing Kernels** (4+ hrs - RISKY)  
Debug why fa_s512_v3.cu or fa_inverted_prod.cu fails  
Pros: Might be faster if bug is simple  
Cons: Could waste hours on fundamental issues

## Recommendation

**Execute Option A RIGHT NOW (5 min)**:
```bash
cd docs/archive/session_logs
grep -r "correctness.*PASS\|torch.allclose.*True" . | head -20
```

If any kernel passed correctness in session logs → use that as baseline  
If NONE → proceed with **Option B** (build correct kernel from scratch)

## EvoEngineer Lesson

**Critical mistake**: Assumed V3 baseline was correct without testing  
**Fix**: ALWAYS verify baseline correctness before optimization work

## GPU Cost During This Session

**Duration**: ~1.5 hours  
**Cost**: ~$1.05  
**Value**: Discovered ALL kernels broken → critical finding!

**Keep iterating!** 🚀

