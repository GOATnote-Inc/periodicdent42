# Phase 5 Q@K^T WMMA Status
**Date**: Oct 16, 2025  
**Status**: 🔴 **CORRECTNESS FAILURE** - WMMA compiles but produces wrong results

---

## ✅ Achievements

### Infrastructure (COMPLETE)
- ✅ WMMA headers configured correctly (`#include <mma.h>`)
- ✅ Correct namespace (`nvcuda::wmma` not `nv::wmma`)
- ✅ Correct types (`__half` not `half`)
- ✅ Fragment types defined for Ada (sm_89)
- ✅ WMMA test program compiles and runs successfully
- ✅ Phase 5 kernel compiles with USE_WMMA=1

### Scalar Fallback (COMPLETE)
- ✅ Correctness: max_diff=0.000244 ✅ PASS
- ✅ Performance: 1136.82 μs (matches Phase 4)

---

## 🔴 Critical Issue: WMMA Correctness Failure

### Test Results (USE_WMMA=1)
```
Correctness Results:
  max_diff:  0.255859  ❌ FAIL (expected < 0.001)
  mean_diff: 0.000052
  Status:    ❌ FAIL
```

### Root Cause Analysis (Hypotheses)

**Hypothesis 1: Wrong WMMA shape for Ada**
- Using: 16x16x16 (`fragment<matrix_a, 16, 16, 16, __half, row_major>`)
- Ada (sm_89) may not support this shape
- Possible alternatives: 8x32x16 (explicitly listed in headers)
- **Action**: Check CUDA documentation for Ada WMMA shapes

**Hypothesis 2: Incorrect memory layout**
- K is loaded as `col_major` to represent K^T
- But actual data in SMEM may not match expected layout
- **Action**: Verify K^T transpose semantics

**Hypothesis 3: Wrong pointer cast**
- Casting 2D array `Q_tile[BLOCK_M][HEAD_DIM]` to `const half*`
- Should be: `(const half*)&Q_tile[0][0]` or similar
- **Action**: Fix pointer arithmetic

**Hypothesis 4: Incorrect load/store strides**
- `load_matrix_sync` needs correct ldm (leading dimension)
- Using `HEAD_DIM` for Q/K, `BLOCK_N` for S
- May be off-by-one or incorrect
- **Action**: Verify strides match SMEM layout

**Hypothesis 5: Warp tile coordination bug**
- Multiple warps writing to S_tile without proper coordination
- Race condition possible
- **Action**: Add explicit sync after WMMA, check tile boundaries

---

## 🔧 Immediate Next Steps

### Priority 1: Diagnose WMMA Implementation
1. **Add debug output** to print intermediate values
2. **Test single-warp case** (force only warp 0 to compute)
3. **Verify pointer arithmetic** in `wmma_qk_transpose` helper
4. **Check WMMA shape support** for Ada (sm_89)

### Priority 2: Simplify for Debugging
1. **Use 8x32x16 WMMA shape** (explicitly supported in headers)
2. **Test on small matrices** (16x16x16) with known values
3. **Compare scalar vs WMMA** output element-by-element

### Priority 3: Alternative Approach
**If WMMA debugging takes > 2 hours**: Consider switching to **CUTLASS** library
- CUTLASS provides higher-level abstractions
- Better documented for Ada architecture
- Used by production kernels (FlashAttention-2)
- Trade-off: More complex dependencies, but proven correct

---

## 📊 Time Investment

| Task | Estimated | Status |
|------|-----------|--------|
| WMMA infrastructure | 1 hour | ✅ COMPLETE |
| Q@K^T integration | 2 hours | 🟡 50% (compiles, fails correctness) |
| **Debugging WMMA** | **1-3 hours** | 🔴 **CURRENT BLOCKER** |
| P@V integration | 2-3 hours | ⏸️ Blocked |
| FP16 accumulation | 1 hour | ⏸️ Blocked |
| Validation | 1-2 hours | ⏸️ Blocked |
| **Total Remaining** | **5-9 hours** | **Blocked on correctness** |

---

## 🚦 Decision Point

### Option A: Debug WMMA (1-3 hours)
**Pros**:
- Learn low-level WMMA API
- Full control over implementation
- Educational value

**Cons**:
- Time-consuming debugging
- May hit more Ada-specific issues
- WMMA API is legacy (CUTLASS recommended)

**Recommendation**: ⚠️ **Try for 1 hour max, then pivot**

### Option B: Switch to CUTLASS (3-4 hours)
**Pros**:
- Production-proven library
- Ada-optimized templates
- Well-documented
- Used by FlashAttention-2

**Cons**:
- More complex setup
- Additional dependencies
- Steeper learning curve

**Recommendation**: ✅ **BEST** if WMMA debugging exceeds 1 hour

### Option C: Optimize Scalar Further (1-2 hours)
**Pros**:
- No correctness risk
- Guaranteed progress
- Can achieve 2-3× speedup with better tiling

**Cons**:
- Won't achieve 5-10× goal
- Misses Tensor Core benefits

**Recommendation**: ⚠️ **FALLBACK** option

---

## 📁 Evidence

### Files Created
- ✅ `cudadent42/bench/kernels/fa_phase5_wmma.cu` (562 lines)
- ✅ `cudadent42/bench/kernels/fa_phase5_wmma_bindings.cpp` (74 lines)
- ✅ `bench/build_phase5_variant.py` (78 lines)
- ✅ `scripts/test_phase5_qk.py` (157 lines)
- ✅ `scripts/test_wmma_availability.cu` (23 lines)

### Commits
- `feat(phase5): Q@K^T WMMA integration + bindings` (711667f)
- `test(phase5): add Q@K^T correctness + perf test` (8577f4c)
- `fix(phase5): guard WMMA includes with USE_WMMA macro` (874fba6)
- `test: add WMMA availability check` (7618c18)
- `fix(phase5): correct WMMA namespace (nvcuda::wmma)` (6f17fe0)

### Test Logs
- Scalar fallback: ✅ PASS (max_diff=0.000244, 1136.82 μs)
- WMMA path: ❌ FAIL (max_diff=0.255859)

---

## 🎯 Recommendation

**Immediate**: Spend 1 hour debugging WMMA with these specific checks:
1. Print first 16x16 tile output (scalar vs WMMA)
2. Test with identity matrices (Q=I, K=I → S=I)
3. Verify pointer arithmetic in helper functions
4. Check if 16x16x16 is supported on Ada (may need 8x32x16)

**If not resolved in 1 hour**: Pivot to **Option B (CUTLASS)** for guaranteed correctness

**Rationale**: Time is valuable. CUTLASS is production-proven and will get us to the goal faster than debugging low-level WMMA.

---

**Status**: 🔴 **BLOCKED on WMMA correctness**  
**Time Invested**: ~3 hours (infrastructure + integration)  
**Time to Resolution**: 1 hour (debug) OR 3-4 hours (pivot to CUTLASS)  
**Next**: User decision on debug vs pivot

