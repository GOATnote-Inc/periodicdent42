# TMA Kernel Iteration Log - November 1, 2025

## Goal
Push beyond 68.8 TFLOPS using architectural changes (TMA + CuTe DSL).

## Iteration 1: Direct TMA Port ❌

**Approach:** Port user-provided CuTe TMA kernel with SM90 descriptors

**Errors:**
1. `Unknown TMA Format!` - TMA doesn't recognize `half` type directly
2. `gmem_ptr` missing `.coord_` member - incorrect pointer type
3. TMA descriptor creation API mismatch with CUTLASS 4.3.0

**Root Cause:**  
TMA descriptor API in CuTe is complex and version-sensitive. The kernel assumes specific CUTLASS internals that may differ between versions.

**Learning:**  
- TMA requires careful type mapping (half → CU_TENSOR_MAP_DATA_TYPE_FLOAT16)
- Descriptor creation is non-trivial, needs proper layouts and strides
- CuTe DSL has steep learning curve

---

## Alternative Paths (Ranked)

### 1. cuBLASLt Per-Block ⭐⭐⭐
- **Why:** Proven, architectural fit, uses tensor cores automatically
- **Risk:** Low - known to work
- **Estimated gain:** 3-5× (200-300 TFLOPS range)
- **Complexity:** Medium - need to batch per-block calls

### 2. Larger Tiles (256×256×64) ⭐⭐
- **Why:** Better compute/memory ratio, more reuse
- **Risk:** Low - incremental change
- **Estimated gain:** 1.2-1.5× (80-100 TFLOPS)
- **Complexity:** Low - just change constants

### 3. Fix WMMA Accumulation ⭐
- **Why:** Enable real tensor cores (not just compiler hints)
- **Risk:** High - previous attempts failed
- **Estimated gain:** 2-3× (150-200 TFLOPS)
- **Complexity:** High - load-add-store pattern complex

### 4. Study CUTLASS Examples Deeper ⭐⭐
- **Why:** Learn from proven TMA/WGMMA usage
- **Risk:** Medium - learning curve
- **Estimated gain:** Unknown - educational
- **Complexity:** Medium - requires deep study

---

## Next Step Decision

**Recommendation:** Try **cuBLASLt per-block** approach
- Lowest risk
- Proven to work
- Uses H100 tensor cores automatically
- If successful → 3× improvement possible

**Alternative:** Try **larger tiles** first (quick win, low risk)

---

**Status:** 1 iteration, 0 wins, TMA path harder than expected  
**Current best:** 68.8 TFLOPS (vectorized scalar)  
**Target:** >200 TFLOPS (tensor core utilization)

## Iteration 5: CUTLASS Types + Loading ✅

**Approach:** Use `cutlass::half_t` + `cutlass::arch::global_load`

**Result:** **69.6 TFLOPS** (+0.8 over baseline)

**Key Changes:**
- Replaced `half` with `cutlass::half_t` (TMA-compatible type)
- Replaced `__ldg` with `cutlass::arch::global_load` (proper CUTLASS API)
- Maintained same kernel structure as 68.8 TFLOPS baseline

**Status:** ✅ **FIRST IMPROVEMENT!**

**Learning:**
- CUTLASS types work correctly on H100
- Type-correct foundation enables TMA path
- Small gain (0.8 TFLOPS) validates approach

---

**Progress:** 5 iterations, 1 win, ready for TMA descriptors

## Iteration 6-8: TMA Descriptor Challenges ❌

**Attempts:**
- Iter 6: Full TMA kernel with barriers → wmma namespace errors
- Iter 7: Fixed includes → cutlass::arch barrier API missing  
- Iter 8: Compact version with cute barriers → TMA type format errors

**Recurring Issue:** `Unknown TMA Format!` for `half` type
- TMA descriptor creation requires specific type mapping
- `half` → `CU_TENSOR_MAP_DATA_TYPE_FLOAT16` mapping not automatic
- CUTLASS type system integration needed

**Status:** 3 more iterations on TMA path, core issue remains

---

## Current Achievement

**Validated:** 69.6 TFLOPS (Iteration 5)  
**Method:** CUTLASS types + arch::global_load  
**Improvement:** +0.8 TFLOPS (+1.2%) over 68.8 baseline

**Path forward:**
1. Study CUTLASS Example 62 TMA usage in detail
2. Match exact type patterns from working examples
3. Continue TMA iterations (10+ expected per user guidance)

---

**Total iterations:** 8  
**Wins:** 1 (69.6 TFLOPS)  
**Learning:** TMA steep but correct path  
**Next:** Study Example 62, iterate 9-20
