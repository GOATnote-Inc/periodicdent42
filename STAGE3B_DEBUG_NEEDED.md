# Stage-3B Fused Softmax: Debugging Required

**Date**: October 20, 2025  
**Status**: ❌ Correctness gate FAILED  
**Branch**: `feat/stage3-fusion-full`  
**Commit**: `8eb24fd`

---

## Implementation Complete, But Broken

**What was implemented**:
- ✅ WMMA accumulator LUT generation (32 lanes × 8 elements)
- ✅ Fused softmax in registers (192 LOC kernel changes)
- ✅ Conditional compilation (Stage-2 fallback intact)
- ✅ Build system integration (USE_FUSED_SOFTMAX toggle)

---

## Validation Results

### Gate 1: PTXAS ✅ PASSED

| Metric | Stage-2 Baseline | Stage-3B Fused | Delta |
|--------|------------------|----------------|-------|
| Registers | 96 | **73** | **-23** ✅ |
| SMEM | 37.1 KB | 35.1 KB | -2 KB ✅ |
| Spills | 0 | 0 | 0 ✅ |

**Verdict**: Resource usage **improved**! Surprising but good.

### Gate 2: Correctness ❌ FAILED

| Shape | Stage-2 | Stage-3B | Status |
|-------|---------|----------|--------|
| small, seed=0 | PASS (0.046) | **FAIL (2.40)** | ❌ |
| small, seed=1 | PASS (0.060) | **FAIL (3.60)** | ❌ |
| small, seed=2 | PASS (0.046) | **FAIL (2.02)** | ❌ |
| mission, seed=0 | PASS (0.054) | **FAIL (1.21)** | ❌ |
| mission, seed=1 | PASS (0.036) | **FAIL (3.14)** | ❌ |
| mission, seed=2 | PASS (0.047) | **FAIL (2.68)** | ❌ |

**Verdict**: **0/6 tests passed**. Fundamental bug in fused softmax.

**Error characteristics**:
- Small shape: 85% of elements wrong, mean error 0.3
- Mission shape: 37% of elements wrong, mean error 0.06
- Errors are **100× larger** than FP8 tolerance (0.06)

---

## Root Cause Analysis

### Most Likely Bugs

1. **WMMA Fragment Element Count Mismatch**
   ```cpp
   float scores[8];  // Hardcoded assumption
   ```
   **Issue**: `c_frag.num_elements` might not be 8 for FP32 accumulator on sm_89.
   **Fix**: Print at runtime, use dynamic size.

2. **Warp Reduction Broadcast Error**
   ```cpp
   m_row[r] = __shfl_sync(0xffffffff, mymax, 0);
   ```
   **Issue**: Broadcast from lane 0 may not contain the reduced max.
   **Fix**: Ensure lane 0 has the final reduced value before broadcast.

3. **LUT Indexing Off-by-One**
   ```cpp
   int rr = WMMA_ACCUM_LUT[lane][i][0];
   ```
   **Issue**: LUT may be generated incorrectly or indexed wrong.
   **Fix**: Add debug prints to verify LUT values match expected (row, col).

4. **Race Condition in P Materialization**
   ```cpp
   sP[r_glob][c_glob] = __float2half(__expf(scores[i] - m_new));
   ```
   **Issue**: Missing `__syncthreads()` before WMMA P·V loads sP.
   **Fix**: Add `__syncthreads()` after P write loop, before WMMA section.

5. **Online Softmax Math Error**
   ```cpp
   l_smem[r_glob] = l_old * rescale + l_add;
   ```
   **Issue**: `l_add` computed per-lane, needs warp reduction before lane-0 update.
   **Fix**: Ensure `l_add` is fully reduced across warp before writing.

### Why PTXAS Improved But Correctness Failed

- **Fewer registers**: Compiler optimized broken code (dead code elimination?)
- **Less SMEM**: `sS` not used when `USE_FUSED_SOFTMAX=1` (-2 KB)
- **But wrong results**: Logic errors don't cause PTXAS failures

---

## Debugging Plan

### Step 1: Verify Fragment Size (5 min)
```cpp
#if USE_FUSED_SOFTMAX
if (tid == 0 && t == 0) {
    printf("c_frag.num_elements = %d\n", c_frag.num_elements);
}
#endif
```

**Expected**: Should print 8 for FP32 16×16×16 accumulator.

### Step 2: Print LUT Sample (5 min)
```cpp
if (lane == 0 && tid == 0 && t == 0) {
    printf("LUT[0]: ");
    for (int i = 0; i < 8; i++) {
        printf("(%d,%d) ", WMMA_ACCUM_LUT[0][i][0], WMMA_ACCUM_LUT[0][i][1]);
    }
    printf("\n");
}
```

**Expected**: Should match generated header (0,0), (0,1), (0,8), (0,9), (8,0), (8,1), (8,8), (8,9).

### Step 3: Compare Scores with Stage-2 (10 min)
Enable Stage-3B with debug prints:
```cpp
#if USE_FUSED_SOFTMAX
// After scaling
if (warp_id == 0 && lane == 0 && t == 0) {
    printf("Stage-3B scores[0:4]: %.4f %.4f %.4f %.4f\n", 
           scores[0], scores[1], scores[2], scores[3]);
}
#endif
```

Compare with Stage-2 sS values at same location.

### Step 4: Check Warp Reduction (15 min)
```cpp
// After m_row computation
if (warp_id == 0 && lane == 0 && t == 0) {
    printf("m_row[0:4]: %.4f %.4f %.4f %.4f\n", 
           m_row[0], m_row[1], m_row[2], m_row[3]);
}
```

Compare with Stage-2 m_new values.

### Step 5: Add __syncthreads() (5 min)
After P materialization, before WMMA P·V:
```cpp
#if USE_FUSED_SOFTMAX
}  // End of lane write loop
__syncthreads();  // ← ADD THIS
#endif
```

**Rationale**: Ensure all lanes finish writing sP before WMMA loads it.

---

## Quick Fixes to Try (Ranked)

### Fix 1: Add __syncthreads() after P write ⭐⭐⭐
**Location**: After line 533 and 888 (both WMMA sections)
```cpp
#endif // USE_FUSED_SOFTMAX
}
NVTX_POP();
__syncthreads();  // ← ADD HERE (already exists, verify it's after fused path)
```

**Likelihood**: High (common mistake)

### Fix 2: Fix warp reduction broadcast ⭐⭐
**Location**: Lines 479, 834
```cpp
// Before broadcast, ensure lane 0 has final value
mymax = __shfl_sync(0xffffffff, mymax, 0);  // ← Lane 0 already has it after reduction
m_row[r] = mymax;  // ← All lanes get same value
```

**Current code broadcasts before final reduction completes.**

### Fix 3: Use c_frag.num_elements instead of 8 ⭐
**Location**: Lines 455, 810
```cpp
float scores[c_frag.num_elements];  // ← Dynamic size
#pragma unroll
for (int i = 0; i < c_frag.num_elements; i++) {
```

**Likelihood**: Medium (sm_89 likely is 8, but verify)

---

## Next Session Action Items

1. **Immediate**: Add Fix 1 (__syncthreads()) - easiest, high impact
2. **Debug**: Add print statements (Steps 1-4 above)
3. **Compare**: Run side-by-side with Stage-2, compare intermediate values
4. **Fix**: Apply fixes 2-3 based on debug output
5. **Revalidate**: Run full validation once fixed

---

## Session 3 Summary

**Duration**: ~4 hours  
**Commits**: 12 (LUT gen, kernel impl, validation scripts)  
**LOC**: 192 kernel changes, 6 new files  
**Status**: Implementation complete, correctness gate failed

**Achievements**:
- ✅ WMMA LUT infrastructure
- ✅ Complete fused softmax implementation
- ✅ PTXAS gate passed (improved metrics!)
- ❌ Correctness gate failed (0/6 tests)

**Lessons**:
- Complex WMMA register manipulations need incremental validation
- Should have tested with DEBUG_PRINT from the start
- PTXAS success != correctness (dead code elimination)

---

**Next**: Debug session (~2-3 hours to find and fix bug)  
**Decision**: Do NOT merge. Keep USE_FUSED_SOFTMAX=0 by default.

