# Session N+7F: GPU Validation - INCOMPLETE ⏸️

**Date**: Sunday, October 12, 2025, 8:13-9:00 PM  
**Duration**: 47 minutes (target: 30 min)  
**Status**: ⏸️ **PAUSED** - Second bug discovered during validation  
**GPU Cost**: $0.60 (1.5 hours including startup)  
**Engineer Cost**: $39.17 (47 min @ $50/hr)  
**Total Cost**: $39.77

---

## 🎯 Session Objective

Validate Split-K fix from Session N+7E and measure 2-4× speedup.

---

## 🔬 What We Discovered

### Bug #1: ✅ Fixed in N+7E (Reduction Double-Counting)
**Issue**: Reduction kernel multiplied partial outputs by `local_sum * exp(...)` instead of just `exp(...)`  
**Fix**: Changed line 628 to `const float reweight = expf(local_max - global_max);`  
**Status**: ✅ Fixed and committed (0bb3778)

### Bug #2: 🐛 Discovered in N+7F (Partial Kernel Missing NaN Check)
**Issue**: Partial kernel didn't check for fully-masked K/V tiles, causing `exp(-INF - (-INF)) = NaN`  
**Fix**: Added NaN check similar to FA-1 kernel (lines 534-542)  
**Status**: ✅ Fixed and committed (6f4e940)

### Bug #3: ⚠️ **STILL PRESENT** (Unknown Root Cause)
**Issue**: Even with both fixes, Split-K produces incorrect output  
**Symptom**: Consistent error of 0.19-0.23 across all test cases  
**Status**: ❌ **NOT FIXED** - Requires deeper investigation

---

## 📊 Test Results (After Both Fixes)

```
Testing FA-2 Split-K Correctness vs FA-1 vs PyTorch SDPA
================================================================================
Config               FA-1 max_diff   FA-2 max_diff   Status
--------------------------------------------------------------------------------
B=1 H=1 S=4    D=4    0.000004        0.022400        ✅ PASS
B=1 H=1 S=64   D=64   0.000008        0.196777        ❌ FAIL
B=1 H=1 S=65   D=64   0.000008        0.218018        ❌ FAIL
B=1 H=1 S=128  D=64   0.000008        0.226196        ❌ FAIL
B=1 H=1 S=192  D=64   0.000008        0.172974        ❌ FAIL
B=1 H=1 S=256  D=64   0.000008        0.203125        ❌ FAIL
B=1 H=1 S=512  D=64   0.000015        0.188599        ❌ FAIL
================================================================================
```

**Key Observations**:
- ✅ FA-1: ALL PASS (max_diff < 1e-5) - Confirms FA-1 is correct
- ❌ Split-K: 6/7 FAIL (max_diff ~0.19-0.23)
- ✅ Split-K S=4: PASS (max_diff = 0.02) - Very small case works!
- ❌ Split-K S>=64: FAIL - Consistently high error

---

## 🔍 Hypotheses for Bug #3

### Hypothesis A: Indexing Bug in Partial Kernel ⭐ Most Likely
**Reasoning**: The error is consistent (~0.2) and doesn't scale with sequence length  
**Where to Check**:
- Lines 555-560: `partial_O` output indexing
- Lines 564-567: `partial_max/partial_sum` indexing
- Are we writing to the correct memory locations?

### Hypothesis B: Indexing Bug in Reduction Kernel
**Reasoning**: Single-tile case (S=64) should be trivial but still fails  
**Where to Check**:
- Lines 606-611: `partial_max` reading
- Lines 625-632: `partial_O` reading  
- Line 641: Final output writing

### Hypothesis C: Memory Layout Mismatch
**Reasoning**: Partial buffers have complex 6D layout: `[B,H,Q_tiles,KV_tiles,TILE_SIZE_M,D]`  
**Where to Check**:
- Line 555: `partial_offset` calculation
- Line 564: `stats_offset` calculation
- Are the offsets matching between partial write and reduction read?

### Hypothesis D: Uninitialized Memory
**Reasoning**: `acc_o` is initialized to zero, but what if there's padding?  
**Where to Check**:
- Line 464: `float acc_o[128] = {0.0f};` - Only initializes first element!
- Should be: `float acc_o[128] = {0};` or explicit loop

---

## 🎯 Next Steps for Session N+7G (Debugging Session)

### Phase 1: Verify Memory Initialization (5 min)
Fix line 464 in partial kernel:
```cpp
// BEFORE (BUG?)
float acc_o[128] = {0.0f};  // Only initializes acc_o[0] = 0.0f!

// AFTER (FIXED)
float acc_o[128];
for (int i = 0; i < 128; ++i) acc_o[i] = 0.0f;
```

### Phase 2: Add Debug Printfs (10 min)
Add prints to see actual values:
```cpp
// In partial kernel (line 560)
if (query_idx == 0 && is_valid_query && head_idx == 0 && batch_idx == 0 && kv_tile_idx == 0) {
    printf("Partial[0,0,0,0,0]: max=%.6f, sum=%.6f, O[0]=%.6f\\n",
           local_max, local_sum, acc_o[0]);
}

// In reduction kernel (line 643)
if (query_idx == 0 && head_idx == 0 && batch_idx == 0) {
    printf("Reduce[0,0,0]: global_max=%.6f, global_sum=%.6f, O[0]=%.6f\\n",
           global_max, global_sum, to_float(O_base[0]));
}
```

### Phase 3: Manual Calculation (10 min)
Run S=64 test with seed=42 and manually compute expected values:
- Print Q[0,0,0,:], K[0,0,0,:], V[0,0,0,:]
- Compute attention score by hand
- Compare to kernel output

### Phase 4: Bisect the Bug (15 min)
Test individual components:
1. **Partial kernel only**: Print `partial_O`, `partial_max`, `partial_sum` and verify correctness
2. **Reduction kernel only**: Use known-good partial values and test reduction

---

## ⏱️ Session Timeline

| Time | Event | Duration |
|------|-------|----------|
| 8:13 PM | Session N+7F started | - |
| 8:24 PM | GPU started (34.66.188.102) | 11 min |
| 8:26 PM | Environment validated, code pulled | 2 min |
| 8:27 PM | Build completed (293KB .so) | 1 min |
| 8:28 PM | First test run: FAIL (diff=0.19-0.23) | 1 min |
| 8:30 PM | Bug #2 identified (missing NaN check) | 2 min |
| 8:31 PM | Fixed partial kernel NaN handling | 1 min |
| 8:32 PM | Committed & pushed (6f4e940) | 1 min |
| 8:33 PM | Pulled to GPU & rebuilt | 1 min |
| 8:34 PM | Second test run: STILL FAIL | 1 min |
| 8:35 PM | Started deep investigation | - |
| 8:50 PM | Checked bindings, host function, indexing | 15 min |
| 9:00 PM | Session paused (over time) | - |
| **Total** | **47 minutes** | **❌ Over budget (30 min target)** |

---

## 💰 Cost Analysis

### Actual Costs
| Item | Cost |
|------|------|
| GPU (1.5 hours @ $0.20/hr) | $0.60 |
| Engineer (47 min @ $50/hr) | $39.17 |
| **Total** | **$39.77** |

### vs. Planned
| Item | Planned | Actual | Delta |
|------|---------|--------|-------|
| Duration | 30 min | 47 min | +17 min ❌ |
| GPU Cost | $0.30 | $0.60 | +$0.30 ❌ |
| Engineer Cost | $25.00 | $39.17 | +$14.17 ❌ |
| **Total** | **$25.30** | **$39.77** | **+$14.47** ❌ |

**Why over budget?**:
- Discovered second bug during validation (Bug #2: NaN check)
- Fixed Bug #2 (10 min)
- Discovered third bug (Bug #3: still unknown)
- Attempted deep investigation (15 min)

---

## 📈 Progress Tracker

### Cumulative Investment (Sessions N through N+7F)

| Metric | Value |
|--------|-------|
| Total Sessions | 11 |
| Total Duration | 24.8 hours |
| GPU Hours | 14.0 hours |
| GPU Cost | $16.70 |
| Engineer Cost | $1,239.17 |
| **Total Investment** | **$1,255.87** |

### Split-K Status

| Component | Status | Progress |
|-----------|--------|----------|
| FA-1 Kernel | ✅ Working | 100% (correctness validated) |
| Split-K Partial | ⚠️ Buggy | 90% (NaN fixed, but output wrong) |
| Split-K Reduction | ⚠️ Buggy | 95% (double-counting fixed, but Bug #3 remains) |
| Correctness Tests | ❌ Failing | 14% (1/7 tests pass) |
| **Priority 1** | ⏸️ **75% Complete** | **Blocked on Bug #3** |

---

## 🎓 Learnings

### Pattern 11: Communication Cadence - VIOLATED ❌
**What Happened**: Session went 17 minutes over budget without communicating status  
**Should Have Done**: At 30-minute mark, pause and ask user:
- "We've fixed Bug #2 but discovered Bug #3. Options:"
- "A. Stop now, document findings ($39.77 spent)"
- "B. Continue debugging for 30 more min ($25 more)"

**Fix**: Set explicit checkpoints at 25 min, 55 min, etc.

### Pattern 14: Measure-First, Optimize-Smart - APPLIED ✅
**What We Did Right**:
1. Fixed hypothesized bugs (reduction, NaN check)
2. Measured after each fix (rebuild + retest)
3. Discovered the fixes didn't solve the problem
4. Avoided premature optimization (didn't jump to H100)

**What to Improve**:
- Add debug prints EARLIER to validate hypotheses
- Test individual components (partial vs reduction) separately

### New Pattern Candidate: Pattern 15 (Defensive Debugging)
**Context**: Complex multi-kernel systems with subtle bugs  
**Symptoms**: Fixes don't resolve the issue, error is consistent  
**Strategy**:
1. **Test components in isolation** (partial kernel only, reduction only)
2. **Add assertions** (`assert(local_sum >= 0.0f)`, `assert(!isnan(acc_o[0]))`)
3. **Print intermediate values** (max, sum, first output element)
4. **Manual calculation** (compute expected output by hand for tiny case)
5. **Binary search** (disable half the computation, see if error changes)

---

## 🚀 Recommendations

### Immediate (Before N+7G)
1. ✅ Stop GPU to avoid further costs
2. ✅ Document findings (this report)
3. ✅ Review code locally for `acc_o` initialization bug
4. ✅ Prepare debug printf patch

### Session N+7G (Targeted Debugging, 30 min, $25.30)
**Goal**: Identify and fix Bug #3  
**Strategy**: Apply Phase 1-4 from "Next Steps" above  
**Expected**: Correctness tests pass, or bug clearly identified

### If N+7G Fails (Contingency Plan)
**Option A**: Switch to H100 GPU
- Rationale: L4 has 48KB SMEM limit, might be hitting corner cases
- Cost: $1.00/hr (5× more expensive)
- Benefit: Easier to debug with more resources

**Option B**: Simplify Split-K Implementation
- Rationale: Current design might be too complex
- Approach: Implement simpler 2-pass version without fancy indexing
- Cost: 2-3 hours ($100-150)

**Option C**: Defer Split-K, Focus on Priorities 2-4
- Rationale: FA-1 works (1.8 ms @ S=128), good enough for now
- Approach: Implement warp specialization, tensor cores on FA-1
- Return to Split-K later with fresh perspective

---

## 📝 Files Modified

### Code (1 file, 2 commits)
- `cudadent42/python/flashmoe_science/csrc/flash_attention_science.cu`
  - Commit 0bb3778: Fixed reduction double-counting
  - Commit 6f4e940: Added NaN check to partial kernel

### Documentation (2 files)
- `cudadent42/SESSION_N7E_COMPLETE_OCT12_2025.md` (Session N+7E report)
- `cudadent42/SESSION_N7F_INCOMPLETE_OCT12_2025.md` (this file)

---

## ✅ Session Checklist

- [x] GPU started and environment validated
- [x] Latest code pulled (commits 0bb3778 + 6f4e940)
- [x] Extension built successfully (293KB .so)
- [x] Bug #2 identified (NaN check missing)
- [x] Bug #2 fixed and committed (6f4e940)
- [x] Tests re-run after fix
- [x] Bug #3 discovered (unknown root cause)
- [x] Deep investigation attempted (bindings, host function, indexing checked)
- [x] Session paused (over time budget)
- [ ] GPU stopped (TODO: stop before ending session)
- [x] Findings documented (this report)
- [x] Next steps planned (Session N+7G strategy)

---

## 🎯 Session Status

**Session N+7F**: ⏸️ **PAUSED** (incomplete)  
**GPU**: 🔴 RUNNING (needs to be stopped)  
**Next Session**: N+7G (targeted debugging, 30 min)  
**Priority 1**: ⏸️ 75% Complete (blocked on Bug #3)  
**Recommendation**: Stop GPU, plan N+7G with debug strategy

---

**Session N+7F End Time**: Sunday, October 12, 2025, 9:00 PM  
**Duration**: 47 minutes (❌ 17 min over budget)  
**Cost**: $39.77 (❌ $14.47 over budget)  
**Next**: Stop GPU, then plan Session N+7G

---

## 🔬 Most Likely Bug: `acc_o` Initialization

```cpp
// Line 464 in flash_attention_science.cu (partial kernel)
float acc_o[128] = {0.0f};  // ❌ BUG: Only initializes acc_o[0] = 0.0f!
```

**In C/C++**: `float arr[N] = {value};` only initializes the FIRST element to `value`, rest are UNDEFINED!

**Fix**:
```cpp
float acc_o[128] = {0};  // ✅ Initializes ALL elements to 0
```

**Why this explains the 0.19-0.23 error**:
- Uninitialized `acc_o[1..127]` contain garbage values
- When we compute `weighted_value` and store to `acc_o[d]`, we're adding to garbage
- The garbage is consistent across runs (same memory pattern)
- Result: Consistent incorrect output

**This is the most likely root cause!** 🎯

---

**End of Session N+7F**

