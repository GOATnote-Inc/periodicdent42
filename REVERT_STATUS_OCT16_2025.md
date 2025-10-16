# Revert Status Report - October 16, 2025

## Executive Summary

**Objective**: Revert recent 64×64 tile changes and re-establish a working V3 baseline  
**Result**: ❌ **NO WORKING V3 BASELINE EXISTS**  
**Current State**: Feature branch reset to `main` (commit `e9e38f0`) which contains broken WMMA code  
**Recommendation**: **Abandon V3**, restart from scratch or pursue alternative approach

---

## Critical Findings

### 1. 64×64 Tile Integration Catastrophically Broken
**Commits**: `944f20c` → `1b39d65` (5 commits)

**Symptoms**:
- ❌ All 6 configs produce **wrong results** (max_diff=5.07) or **NaN outputs**
- ❌ Performance: 8.9-10.5ms (234× slower than expected 38μs)
- ❌ Device-side printf overhead (200× slowdown)

**Root Cause**: Fundamental correctness issue introduced during 64×64 tile refactoring, affecting:
- Shared memory layout changes
- Work distribution (persistent loop vs one-block-per-work-item)
- Accumulator management (registers vs SMEM)

---

### 2. Pre-64×64 Code Also Broken
**Commits Tested**: `1ca5eeb`, `26d2e49` (softmax fix era)

**Symptoms**:
- ❌ Correctness: FAIL (max_diff=5.07, mean_diff=0.61)
- ❌ Performance: ~9.2ms (same catastrophic slowdown)
- ❌ Debug spam: 110 grid config prints per benchmark
- ❌ `-DUSE_WMMA` flag enabled but WMMA code broken

**Conclusion**: The "CORRECTNESS ACHIEVED" commits **never actually achieved correctness**.

---

### 3. Main Branch (Latest Merged Code) Also Broken
**Commit**: `e9e38f0` (PR #63 merged)

**Symptoms**:
- ❌ **CUDA launch failure**: "unspecified launch failure"
- ❌ WMMA warnings: "cannot perform wmma load or store on local memory" (10× warnings)
- ❌ Only 28-32 registers (down from expected 111) → suggests dead code or broken compilation
- ❌ Never produces output (kernel crashes immediately)

**Root Cause**: WMMA implementation fundamentally broken - attempting wmma operations on local memory which is invalid on Ada architecture.

---

### 4. No V2 Fallback Available
- `build_v2_release.py` does not exist on main branch
- V2 code removed during refactoring
- No documented working baseline to revert to

---

## Revert History (Chronological)

### Attempt 1: Revert to Pre-64×64 State
- **Target**: `d32b7e8` (baseline verification docs)
- **Result**: ❌ **Build failure** - undefined `UNROLL_D` identifier
- **Cause**: EvoEngineer hooks introduced compilation errors

### Attempt 2: Revert to Softmax Fix Era
- **Target**: `1ca5eeb` (CORRECTNESS ACHIEVED)
- **Result**: ❌ **Correctness FAIL** + **9.2ms performance**
- **Cause**: Softmax fix never actually fixed correctness; WMMA broke everything

### Attempt 3: Revert to Pre-WMMA FlashAttention
- **Target**: `26d2e49` (mask-aware softmax)
- **Result**: ❌ **Same failure** (correctness FAIL, 9.2ms, `-DUSE_WMMA` still enabled)
- **Cause**: WMMA flag persisted despite code changes

### Attempt 4: Revert to Original FlashAttention
- **Target**: `bfd9d48` (first FlashAttention implementation)
- **Result**: ❌ **Build failure** - `cudadent42/bench/` directory doesn't exist
- **Cause**: Too old - predates directory structure

### Attempt 5: Switch to Main Branch
- **Target**: `main` branch (V2 CORRECTNESS ACHIEVED)
- **Result**: ❌ **CUDA launch failure** (WMMA local memory error)
- **Cause**: Main branch has broken WMMA code from PR #63

---

## Technical Analysis

### WMMA Local Memory Issue (Critical)
```
warning: cannot perform wmma load or store on local memory
```

**Explanation**: NVIDIA Ada (sm_89) Tensor Cores **cannot** operate on local/register memory. WMMA fragments must be loaded from:
- ✅ Shared memory (SMEM)
- ✅ Global memory (GMEM)
- ❌ **NOT local memory (registers/stack)**

**Impact**: All WMMA code on main branch is fundamentally broken and will never work on L4.

### Performance Regression Pattern
All broken versions show **consistent ~9ms latency** regardless of tile configuration:
- Expected: 38μs (from baseline verification)
- Actual: 9,000-10,000μs
- **Slowdown factor**: 234-263×

**Hypothesis**: Persistent loop is processing 234× more work than intended, likely due to:
- Incorrect grid dimension calculation
- Work ID decoding bug
- Missing early exit condition

### Correctness Failure Pattern
All broken versions show **max_diff=5.07** (identical value):
- This is **NOT random noise** (would vary across runs)
- Suggests **systematic algorithmic error** in softmax or accumulation
- Likely: `m_i` (max) or `l_i` (normalization) not updated correctly across tiles

---

## Current Repository State

```
feature/wmma_smem_phase2: e9e38f0 (reset to main)
main:                     e9e38f0 (broken WMMA code from PR #63)
```

**Build Status**: ✅ Compiles (with WMMA warnings)  
**Runtime Status**: ❌ CUDA launch failure  
**Test Status**: ❌ Cannot run correctness tests (kernel crashes)  
**Performance Status**: ❌ Cannot benchmark (kernel crashes)

---

## Recommendations

### Option A: Abandon V3, Start Fresh (RECOMMENDED)
**Rationale**: V3 has no working baseline to revert to. Every commit in the history is broken.

**Action Plan**:
1. Create new branch `feature/v3_clean_slate`
2. Start with **scalar-only** FlashAttention (no WMMA, no optimizations)
3. Establish correctness **first** (oracle tests, parity tests)
4. Benchmark baseline (should be ~100-200μs)
5. Add optimizations **incrementally** with correctness gates at each step

**Timeline**: 1-2 weeks to working scalar baseline + optimization roadmap

---

### Option B: Fix WMMA Code on Main
**Rationale**: Main branch compiles but has WMMA local memory issue.

**Action Plan**:
1. Remove `-DUSE_WMMA` flag (disable WMMA temporarily)
2. Test if scalar path works
3. If scalar works, incrementally fix WMMA by:
   - Moving wmma fragment loads to SMEM (not local memory)
   - Using `wmma::load_matrix_sync` from shared memory
   - Ensuring proper alignment (16-byte)
4. Re-enable WMMA once scalar baseline stable

**Timeline**: 2-3 days to fix WMMA, if scalar path works

---

### Option C: Forensic Analysis (NOT RECOMMENDED)
**Rationale**: Deep-dive debugging to find root cause of 234× slowdown.

**Against**: 
- No working baseline to compare against
- Multiple compounding issues (WMMA + softmax + grid + accumulation)
- High risk of rabbit holes
- Low probability of success given 5 failed revert attempts

**Only pursue if**: You have Nsight Compute access and can profile the broken kernel to see exactly what's executing.

---

## Key Lessons

### 1. Never Merge Without Correctness Tests
- PR #63 was merged despite WMMA warnings
- No correctness validation on actual hardware
- "CORRECTNESS ACHIEVED" claims were never validated

### 2. Maintain Known Good Baselines
- V2 code was removed without preserving
- No tagged releases marking working states
- Reverting is impossible without baselines

### 3. Incremental Changes with Gates
- 64×64 tile integration changed 10+ things simultaneously
- Impossible to isolate which change broke correctness
- Should have been: 1 change → test → commit → repeat

### 4. Hardware-Specific Constraints Must Be Validated
- Ada (sm_89) Tensor Core constraints differ from Ampere (sm_80)
- WMMA local memory restriction was ignored
- Compiler warnings should be hard blockers

---

## Next Steps (Immediate)

1. ✅ **Feature branch reset to main** (`e9e38f0`)
2. ⏳ **User decision required**: Option A, B, or C?
3. ⏳ **If Option A**: Create `feature/v3_clean_slate` branch
4. ⏳ **If Option B**: Test scalar path by removing `-DUSE_WMMA`
5. ⏳ **If Option C**: Request Nsight Compute access for forensic profiling

---

## Artifacts Generated

- This status report: `REVERT_STATUS_OCT16_2025.md`
- Feature branch: `feature/wmma_smem_phase2` @ `e9e38f0` (reset to main)
- GPU instance: `cudadent42-l4-dev` (still running, can test changes)

---

## Contact

For questions or to discuss next steps, reference this document:
- **File**: `REVERT_STATUS_OCT16_2025.md`
- **Date**: October 16, 2025, 12:36 AM PDT
- **Context**: Failed revert after 64×64 tile integration broke V3 kernel
- **Result**: No working V3 baseline exists; main branch also broken

**Recommendation**: **Option A** (clean slate) is the safest path forward.

