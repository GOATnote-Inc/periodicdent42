# V3 CUDA Kernel Engineering Log

**Session Start**: 2025-10-14 (Continuation)  
**Objective**: Fix illegal memory access in V3, validate correctness, achieve ≤0.255 ms or fallback to V2  
**Engineer**: CUDA Principal (AI-assisted)

---

## Fix Plan

### Root Cause Hypothesis
V3 kernel has "CUDA illegal memory access" at runtime. Likely causes:
1. **cp.async address alignment**: Must be 16-byte aligned for cp.async.ca<16>
2. **Persistent block work distribution**: Bounds checking on (batch, head, m_block) iteration
3. **Register array indexing**: Out-of-bounds in Q_reg, O_acc, m_i, l_i
4. **SMEM indexing**: Stage buffer pointer arithmetic (stage 0/1)
5. **half2 alignment**: Assumes 2-element alignment, may not hold

### Fix Strategy
1. Add DEBUG guards with bounds checking
2. Run compute-sanitizer (memcheck, racecheck, initcheck)
3. Fix reported issues one at a time
4. Validate correctness (7 test cases)
5. Performance gate (≤0.255 ms with CI + effect size)
6. Nsight profiling if accepted
7. Decision: Promote V3 or fallback to V2

---

## Diffs Applied

### Timestamp: 2025-10-14T00:30:00Z - Step 1 Complete (ROOT CAUSE FOUND & FIXED)

**Files Modified**:
- `cudadent42/bench/kernels/fa_s512_v3.cu`
- `cudadent42/bench/kernels/fa_s512_v3_bindings.cpp`

**Root Cause Identified by compute-sanitizer**:
- **Location**: `load_Q_to_registers` at line 121
- **Bug**: Invalid configs with `BLOCK_M=32, NUM_WARPS=6`
  - `rows_per_warp = 32 / 6 = 5` (integer division)
  - Leaves 2 orphaned rows (32 % 6 = 2)
  - Threads 160-191 (warp 5) attempt OOB writes to `Q_reg`
  - `Q_reg` sized as `[5][64]`, but threads access indices 5 and 6

**Fix Applied**:
1. Config 1: `Traits_32_64_6_2_1_1` → `Traits_32_64_4_2_1_1` (WARPS: 6→4)
2. Config 2: `Traits_32_32_6_2_1_1` → `Traits_32_32_4_2_1_1` (WARPS: 6→4)
3. Config 3: `Traits_48_64_8_2_1_1` (unchanged, already valid: 48 % 8 == 0)
4. Updated bindings to match new function names

**Reason**: Ensure `BLOCK_M % NUM_WARPS == 0` for all configs to prevent OOB register array access

---

### Timestamp: 2025-10-14T00:00:00Z - Step 0 Complete

**Files Modified**:
- `cudadent42/bench/kernels/detail/debug_utils.cuh` (NEW)
- `cudadent42/bench/kernels/fa_s512_v3.cu`

**Changes**:
1. Created `debug_utils.cuh` with:
   - `oob()` helper for bounds checking
   - `CUDA_DEBUG_ASSERT()` macro (enabled with -DDEBUG_V3)
   - `is_aligned_16()` for cp.async alignment validation
   - `gmem_in_bounds()` and `smem_in_bounds()` helpers

2. Modified `fa_s512_v3.cu`:
   - Added `#include "detail/debug_utils.cuh"`
   - Added static_assert in `smem_bytes()` to enforce 48KB limit
   - Added work distribution bounds checking (batch_idx, head_idx, m_block)
   - Added cp.async alignment checks in `load_K_async()` and `load_V_async()`
   - Added stage/row/col bounds checking before all SMEM accesses

**Reason**: Establish debug infrastructure to pinpoint illegal memory access root cause via compute-sanitizer

---

## Commands Run

### Timestamp: [PENDING]

```bash
# Commands will be logged here
```

**Output**: [PENDING]

---

## Results

### Pass/Fail Gates

- [ ] Compute-sanitizer memcheck: CLEAN
- [ ] Compute-sanitizer racecheck: CLEAN
- [ ] Compute-sanitizer initcheck: CLEAN
- [ ] Correctness: 7/7 tests pass (atol=1e-2, rtol=5e-2)
- [ ] Performance: ≤0.255 ms (mean)
- [ ] Statistics: 95% CIs non-overlapping vs V2
- [ ] Effect size: Hedges' g ≥ 0.8
- [ ] Nsight: ↑L2 hit-rate ≥+8pp, ↓DRAM% ≥−10pp

---

## Next Decision

**Status**: IN PROGRESS  
**Next Action**: Step 0 - Add debug utilities and guards

---

## Artifacts

- `artifacts/sanitizers/`: Compute-sanitizer logs
- `artifacts/correctness/`: Test results JSON/CSV
- `artifacts/bench/`: Performance benchmark CSV
- `artifacts/nsight/`: Nsight Compute profiles

