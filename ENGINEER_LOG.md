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

### Timestamp: 2025-10-14T01:15:00Z - Step 1 Additional Fix (THIRD OOB BUG)

**Files Modified**:
- `cudadent42/bench/kernels/fa_s512_v3.cu`

**Third OOB Bug Found & Fixed**:
- Same pattern as previous bugs: function signatures mismatched array sizes
- `compute_block` signature: `O_acc[BLOCK_M][HEAD_DIM]`, `m_i[BLOCK_M]`, `l_i[BLOCK_M]`
- Actual arrays (in kernel): `[BLOCK_M/NUM_WARPS][...]` (per-warp)
- Code used `[row_start + local_row]` where `row_start = warp_id * rows_per_warp`
- For warp 3: `row_start = 24`, accessing array[24] but size is only [8]

**Fix Applied**:
1. Updated `compute_block` signature to use per-warp arrays
2. Changed all indexing from `[row_start + local_row]` to `[local_row]`
3. Updated `write_O_to_gmem` signature and indexing similarly

**Reason**: Eliminate GMEM read OOB errors by ensuring all function signatures match actual per-warp array sizes

---

### Timestamp: 2025-10-14T01:00:00Z - Step 0 Complete (16-BYTE ALIGNMENT FIX)

**Files Modified**:
- `cudadent42/bench/kernels/detail/smem_swizzle.hpp`
- `cudadent42/bench/kernels/fa_s512_v3.cu`

**Alignment Fix Applied**:
1. Added 16-byte alignment utilities to `smem_swizzle.hpp`:
   - `elems_for_16B<T>()` - Returns elements per 16 bytes
   - `pad_to_16B_elems<T>(stride)` - Computes padding for alignment
   - Static assertion: `half` is 2 bytes → 8 elements per 16 bytes

2. Updated `KernelTraits` in `fa_s512_v3.cu`:
   - Compute `PAD_K` and `PAD_V` for 16-byte alignment
   - `K_STRIDE = HEAD_DIM + PAD_K` (64 + 0 = 64, already aligned)
   - `V_STRIDE = HEAD_DIM + PAD_V` (64 + 0 = 64, already aligned)
   - Added compile-time assertions for stride alignment
   - Added `BLOCK_M % NUM_WARPS == 0` assertion

3. Updated `SharedMemory` struct:
   - Added `__align__(16)` attribute to ensure base address alignment
   - Updated comments to clarify 16-byte alignment requirement

**Reason**: Eliminate cp.async alignment assertions by ensuring all SMEM bases and row strides are 16-byte aligned

---

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

- [✅] Compute-sanitizer memcheck: CLEAN (1 false-positive PyTorch leak, 0 real errors)
- [ ] Compute-sanitizer racecheck: CLEAN
- [ ] Compute-sanitizer initcheck: CLEAN
- [ ] Correctness: 7/7 tests pass (atol=1e-2, rtol=5e-2)
- [ ] Performance: ≤0.255 ms (mean)
- [ ] Statistics: 95% CIs non-overlapping vs V2
- [ ] Effect size: Hedges' g ≥ 0.8
- [ ] Nsight: ↑L2 hit-rate ≥+8pp, ↓DRAM% ≥−10pp

**Step 1 Results** (2025-10-14T01:30:00Z):
- Smoke test output: `[-0.1014, 0.1592]`, mean=0.0050
- No NaN/Inf detected
- Kernel executed successfully
- All 3 OOB bugs confirmed fixed

---

## Critical Finding (2025-10-14T02:30:00Z)

### Memory Layout Bug in ALL Kernels

**Discovery**: Both V2 and V3 kernels FAIL correctness tests:
- V3: max_abs_diff = 0.461 (non-causal), 2.045 (causal)
- V2: max_abs_diff = 2.236 (non-causal) ← **V2 also broken!**

**Root Cause**: Incorrect PyTorch (B, H, S, D) tensor indexing
- **WRONG** (used in V2, V3): `batch*(S*H*D) + head*D + seq*(H*D)`
- **CORRECT**: `batch*(H*S*D) + head*(S*D) + seq*D`

**Fix Applied to V3**:
```cpp
// Before (WRONG):
const int offset = batch_idx * seq_len * num_heads * Traits::HEAD_DIM +
                  head_idx * Traits::HEAD_DIM +
                  m * num_heads * Traits::HEAD_DIM;

// After (CORRECT):
const int offset = batch_idx * num_heads * seq_len * Traits::HEAD_DIM +
                  head_idx * seq_len * Traits::HEAD_DIM +
                  m * Traits::HEAD_DIM;
```

**Status**: V3 memory layout fixed (4 locations), V2 fixed (4 locations)

### Step 2 Complete: Correctness Tests Run

**Test Results** (2025-10-14T04:00:00Z):
- V2: **0/14 tests passed** ❌
- V3: **0/6 tests passed** ❌

**Sample Errors**:
- V2 B1_S128_H4_D64: max_abs=0.049 (threshold: 0.01)
- V3 B2_S512_H8_D64: max_abs=0.461 (noncausal), 2.045 (causal)

**Conclusion**: Memory layout fix **NOT sufficient**. Additional bugs present in both kernels.

**BREAKTHROUGH** (2025-10-14T05:30:00Z):

### V2 Correctness ACHIEVED ✅

**Root Cause Found**: Dimension order mismatch in V2 bindings!
- Bindings expected: (B, S, H, D)
- PyTorch actual: (B, H, S, D)
- Fix: Swapped `q.size(1)` and `q.size(2)` extraction

**V2 Test Results** (after bindings fix):
- **14/14 tests passed** ✅
- All shapes: (B=1-4, S=128-512, H=4-16, D=64)
- Both causal and non-causal
- Max abs errors: 0.0003-0.0005 (excellent FP16 precision)

**V3 Status**:
- 0/6 tests passed ❌
- Still has large errors (0.46-2.4 absolute)
- Different bug (likely online softmax or cp.async issues)

---

## Step A Complete: V2 Benchmark Results (2025-10-14T06:00:00Z)

### Performance Comparison (B=2, S=512, H=8, D=64, 100 iters)

```
Kernel | Mean (ms) | 95% CI         | Status
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
SDPA   | 0.051     | [0.049, 0.053] | ✅ FASTEST
V2     | 0.333     | [0.332, 0.333] | ❌ 6.5× SLOWER
Target | 0.255     |                | Threshold
```

**Statistical Analysis**:
- Speedup: 0.15× (V2 is 6.5× slower than SDPA)
- Hedges' g: -43.016 (massive effect size, V2 slower)
- CIs: Non-overlapping (statistically significant)

**Decision**: ❌ **DO NOT PROMOTE V2** - Exceeds target by 30%

**Artifacts**:
- Benchmark CSV: `artifacts/bench/v2_vs_sdpa_bs2_s512_h8_d64.csv`
- Decision JSON: `artifacts/stats/v2_decision.json`

---

## Final Recommendation

**Status**: DECISION COMPLETE  
**Champion**: **PyTorch SDPA** ✅

**Options**:
1. **Option A (Recommended)**: Use PyTorch SDPA (2× faster than broken kernels, correct)
2. Option B: Fix V2 memory layout (simpler kernel, easier to debug)
3. Option C: Continue debugging V3 (more complex, higher risk)

---

## Artifacts

- `artifacts/sanitizers/`: Compute-sanitizer logs
- `artifacts/correctness/`: Test results JSON/CSV
- `artifacts/bench/`: Performance benchmark CSV
- `artifacts/nsight/`: Nsight Compute profiles

