# Stage-3B Fused Softmax Hotfix Status

**Date**: October 20, 2025  
**Branch**: `feat/stage3-fusion-full`  
**Last Commit**: `8130696` - Zero ENTIRE sP tile fix  
**Status**: ❌ **CORRECTNESS GATE FAILED** (0/6 tests)

---

## 📊 Current Validation Results

### PTXAS Gate: ✅ **PASSED**
```
Stage-2 (control):    96 regs, 37.1 KB SMEM, 0 spills
Stage-3B (fused):     83 regs, 35.1 KB SMEM, 0 spills
Improvement:          -13 regs, -2 KB SMEM 🎉
```

### Correctness Gate: ❌ **FAILED**

| Shape   | Seed | max_err (S2) | max_err (S3B) | Δ      | Status |
|---------|------|--------------|---------------|--------|--------|
| small   | 0    | 0.046        | **2.402**     | 52×    | ❌ FAIL |
| small   | 1    | 0.060        | **3.598**     | 60×    | ❌ FAIL |
| small   | 2    | 0.046        | **2.023**     | 44×    | ❌ FAIL |
| mission | 0    | 0.162        | **1.061**     | 6.5×   | ❌ FAIL |
| mission | 1    | 0.083        | **4.121**     | 50×    | ❌ FAIL |
| mission | 2    | 0.079        | **3.033**     | 38×    | ❌ FAIL |

**Key Observation**: Stage-3B errors are **6-60× worse** than Stage-2, indicating a fundamental algorithmic bug, not just accumulated FP8 noise.

---

## 🛠 Fixes Applied (Session 4)

### Fix #1: KV Mask + Dynamic Fragment Size (Commit `c374200`)
**Hypothesis**: Including invalid columns from partial KV tiles in max/sum  
**Changes**:
- Compute `kv_local = clamp(kv_len - warp_n, 0, WMMA_N)` for each tile
- Mask reductions: `if (rr == r && cc < kv_local)`
- Dynamic fragment size: `const int FRAG_ELEMS = c_frag.num_elements;`
- Block sync after P materialization: `__syncthreads()` before WMMA(P,V)

**Result**: ❌ FAILED — Same error magnitudes (max_err 2.4-4.1)

### Fix #2: Zero ENTIRE sP Tile (Commit `8130696`)
**Hypothesis**: Valid columns had stale data from previous KV tiles  
**Changes**:
- Changed pre-zero condition from `if (cc >= kv_local)` to `if (true)`  
- Zeros ALL 256 elements in each warp's 16×16 sP tile

**Result**: ❌ FAILED — No change in error magnitudes

---

## 🔍 Remaining Hypotheses

### High Priority
1. **Cross-Warp m_smem Race Condition**  
   - Step 2: lane 0 writes `m_smem[r_glob]` (per-row)
   - Step 3: ALL lanes read `m_smem[r_glob]` to compute P
   - Issue: `__syncwarp()` only syncs within warp; cross-warp reads may see stale values
   - **Test**: Add `__syncthreads()` after step 2 (online softmax update)

2. **WMMA Fragment Layout Mismatch**  
   - The `WMMA_ACCUM_LUT[lane][i][0/1]` maps fragment elements to (row, col)
   - Possible issue: Not all lanes see all rows/cols → incomplete reductions
   - **Test**: Print LUT and verify 16 rows × 16 cols are covered across all lanes

3. **Warp Reduction Over Masked Values**  
   - When masking `if (rr == r && cc < kv_local)`, some lanes contribute `-INFINITY`
   - `warp_reduce_max(-INFINITY)` across lanes might not broadcast correctly
   - **Test**: Initialize `mymax = -INFINITY` only for valid (r, c) pairs; else use a sentinel

### Medium Priority
4. **Online Softmax Math Error**  
   - The rescale/accumulate logic might be incorrect when fused
   - Compare against scalar softmax loop in Stage-2 (line-by-line)
   - **Test**: Add DEBUG_PRINT for `m_old`, `m_new`, `l_add`, `rescale` per row

5. **P Materialization Using Wrong m_new**  
   - In step 3, we read `m_smem[r_glob]`, but this was updated by lane 0 only
   - All lanes must see the updated value (requires sync)
   - **Test**: Verify `__syncwarp()` is sufficient for intra-warp visibility

### Low Priority
6. **kv_local Calculation Error**  
   - `kv_local = kv_len - warp_n` might be wrong if `warp_n` indexing is off
   - **Test**: Print `warp_n`, `kv_len`, `kv_local` for partial tiles

7. **Partial Tile Boundary Conditions**  
   - Edge cases for last Q tile (rows_in_tile < TILE_M) or last KV tile (kv_len < TILE_N)
   - **Test**: Run with S=17 (partial tiles) and S=32 (exact fit)

---

## 🧪 Recommended Next Steps

### Immediate (2 hours)
1. **Add Cross-Block Sync After Online Softmax Update**
   ```cuda
   // After step 2 (online softmax), before step 3 (materialize P)
   __syncwarp();  // Already present
   __syncthreads();  // ADD THIS — ensure all warps see updated m_smem
   ```

2. **Enable DEBUG_PRINT with Tiny Test (S=32, B=1, H=1)**
   - Modify `build.py` to read `DEBUG_PRINT` env var (currently ignored)
   - Print: `kv_local`, `m_row[0]`, `m_smem[0]`, `l_smem[0]`, first few P values
   - Compare with Stage-2 scalar path

3. **Verify WMMA_ACCUM_LUT Coverage**
   - Print LUT on host: `scripts/generate_wmma_lut.py`
   - Confirm: 32 lanes × 8 elements = 256 entries cover 16×16 tile
   - Check: Each (row, col) appears exactly once

### Fallback (if no progress in 2 hours)
4. **Revert to Stage-2 + Document as "Valid Negative Result"**
   - Stage-2 (cp.async + WMMA P·V): **656 μs baseline**
   - Merge Stage-2 to main, tag `v2.0-stage2-wmma-pv` (already done)
   - Document Stage-3B as "attempted fusion; correctness gate failed after 2 fixes; requires deeper investigation"

5. **Pivot to Stage-4 (3-Stage cp.async)**
   - Simpler optimization (no new math, just pipelining)
   - Expected: +5-10% speedup, low risk
   - Time: 4-6 hours

---

## 📁 Artifacts

### Logs
```
results/fp8_wmma_baseline/20251020-201008/  # Stage-2 control
results/fp8_wmma_baseline/20251020-201033/  # Stage-3B (stale fix)
.build_s2.log, .build_s3b.log
.corr_s2.log, .corr_s3b.log
```

### Commits
- `c9e5edd`: Session 3 checkpoint (fused softmax impl, 0/6 tests)
- `c374200`: KV mask + dynamic fragment size + block sync
- `8130696`: Zero ENTIRE sP tile (not just invalid columns)

### Code
- Kernel: `cudadent42/bench/kernels/sdpa_fp8_stage_c_wmma.cu` (lines 451-550, 826-925)
- LUT: `cudadent42/bench/kernels/wmma16x16_accum_lut.h`
- Runner: `tasks/fp8_sdpa_stage_c_wmma/runner.py`

---

## 🎯 Success Criteria (Stage-3B)

| Gate | Threshold | Current | Status |
|------|-----------|---------|--------|
| PTXAS | ≤128 regs, ≤48 KB SMEM, 0 spills | 83 regs, 35 KB, 0 spills | ✅ PASS |
| Correctness | 6/6 tests, max_err ≤ 0.06 | 0/6 tests, max_err = 4.1 | ❌ FAIL |
| Performance | p50 ≤ 590 μs (≥+10% vs 656 μs) | Not measured (blocked by correctness) | ⏸ BLOCKED |

**Decision Point**: If correctness not fixed in next session → revert to Stage-2, document, and move to Stage-4 or Stage-5.

---

**Last Updated**: 2025-10-20 20:15 UTC  
**Next Session**: Debug cross-warp sync + WMMA LUT verification

