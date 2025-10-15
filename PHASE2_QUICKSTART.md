# Phase 2 Quick Start - WMMA + SMEM Q Tile
## Implementation Guide for V3 Kernel Fix

---

## Objective

Fix WMMA local memory issue by moving Q tile to shared memory.

**Goal**: Enable Tensor Cores while maintaining stability and correctness.

**Target Performance**: ≥ scalar baseline (0.038ms), goal: ≥ 0.8× SDPA (0.018ms)

---

## Prerequisites

✅ **Phase 1 Complete** (merged in main)
- Root cause identified
- Scalar baseline established (0.038ms)
- Evidence documented
- Fix strategy defined

---

## One-Command Start

```bash
cd /Users/kiteboard/periodicdent42 && \
git checkout main && \
git pull origin main && \
git checkout -b feature/wmma_smem_phase2 && \
echo "✅ Phase 2 branch ready!"
```

---

## Implementation Steps

### Step 1: Update Shared Memory Structure

**File**: `cudadent42/bench/kernels/fa_s512_v3.cu`

**Location**: `SharedMemory` struct (~line 50)

**Change**:
```cpp
template<typename Traits>
struct SharedMemory {
    __align__(16) half K[Traits::STAGES][Traits::BLOCK_N][Traits::K_STRIDE];
    __align__(16) half V[Traits::STAGES][Traits::BLOCK_N][Traits::V_STRIDE];
    __align__(16) float O_accum[Traits::BLOCK_M][Traits::HEAD_DIM];
    
#if defined(USE_WMMA)
    __align__(16) half Q_tile[Traits::BLOCK_M][Traits::HEAD_DIM];  // ADD THIS
#endif
};
```

**Rationale**: WMMA requires data in shared memory or global memory, not registers.

---

### Step 2: Cooperative Q Load to SMEM

**File**: `cudadent42/bench/kernels/fa_s512_v3.cu`

**Location**: After Q_reg loading (~line 200)

**Add**:
```cpp
#if defined(USE_WMMA)
    // Cooperatively store Q_reg to SMEM for WMMA
    for (int i = 0; i < rows_per_warp; ++i) {
        const int row = row_start + i;
        if (row < Traits::BLOCK_M) {
            for (int d = lane_id; d < Traits::HEAD_DIM; d += 32) {
                smem->Q_tile[row][d] = Q_reg[i][d];
            }
        }
    }
    __syncthreads();  // Ensure all Q data in SMEM before WMMA
#endif
```

**Rationale**: All threads cooperatively fill Q_tile in SMEM.

---

### Step 3: Update WMMA Call

**File**: `cudadent42/bench/kernels/fa_s512_v3.cu`

**Location**: QK^T WMMA call (~line 250)

**Change**:
```cpp
#if defined(USE_WMMA)
    qk_row_wmma<Traits>(
        &smem->Q_tile[row_start + local_row][0],  // CHANGE: SMEM (not Q_reg)
        &smem->K[stage][0][0],
        S_row,
        softmax_scale,
        Traits::BLOCK_N
    );
#else
    qk_row_scalar<Traits>(
        Q_reg[local_row],
        &smem->K[stage][0][0],
        S_row,
        softmax_scale,
        Traits::BLOCK_N
    );
#endif
```

**Rationale**: Point WMMA at SMEM Q_tile instead of Q_reg.

---

### Step 4: Update WMMA Function Signature

**File**: `cudadent42/bench/kernels/fa_s512_v3.cu`

**Location**: `qk_row_wmma` function (~line 100)

**Change**:
```cpp
template<typename Traits>
__device__ __forceinline__ void qk_row_wmma(
    const half* Q_smem,  // CHANGE: was half Q_row[Traits::HEAD_DIM]
    const half* K_smem,
    float* S_row,
    const float scale,
    const int num_cols
) {
    // ... WMMA implementation uses Q_smem directly ...
}
```

**Rationale**: WMMA loads from SMEM pointer, not register array.

---

### Step 5: Update Build Script

**File**: `cudadent42/bench/build_v3_release.py`

**Location**: Build flags section (~line 40)

**Verify** (should already be present):
```python
# USE_WMMA toggle from environment
if os.environ.get("USE_WMMA", "1") != "0":
    extra_cuda_cflags.append("-DUSE_WMMA")
```

No changes needed (USE_WMMA flag already added in Phase 1).

---

## Build & Test Sequence

### 1. Build with WMMA Enabled
```bash
cd cudadent42/bench && \
export USE_WMMA=1 && \
python3 build_v3_release.py
```

**Expected**: Build succeeds, PTXAS shows WMMA usage.

### 2. Run Sanitizer (Critical!)
```bash
compute-sanitizer --tool memcheck \
  python3 tests/oracles/tile_oracle_v3.py \
  --config_id 1
```

**Expected**: 0 errors (vs 3389 in Phase 1)

### 3. Quick Parity Test
```bash
cd cudadent42/bench/tests && \
python3 oracles/tile_oracle_v3.py --config_id 1
```

**Expected**: "Passed" with max error < 0.01

### 4. Benchmark
```bash
cd /Users/kiteboard/periodicdent42 && \
python3 scripts/bench_s512_tc_vs_sdpa.py
```

**Expected**: 
- WMMA runs without errors
- p50 latency: ≥ 0.038ms (scalar baseline)
- Goal: ≥ 0.018ms (0.8× SDPA)

### 5. Nsight Profile
```bash
ncu --set full --kernel-name fa_s512_v3_kernel \
  python3 scripts/bench_s512_tc_vs_sdpa.py
```

**Check**:
- Tensor Core utilization > 0%
- SM busy ≥ 70%
- No local memory spills

---

## Success Criteria

### Minimum (Must Pass)
- ✅ Build succeeds with USE_WMMA=1
- ✅ Sanitizer: 0 errors
- ✅ Parity test passes (error < 0.01)
- ✅ Benchmark completes without runtime errors
- ✅ Performance ≥ scalar baseline (0.038ms)

### Target (Phase 2 Goal)
- ✅ Tensor Core utilization > 0%
- ✅ Performance ≥ 0.8× SDPA (0.018ms)
- ✅ Stable across 100 iterations

### Stretch (Phase 3)
- ⏳ Performance ≥ SDPA (0.022ms)
- ⏳ Multiple configs optimized
- ⏳ Production-ready

---

## Debugging Guide

### If Sanitizer Still Shows Errors

**Check 1**: Q_tile allocated in SMEM?
```bash
grep -A 5 "struct SharedMemory" cudadent42/bench/kernels/fa_s512_v3.cu
# Should show Q_tile[BLOCK_M][HEAD_DIM] inside #if defined(USE_WMMA)
```

**Check 2**: Q_reg copied to SMEM before WMMA?
```bash
grep -B 5 "__syncthreads()" cudadent42/bench/kernels/fa_s512_v3.cu | grep Q_tile
# Should show cooperative store to Q_tile
```

**Check 3**: WMMA uses SMEM pointer?
```bash
grep "qk_row_wmma" cudadent42/bench/kernels/fa_s512_v3.cu | head -3
# Should show &smem->Q_tile[...] as first argument
```

### If Performance is Worse Than Scalar

**Hypothesis 1**: SMEM bank conflicts
- **Check**: Nsight Compute → SMEM stats
- **Fix**: Pad Q_tile (use K_STRIDE pattern)

**Hypothesis 2**: Insufficient occupancy
- **Check**: Nsight → Occupancy
- **Fix**: Reduce register usage with --maxrregcount

**Hypothesis 3**: Excessive syncs
- **Check**: Remove extra __syncthreads()
- **Fix**: One sync after Q load only

**Hypothesis 4**: WMMA overhead > scalar benefit
- **Reality Check**: S=512, D=64 may be too small for WMMA
- **Alternative**: Try larger batch (B=4, H=16)

---

## Estimated Timeline

### Optimistic (2 hours)
- 30 min: Code changes (Steps 1-4)
- 15 min: Build & sanitizer validation
- 30 min: Parity & benchmark
- 15 min: Nsight profile
- 30 min: PR documentation

### Realistic (4 hours)
- 1 hour: Code changes + debugging
- 30 min: Sanitizer troubleshooting
- 1 hour: Performance tuning
- 30 min: Nsight analysis
- 1 hour: Documentation & PR

### Conservative (1 day)
- If WMMA doesn't improve over scalar
- Alternative: Try split-HEAD_DIM or larger shapes

---

## Files to Modify

### Core Implementation
1. `cudadent42/bench/kernels/fa_s512_v3.cu` (4 changes)
   - SharedMemory struct
   - Q load to SMEM
   - WMMA call site
   - qk_row_wmma signature

### Optional Enhancements
2. `cudadent42/bench/build_v3_release.py` (already done)
3. `cudadent42/bench/tests/oracles/tile_oracle_v3.py` (no changes)
4. `scripts/bench_s512_tc_vs_sdpa.py` (no changes)

**Total Lines Changed**: ~50 lines across 1 file

---

## GPU Cost Estimate

**Scenario**: Phase 2 implementation (4 hours)

- Debugging/dev: 2 hours @ $0.30/hour = $0.60
- Benchmarking: 1 hour @ $0.30/hour = $0.30
- Nsight profiling: 1 hour @ $0.30/hour = $0.30

**Total**: ~$1.20 (conservative)

**Project Total** (Phase 1+2): ~$2.40-3.60

---

## Expected Outcomes

### Best Case ✅
- Sanitizer: 0 errors
- Performance: 0.018ms (0.8× SDPA)
- Tensor Core utilization: 60-80%
- **Result**: Phase 2 complete, ready for optimization

### Good Case ✅
- Sanitizer: 0 errors
- Performance: 0.028ms (between scalar and SDPA)
- Tensor Core utilization: 20-40%
- **Result**: Phase 2 complete, Phase 3 optimization needed

### Acceptable Case ⚠️
- Sanitizer: 0 errors
- Performance: 0.038ms (matches scalar)
- Tensor Core utilization: 10-20%
- **Result**: WMMA overhead = scalar benefit (need larger shapes)

### Unacceptable Case ❌
- Sanitizer: errors remain
- **Action**: Deep dive with cuobjdump, verify SMEM layout

---

## Branch & PR Workflow

### 1. Create Branch
```bash
git checkout -b feature/wmma_smem_phase2
```

### 2. Make Changes
- Implement Steps 1-4
- Test with sanitizer
- Benchmark

### 3. Commit Pattern
```bash
git add cudadent42/bench/kernels/fa_s512_v3.cu
git commit -m "fix(v3): move Q tile to SMEM for WMMA compatibility

- Add Q_tile[BLOCK_M][HEAD_DIM] to SharedMemory
- Cooperative load Q_reg → SMEM before WMMA
- Update qk_row_wmma to use SMEM pointer
- Sanitizer: 0 errors (was 3389)
- Performance: TBD ms (target: ≥ 0.038ms)"
```

### 4. Benchmark Commit
```bash
git add cudadent42/artifacts/bench/
git commit -m "bench(v3): WMMA+SMEM baseline results

- p50: X.XXX ms (baseline: 0.038ms)
- Sanitizer: 0 errors ✅
- Tensor Core utilization: XX%
- [Performance assessment]"
```

### 5. Create PR
```bash
gh pr create \
  --base main \
  --head feature/wmma_smem_phase2 \
  --title "Phase 2: WMMA + SMEM Q tile implementation" \
  --body "$(cat PHASE2_PR_TEMPLATE.md)"
```

---

## PR Template Preview

```markdown
## Phase 2: WMMA + SMEM Q Tile

Fixes WMMA local memory issue identified in PR #63.

### Changes
- Move Q tile from registers to shared memory
- Enable proper WMMA usage for Tensor Cores
- Validate with compute-sanitizer (0 errors)

### Performance
- Baseline (scalar): 0.038ms
- This PR (WMMA): X.XXXms
- PyTorch SDPA: 0.022ms

### Validation
- ✅ Sanitizer: 0 errors
- ✅ Parity test: Passed
- ✅ 100 iterations: Stable
- ✅ Nsight: TC utilization XX%

### Files Modified
- `fa_s512_v3.cu`: SMEM Q tile implementation

### Next Steps
- Phase 3: EvoEngineer optimization (if needed)
```

---

## Quick Commands Reference

```bash
# Start Phase 2
git checkout -b feature/wmma_smem_phase2

# Build with WMMA
cd cudadent42/bench && USE_WMMA=1 python3 build_v3_release.py

# Sanitizer check
compute-sanitizer --tool memcheck python3 tests/oracles/tile_oracle_v3.py --config_id 1

# Benchmark
cd /Users/kiteboard/periodicdent42 && python3 scripts/bench_s512_tc_vs_sdpa.py

# Nsight profile
ncu --set full --kernel-name fa_s512_v3_kernel python3 scripts/bench_s512_tc_vs_sdpa.py

# Create PR
gh pr create --base main --head feature/wmma_smem_phase2
```

---

## Ready to Start?

**Prerequisites Met**: ✅
- Phase 1 merged
- Root cause understood
- Fix strategy clear

**Command to Begin**:
```bash
cd /Users/kiteboard/periodicdent42 && \
git checkout -b feature/wmma_smem_phase2 && \
echo "🚀 Phase 2 started! Edit cudadent42/bench/kernels/fa_s512_v3.cu"
```

**Expected Duration**: 2-4 hours  
**Expected Cost**: ~$1.20 GPU time  
**Success Criteria**: Sanitizer clean + performance ≥ scalar

---

**Ready when you are! 🚀**

---

**File**: `PHASE2_QUICKSTART.md`  
**Created**: October 15, 2025  
**Status**: Ready for Phase 2 implementation

