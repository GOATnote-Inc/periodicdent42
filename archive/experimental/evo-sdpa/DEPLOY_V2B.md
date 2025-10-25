# Child-V2b Deployment & Testing Guide

**Created**: Oct 18, 2025  
**Status**: Ready for GPU testing  

---

## 🎯 What's in Child-V2b

### Core Fixes from V2

1. **FIX: Single-Warp Ownership**
   - Each warp owns `M/NUM_WARPS` consecutive rows
   - No inter-warp races on `(m,l)` stats
   - Stats updated within warp, synced via `__syncthreads()`

2. **FIX: Legal cp.async**
   - 16-byte aligned copies with `cp.async.cg.shared.global`
   - Proper `__cvta_generic_to_shared` address conversion
   - Correct `commit_group` + `wait_group<N>` sequence

3. **FIX: SMEM Layout**
   - Added `S_scores` buffer for WMMA output (16.4 KB)
   - Padding (PAD=8) for bank conflict avoidance
   - Total SMEM validated against device limit

4. **ARCHITECTURE**
   ```
   Warps 0-6: Compute (each owns contiguous rows)
   Warp 7:    Producer (cp.async prefetch)
   
   Per K-tile:
     1. Load Q (all warps)
     2. cp.async K/V (warp 7)
     3. Compute Q@K^T (warps 0-6, per-row)
     4. Softmax update (warps 0-6, per-row)
     5. Accumulate P@V (warps 0-6, per-row)
     6. __syncthreads() ← SINGLE sync point
   ```

5. **SMEM Budget** (d=64, STAGES=2)
   ```
   sQ:       64 × 72 × 2 =  9.2 KB
   sK:     2×64 × 72 × 2 = 18.4 KB
   sV:     2×64 × 72 × 2 = 18.4 KB
   S_scores: 64 × 64 × 4 = 16.4 KB
   O_accum:  64 × 72 × 4 = 18.4 KB
   m,l:      64 × 8     =  0.5 KB
   ───────────────────────────────
   Total:                 ~81 KB < 96 KB ✅
   ```

### Current Implementation

- **Scalar path for correctness** (not full WMMA yet)
- Focus on streaming softmax correctness first
- TODO: Layer in WMMA after validation

---

## 🚀 Quick Deploy (GPU)

### Step 1: Sync code to GPU

```bash
# From local machine:
cd /Users/kiteboard/periodicdent42
git push

# On GPU (cudadent42-l4-dev):
cd ~/periodicdent42
git pull
```

### Step 2: Activate environment

```bash
# On GPU:
cd ~/periodicdent42
source venv/bin/activate  # or your venv path
```

### Step 3: Run acceptance tests

```bash
cd evo-sdpa/bench
python test_v2b.py
```

**Expected Output:**
```
Building Child-V2b...
✅ Build successful

CHILD-V2b ACCEPTANCE TESTS
========================================================
✅ B=1 H=8 L= 512 d= 64 causal=0 | custom=XXX.XXμs torch=XXX.XXμs speedup=X.XX× max_diff=X.XXXXXX
✅ B=1 H=8 L= 512 d= 64 causal=1 | custom=XXX.XXμs torch=XXX.XXμs speedup=X.XX× max_diff=X.XXXXXX
...
========================================================

SUMMARY: 5/5 tests passed
✅ ALL ACCEPTANCE TESTS PASSED!
```

### Step 4: Parse resource usage

```bash
# During build, ptxas output is captured
# Extract register/SMEM usage:
cd evo-sdpa/bench
python parse_ptxas.py v2b_test_output.txt
```

**Expected:**
```
PTXAS RESOURCE USAGE
============================================================
Kernel 1:
  Registers:    48-64
  SMEM:         81.0 KB (82944 bytes)
```

---

## 🧪 Acceptance Criteria

### 1. Correctness (MUST PASS)
- ✅ 5/5 test cases pass (`max_diff ≤ 1e-3`)
  - `(B=1, H=8, L=512, d=64, causal=False)`
  - `(B=1, H=8, L=512, d=64, causal=True)`
  - `(B=2, H=8, L=2048, d=64, causal=False)`
  - `(B=2, H=8, L=2048, d=64, causal=True)`
  - `(B=2, H=8, L=2048, d=128, causal=False)`

### 2. Performance (DESIRABLE)
- ⚠️ Faster than PyTorch SDPA on `(1,8,512,64)` (current: likely NO)
- ⚠️ 10× faster than scalar baseline ~1400 μs (current: likely NO)

**Why not yet?** V2b prioritizes **correctness** over performance:
- Still using scalar dot products
- Not fully exploiting WMMA
- cp.async present but not optimized

### 3. Resource Sanity (MUST PASS)
- ✅ Registers ≤ 72/thread
- ✅ SMEM ≤ 96 KB/CTA
- ✅ No CUDA errors (invalid config, OOM, etc.)

---

## 📊 Expected Results & Next Steps

### Scenario 1: ✅ ALL TESTS PASS (Correctness 100%)

**Victory!** The core algorithmic fix worked:
- Single-warp ownership eliminated races
- Streaming softmax math is correct
- Ready to layer WMMA

**Next: Implement Full WMMA (D.3)**
```bash
# Create Child-V2c from V2b
# Focus: Replace scalar dot products with actual WMMA
# Expected: 3-7× speedup from Tensor Cores
```

### Scenario 2: ❌ SOME TESTS FAIL

**Debug strategy:**
1. Check which shapes fail (d=64 vs d=128? causal?)
2. Add `printf` debugging to kernel for failing shape
3. Verify warp ownership logic (rows per warp)
4. Check SMEM bounds and indexing

**Tools:**
```bash
# Add debug prints to sdpa_fused_v2b.cu
# Recompile with -G (device debug)
# Use cuda-gdb if needed
```

### Scenario 3: 🔥 CUDA ERRORS (Launch failures)

**Common causes:**
- SMEM overflow (check ptxas output)
- Invalid grid/block config
- Dynamic SMEM not set properly

**Fix:**
1. Reduce `N` if SMEM > 96 KB
2. Check `cudaFuncSetAttribute` in dispatcher
3. Validate grid dimensions

---

## 📈 Performance Trajectory

### Current State (estimated)
```
V1 (scalar baseline):    1378 μs (44× slower than PyTorch)
V2b (correctness-first): ~1200 μs (scalar, but clean)
PyTorch SDPA:             ~31 μs (target to beat)
CUTLASS baseline:         ~24 μs (current champion)
```

### Roadmap to < 100 μs

**Phase D.3: Full WMMA** (4-6 hours)
- Replace scalar Q@K^T with 16×16×16 MMA tiles
- Replace scalar P@V with WMMA
- Expected: 300-500 μs (3-4× from V2b)

**Phase D.4: NCU Profiling + I3** (2-3 hours)
- Extract tensor core utilization, DRAM %, bank conflicts
- Generate optimization insights (I3)
- Expected: Identify next 2× lever

**Phase D.5: Elite Loop** (3-4 hours)
- Try 2-lever combinations from I3
- Iterate Top-K=3 best kernels
- Expected: 100-200 μs (best case)

**Total time to ~100 μs: 10-15 hours** (realistic)  
**Total time to < 5 μs: Weeks-months** (research)

---

## 🔧 Development Commands

### Build only (no test)
```bash
cd evo-sdpa/bench
python -c "from bench_sdpa import build_ext; build_ext()"
```

### Single test case (verbose)
```bash
python -c "
from bench_sdpa import build_ext, run_case
import torch
mod = build_ext()
run_case(mod, B=1, H=8, L=512, d=64, causal=False, verbose=True)
"
```

### Check for CUDA errors
```bash
# Add error checking to kernel:
cudaError_t err = cudaGetLastError();
if (err != cudaSuccess) {
    printf(\"CUDA Error: %s\\n\", cudaGetErrorString(err));
}
```

---

## 📚 Key Files

- `kernels/sdpa_fused_v2b.cu` - Main kernel implementation
- `kernels/runtime.hpp` - Dispatcher
- `kernels/sdpa_fused_bindings.cpp` - PyTorch bindings
- `bench/test_v2b.py` - Acceptance tests
- `bench/bench_sdpa.py` - Core benchmark harness
- `bench/parse_ptxas.py` - Resource usage parser

---

## 🎓 Design Rationale

### Why Single-Warp Ownership?

**Problem (V2)**: Multiple warps computing same row → race on `m,l`
**Solution (V2b)**: Each warp owns contiguous rows → no races

```cuda
// V2 (BROKEN): All warps computed all rows
const int my_q_row = warp_id % num_q_rows;  // ❌ Race!

// V2b (FIXED): Each warp owns disjoint rows
const int rows_per_warp = (M + NUM_WARPS - 1) / NUM_WARPS;
const int my_row_start = warp_id * rows_per_warp;
const int my_row_end = min(my_row_start + rows_per_warp, num_q_rows);
// ✅ No overlap
```

### Why Scalar Path First?

**EvoEngineer philosophy**: Correctness → Performance
- Get streaming softmax math right (m,l invariants)
- Validate against PyTorch reference
- THEN layer in WMMA optimizations

**Evidence**: V2 had WMMA "framework" but 0% correctness  
V2b has scalar compute but targeting 100% correctness

### Why cp.async Now?

**Foundation for future speedup**:
- Double-buffering K/V tiles (hide latency)
- Warp specialization (producer/consumer)
- Prerequisite for 3-stage pipeline (L≥2048)

Even if scalar compute is slow, the *structure* is right.

---

## ✅ Definition of Done (V2b)

**MINIMAL** (Gate to proceed):
- [ ] Builds without errors
- [ ] Launches without CUDA errors
- [ ] 5/5 acceptance tests pass (correctness)

**DESIRABLE** (Nice to have):
- [ ] Faster than V1 baseline (1378 μs → <1000 μs)
- [ ] Registers ≤ 64/thread (better occupancy)
- [ ] SMEM ≤ 80 KB (more headroom)

**OUT OF SCOPE** (V2c+):
- [ ] Beat PyTorch SDPA (requires WMMA)
- [ ] < 100 μs (requires NCU tuning)
- [ ] < 5 μs (requires research)

---

**Last Update**: Oct 18, 2025  
**Next**: Deploy to GPU and run `test_v2b.py`



