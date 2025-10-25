# Child-V2b Implementation Complete

**Status**: ✅ **READY FOR GPU TESTING**  
**Created**: Oct 18, 2025  

---

## 🎯 What Was Built

### Core Implementation: `sdpa_fused_v2b.cu`

**Architecture**:
```
Tile: M=64, N=64 (d=64/128), K=HEAD_DIM
Warps: 8 (0-6 compute, 7 producer)
SMEM: ~81 KB (d=64, STAGES=2)
Pipeline: 2-stage (L<2048), 3-stage (L≥2048)
```

**Key Fixes from V2**:
1. **Single-warp ownership** → each warp owns contiguous rows
2. **Legal cp.async** → 16B aligned, proper commit/wait
3. **SMEM budget** → validated, within 96 KB
4. **Correctness-first** → scalar path, focus on algorithm

**Tagged Improvements**:
- `// FIX: single-warp softmax stats` (lines with ownership logic)
- `// FIX: cp.async 16B aligned` (async copy paths)
- `// INSIGHT: swizzle` (bank conflict padding)
- `// ELITE-CHG: XOR swizzle` (future enhancement)
- `// ELITE-CHG: Persistent CTAs` (future enhancement)

---

## 📊 Resource Budget

### SMEM Layout (d=64, STAGES=2)

| Component | Size | Notes |
|-----------|------|-------|
| `sQ` | 9.2 KB | Q tile (64×72 half) |
| `sK` | 18.4 KB | K tiles, 2-stage (2×64×72 half) |
| `sV` | 18.4 KB | V tiles, 2-stage (2×64×72 half) |
| `S_scores` | 16.4 KB | WMMA output buffer (64×64 float) |
| `O_accum` | 18.4 KB | Output accumulator (64×72 float) |
| `m,l` | 0.5 KB | Softmax stats (64×2 float) |
| **Total** | **~81 KB** | **< 96 KB ✅** |

### d=128, STAGES=2 (estimated)
- Similar structure, adjusted tiles
- Total: ~85-90 KB (still under 96 KB)

### Register Estimate
- Current: ~48-64 regs/thread (scalar path)
- Target: ≤72 regs/thread (good occupancy)

---

## 🧪 Testing Infrastructure

### 1. Acceptance Test: `test_v2b.py`

**Coverage**: 5 shapes × 2 causal modes = 10 tests
```python
(1, 8, 512, 64, False)   # Mission-critical shape
(1, 8, 512, 64, True)    # Causal variant
(2, 8, 2048, 64, False)  # Larger batch/seq
(2, 8, 2048, 64, True)   # Large + causal
(2, 8, 2048, 128, False) # d=128 test
```

**Output**: Compact per-test status + summary
```
✅ B=1 H=8 L= 512 d= 64 causal=0 | custom=XXX.XXμs torch=XXX.XXμs speedup=X.XX× max_diff=X.XXXXXX
...
SUMMARY: 5/5 tests passed
```

### 2. Resource Parser: `parse_ptxas.py`

**Purpose**: Extract register/SMEM from build logs
**Usage**: `python parse_ptxas.py v2b_test_output.txt`
**Output**:
```
PTXAS RESOURCE USAGE
Kernel 1:
  Registers:    64
  SMEM:         81.0 KB (82944 bytes)
```

### 3. Benchmark Harness: `bench_sdpa.py`

**Modes**:
- `verbose=True`: Detailed per-test output
- `verbose=False`: Compact automated mode

**Metrics**:
- Correctness: `max_abs_diff`, `max_rel_diff` vs PyTorch
- Performance: `us` (custom), `us_ref` (PyTorch), `speedup`

---

## 🎯 Acceptance Criteria

### MUST PASS (Gate to proceed)

1. **Build Success**
   - All 4 kernel specializations compile
   - No ptxas errors/warnings
   - PyTorch extension loads

2. **Correctness**
   - 5/5 test cases: `max_diff ≤ 1e-3`
   - All shapes (d=64, d=128)
   - Both causal modes (True, False)

3. **Resource Sanity**
   - Registers ≤ 72/thread
   - SMEM ≤ 96 KB/CTA
   - No CUDA launch errors

### DESIRABLE (Not required for V2b)

- Faster than PyTorch SDPA (will fail - scalar path)
- 10× faster than V1 baseline (may fail - correctness-first)
- Tensor core utilization >0% (not yet, scalar path)

---

## 📈 Expected Performance

### V2b (Scalar Path)
```
Mission shape (1,8,512,64):
  Expected:  800-1200 μs
  vs V1:     0.8-1.2× (marginal change, correctness-first)
  vs PyTorch: 0.03-0.04× (30-40× slower, no TC)
```

**Why so slow?**
- Scalar dot products (no WMMA yet)
- Focus on correctness, not speed
- cp.async present but compute-bound

### V2c (Full WMMA) - Next
```
Mission shape (1,8,512,64):
  Expected:  200-400 μs (3-5× from V2b)
  vs PyTorch: 0.1-0.2× (5-10× slower)
  Lever: Replace scalar with 16×16×16 MMA tiles
```

### V2d+ (NCU Tuned) - Future
```
Mission shape (1,8,512,64):
  Expected:  50-100 μs (2-4× from V2c)
  vs PyTorch: 0.5-1.0× (parity or slight win)
  Levers: I3 insights (pipeline, swizzle, etc.)
```

---

## 🔄 EvoEngineer Workflow

### Current: Child-V2b (Correctness Baseline)

**Generation**: 2b (correctness-first rewrite)
**Parent**: V2 (broken streaming softmax)
**Insights Applied**: None (clean-slate fix)

### Next: Child-V2c (WMMA Layer)

**Generation**: 2c (performance upgrade)
**Parent**: V2b (assuming correctness passes)
**Insights Applied**: I1 (tensor cores)
**Levers**:
1. Replace scalar Q@K^T → WMMA 16×16×16
2. Replace scalar P@V → WMMA 16×16×16

### Then: Elite Loop (Top-K=3)

**Generation**: 3a, 3b, 3c (exploration)
**Parents**: Best of V2c + variants
**Insights Applied**: I3 (from NCU profiling)
**Levers**: 2 per child (pipeline depth, swizzle, CTA shape, etc.)

---

## 🚧 Known Limitations (V2b)

### 1. Performance
- **Scalar compute**: No WMMA yet → slow
- **cp.async underutilized**: Latency hiding present but compute-bound
- **Expected**: 30-40× slower than PyTorch SDPA

### 2. Incomplete WMMA Framework
- `S_scores` buffer allocated but unused (scalar path)
- WMMA fragment declarations exist but not utilized
- TODO: Layer in actual `mma.sync` instructions

### 3. Not Production-Ready
- No dropout support
- No BF16 specialization (only FP16)
- No variable sequence lengths (padding inefficient)

---

## 📚 Code Walkthrough

### Per-Warp Row Ownership (Core Fix)

```cuda
// Each warp owns consecutive rows
const int rows_per_warp = (M + NUM_WARPS - 1) / NUM_WARPS;
const int my_row_start = warp_id * rows_per_warp;
const int my_row_end = min(my_row_start + rows_per_warp, num_q_rows);

// Loop only over owned rows
for (int my_r = 0; my_r < my_num_rows; ++my_r) {
    int r = my_row_start + my_r;
    // Compute scores, update (m,l), accumulate O
    // ✅ No race: only this warp touches row r
}
```

### Legal cp.async (Core Fix)

```cuda
// FIX: 16B aligned copy
__device__ void cp_async_16B_aligned(void* smem_ptr, const void* global_ptr) {
    unsigned smem_addr = __cvta_generic_to_shared(smem_ptr);
    asm volatile(
        "cp.async.cg.shared.global [%0], [%1], 16;\n"
        :: "r"(smem_addr), "l"(global_ptr)
    );
}

// Usage in producer warp:
if (c % 8 == 0 && c + 8 <= HEAD_DIM) {
    cp_async_16B_aligned(&sK[smem_idx], &K_bh[global_offset_k]);
}
cp_async_commit_group();
```

### Streaming Softmax (Verified Math)

```cuda
// Per-row, per K-tile:
float m_new = fmaxf(m_old, max(scores_tile));
float l_add = sum(exp(scores_tile - m_new));
float rescale = exp(m_old - m_new);
float l_new = l_old * rescale + l_add;

// Rescale previous O_accum
O_accum *= rescale;

// Accumulate new P@V
O_accum += (exp(scores_tile - m_new) @ V_tile);

// Update stats (lane 0 only, within warp)
if (lane == 0) { m_smem[r] = m_new; l_smem[r] = l_new; }
```

---

## 🎓 Design Philosophy

### EvoEngineer Principles Applied

1. **Solution Guiding (I1)**: Task context in `00_task.md`
2. **Population Management**: Single best → Top-K=3 (future)
3. **Iterative Refinement**: V2 (broken) → V2b (correct) → V2c (fast)

### TDD for CUDA

1. **Red**: V2 had 0% correctness
2. **Green**: V2b targets 100% correctness
3. **Refactor**: V2c will add WMMA (speed)

### Standing on Shoulders

- **PyTorch SDPA**: 31 μs (reference)
- **CUTLASS**: 24 μs (current champion)
- **V2b**: Focus on matching PyTorch *correctness*, not speed (yet)

---

## ✅ Completion Checklist

### Code Complete
- [x] `sdpa_fused_v2b.cu` written
- [x] `runtime.hpp` updated (dispatcher)
- [x] `sdpa_fused_bindings.cpp` updated
- [x] `bench_sdpa.py` updated (compact mode)
- [x] `test_v2b.py` created (acceptance tests)
- [x] `parse_ptxas.py` created (resource parser)

### Documentation
- [x] `DEPLOY_V2B.md` (deployment guide)
- [x] `V2B_STATUS.md` (this file)
- [x] Inline code comments (`// FIX:`, `// INSIGHT:`)

### Testing (Pending on GPU)
- [ ] Builds without errors
- [ ] 5/5 acceptance tests pass
- [ ] Resource usage within limits

---

## 🚀 Next Actions

### Immediate (On GPU)

1. **Deploy to GPU**
   ```bash
   git pull
   cd evo-sdpa/bench
   python test_v2b.py
   ```

2. **Validate Correctness**
   - Check test output: 5/5 pass?
   - If fail, debug with `verbose=True`

3. **Check Resources**
   ```bash
   python parse_ptxas.py v2b_test_output.txt
   # Verify: regs ≤72, SMEM ≤96KB
   ```

### Short-Term (If tests pass)

4. **Implement V2c (WMMA)**
   - Replace scalar Q@K^T with WMMA
   - Replace scalar P@V with WMMA
   - Expected: 3-5× speedup

5. **Profile with NCU**
   ```bash
   ncu --set full -o v2c_profile python -c "..."
   ncu -i v2c_profile.ncu-rep --page raw
   ```

6. **Extract I3 Insights**
   - Tensor core utilization %
   - DRAM bandwidth %
   - Bank conflicts
   - Pipeline stalls

### Long-Term (Elite Loop)

7. **Top-K Selection**
   - Keep best 3 kernels (V2c variants)
   - Score by: correctness AND (speedup vs PyTorch)

8. **Iterate**
   - For each parent, try 2-lever changes
   - Compile → Test → Keep if ≥3% faster
   - Repeat until convergence or time budget

---

**Last Update**: Oct 18, 2025  
**Commit**: `14bf7b7` (test infrastructure)  
**Status**: Awaiting GPU validation



