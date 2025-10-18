# Child-V2c Status: WMMA Implementation In Progress

**Date**: October 18, 2025  
**Status**: ⚙️ **BUILDS, NEEDS DEBUG** (launch failure, expected for first iteration)  

---

## 🎯 Session Achievements

### ✅ V2b: Correctness Validated (COMPLETE)
- **5/5 acceptance tests passed** (100% correctness)
- **2 bugs found and fixed** systematically (SMEM overflow, warp 7 NaN)
- **TDD methodology validated** (Red → Green achieved)
- **Baseline**: 2452 μs on mission shape (73× slower than PyTorch, expected for scalar)

### ⚙️ V2c: WMMA Skeleton Created (IN PROGRESS)
- **Builds successfully** ✅ (47-48 registers, excellent)
- **Implements**: Real WMMA 16×16×16, K transpose + XOR swizzle, dropped S_scores
- **Status**: Launch failure (debugging needed)
- **Expected**: Multiple iterations to correctness (WMMA is complex)

---

## 📊 V2c Implementation Details

### What Was Implemented

1. **SMEM Layout Redesign**:
   ```
   sQ:      [M][HEAD_DIM_PAD] half (row-major, matrix_a)
   sK:      [HEAD_DIM_PAD][STAGES*N] half (col-major, matrix_b)  ← Transposed!
   sV:      [STAGES*N][HEAD_DIM_PAD] half (row-major)
   O_accum: [M][HEAD_DIM_PAD] float (FP32 accumulator)
   
   Dropped: S_scores buffer (16 KB saved)
   Total: ~63 KB (d=64, STAGES=2) < 99 KB ✅
   ```

2. **XOR Swizzle for K**:
   ```cuda
   int xor_swizzle(int n, int k) {
       int n_blk = n >> 3;
       int n_in = n & 7;
       int k_blk = k >> 3;
       return (n_blk ^ k_blk) * 8 + n_in;
   }
   ```
   Purpose: Eliminate bank conflicts for ldmatrix loads

3. **Real WMMA**:
   ```cuda
   // Q @ K^T
   wmma::fragment<wmma::matrix_a, 16,16,16, half, row_major> a_frag;
   wmma::fragment<wmma::matrix_b, 16,16,16, half, col_major> b_frag;
   wmma::fragment<wmma::accumulator,16,16,16, float> c_frag;
   
   wmma::load_matrix_sync(a_frag, Q_ptr, HEAD_DIM_PAD);
   wmma::load_matrix_sync(b_frag, K_ptr, STAGES*N);  // col-major stride
   wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
   ```

4. **cp.async Fix**:
   - Ada (sm_89) only supports 16B cp.async.cg, not 8B or 4B
   - Fallback to scalar loads if not 16B-aligned

### Resource Usage

```
Registers: 47-48/thread (< 72 target ✅)
SMEM: 63 KB (d=64) < 99 KB ✅
Stack: 0 bytes ✅
Spills: 0 ✅
```

---

## 🐛 Known Issues (Debugging Needed)

### Issue: Launch Failure
**Error**: `CUDA error: unspecified launch failure`
**Status**: Expected for first WMMA iteration

### Likely Causes

1. **WMMA Fragment Handling**:
   - `wmma::store_matrix_sync(&scores[0][0], c_frag, ...)` may not work with local array
   - Need proper stride and memory layout

2. **Score Extraction**:
   - Current code tries to store fragment to local array `float scores[16][16]`
   - WMMA fragments are not directly addressable like arrays

3. **P@V Accumulation**:
   - Converting scores to P fragment may have bugs
   - O_accum load/store logic might be incorrect

4. **SMEM Addressing**:
   - K col-major indexing: `k * (STAGES*N) + n_swizzled`
   - Might have off-by-one or stride errors

### Debugging Strategy

1. **Add printf debugging** for basic sanity:
   ```cuda
   if (threadIdx.x == 0 && blockIdx.x == 0) {
       printf("Warp %d: num_m_tiles=%d, num_n_tiles=%d\n", 
              warp_id, num_m_tiles, num_n_tiles);
   }
   ```

2. **Test individual WMMA**:
   - Create minimal kernel that just does Q@K^T WMMA
   - Validate score values before softmax

3. **Simplify fragment handling**:
   - Instead of storing to local array, use SMEM temporary
   - Or refactor to avoid materializing scores

4. **Use cuda-gdb**:
   - Build with `-G` flag
   - Run with `cuda-gdb` to find exact failure point

---

## 📈 Expected Path Forward

### Short-Term (4-6 hours)

**Goal**: Get V2c to 100% correctness (like V2b)

**Approach**:
1. Debug launch failure (2-3 hours)
2. Fix WMMA fragment handling (1-2 hours)
3. Validate correctness on 5 test shapes (1 hour)

**Expected Performance** (once correct):
- Mission shape: 400-800 μs (3-6× from V2b's 2452 μs)
- vs PyTorch: 12-24× slower (approaching competitive range)

### Medium-Term (2-3 hours)

**Goal**: Extract NCU insights (I3) and apply

**Approach**:
1. Profile with NCU (Nsight Compute)
2. Extract tensor core utilization, bank conflicts, DRAM %
3. Apply 1-2 high-leverage optimizations

**Expected Performance**:
- Mission shape: 200-400 μs (2× from corrected V2c)

### Long-Term (3-4 hours)

**Goal**: Elite loop (Top-K=3) with 2-lever changes

**Approach**:
1. Run 3-5 iterations of EvoEngineer-Full
2. Try: persistent CTAs, micro-tile reshape, pipeline depth, etc.
3. Keep best 3 kernels

**Expected Performance**:
- Mission shape: 100-200 μs (best case)
- vs PyTorch: 3-6× slower (production-competitive)

---

## 🎓 Key Learnings

### 1. Ada cp.async Constraints
**Discovery**: sm_89 only supports 16B cp.async.cg, not 8B or 4B
**Impact**: Must use 16B-aligned copies or fallback to scalar
**Source**: PTX error during build

### 2. WMMA Fragment Complexity
**Challenge**: Fragments are not directly addressable like arrays
**Learning**: Need careful load/store with proper strides
**Next**: Simplify fragment handling or use SMEM temporaries

### 3. TDD for CUDA Works
**V2b**: Correctness-first (scalar) → 100% correct in 2 iterations
**V2c**: Performance (WMMA) → Debugging needed (expected)
**Philosophy**: Green before Fast (proven right)

### 4. Register Usage Excellent
**V2b**: 56-60 regs (scalar)
**V2c**: 47-48 regs (WMMA)
**Result**: WMMA is *more* register-efficient than scalar!

---

## 📚 References for Debugging

### WMMA Documentation
- [NVIDIA WMMA Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#wmma)
- Fragment types, load/store semantics, stride requirements

### CUTLASS Patterns
- [CUTLASS GitHub](https://github.com/NVIDIA/cutlass)
- See `gemm/` and `attention/` for WMMA patterns

### XOR Swizzle
- [Lei Mao's Blog](https://leimao.github.io/article/CUDA-Shared-Memory-Bank/)
- Bank conflict analysis and swizzle patterns

### cp.async
- [NVIDIA cp.async Docs](https://docs.nvidia.com/cuda/parallel-thread-execution/#data-movement-and-conversion-instructions-cp-async)
- Size constraints (4/8/16 B on Ampere, 16 B only on Ada)

---

## ✅ Definition of Done (V2c)

### MUST HAVE (Not Yet Met)
- [ ] Builds without errors ✅ (Done)
- [ ] Launches without errors ❌ (Debugging)
- [ ] 5/5 acceptance tests pass (correctness)
- [ ] Faster than V2b baseline (2452 μs → <1000 μs)

### NICE TO HAVE
- [ ] Tensor core utilization >30% (NCU)
- [ ] Bank conflicts near zero (NCU)
- [ ] 3× faster than V2b (target: 400-800 μs)

### OUT OF SCOPE (V2d+)
- [ ] Beat PyTorch SDPA (~31 μs)
- [ ] < 100 μs (requires elite loop)

---

## 🚀 Quick Commands for Debugging

### Enable CUDA Error Reporting
```bash
export CUDA_LAUNCH_BLOCKING=1
export TORCH_USE_CUDA_DSA=1
```

### Build with Debug Symbols
```python
# In bench_sdpa.py:
extra_cuda_cflags=["-G", "-arch=sm_89", ...]
```

### Run with cuda-gdb
```bash
cuda-gdb --args python -c "from bench_sdpa import ..."
```

### Add printf Debugging
```cuda
if (threadIdx.x == 0 && blockIdx.x == 0 && blockIdx.y == 0) {
    printf("[V2c DEBUG] warp_id=%d, my_row_start=%d, my_num_rows=%d\n",
           warp_id, my_row_start, my_num_rows);
}
```

---

## 💬 Session Summary

**Time Spent**: ~5 hours (V2b validation + V2c skeleton)
**Achievements**:
- ✅ V2b: 100% correctness (5/5 tests, 2 bugs fixed)
- ⚙️ V2c: Skeleton builds (WMMA, transpose, swizzle)
- 📚 Learned: Ada cp.async constraints, register efficiency

**Cumulative Progress**:
- Phase A-C: 18 hours
- EvoEngineer V2b: 4 hours
- EvoEngineer V2c (partial): 1 hour
- **Total**: 23 hours

**Next**: Debug V2c launch failure → Correctness → Performance

**Status**: Excellent progress. TDD working. V2c debugging is expected and manageable.

---

**Last Update**: Oct 18, 2025  
**Commit**: `8337b99`  
**Status**: ⚙️ V2c builds, needs debug (2-4 hour iteration expected)


