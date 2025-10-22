# FlashCore WMMA Unified Architecture - Debug Status

**Date**: October 22, 2025  
**Status**: 🚨 **CORRECTNESS FAILURE** - Debugging in progress

---

## 🎯 Architecture Overview

**Clean separation**:
- `flashcore_v6_wmma_qkt.cu`: Q·K^T kernel (64×64 tiles, cp.async, vectorized)
- `flashcore_v7_1_wmma_pv.cu`: P·V kernel (64×64 tiles, cp.async, vectorized)
- `flashcore_bindings.cpp`: PyTorch C++ bindings
- `flashcore_wmma_common.cuh`: Shared utilities (cp.async, WMMA constants)
- `build_wmma.py`: Unified build script
- `test_wmma.py`: Comprehensive test suite

---

## 📊 Current Results (L4 GPU)

### Compilation
✅ **SUCCESS**: All files compile with sm_89 targeting

### Correctness (FAIL)
```
QK^T: Max error = 11,253,572,608 (❌ CATASTROPHIC)
P·V:  Max error = 0.430420        (❌ FAIL >= 0.05)
```

---

## 🔍 Root Cause Analysis

### QK^T Kernel (`v6_wmma_qkt.cu`)

**Problem**: WMMA fragment layout mismatch

**Current code** (lines 85-87):
```cuda
const half* q_ptr = q_tile + tile_m * kTileK + k;
const half* k_ptr = k_tile + tile_n * kTileK + k;
nvcuda::wmma::load_matrix_sync(a_frag, q_ptr, kTileK);
nvcuda::wmma::load_matrix_sync(b_frag, k_ptr, kTileK);  // col_major
```

**Issue**: 
- We load K as row-major (`[N][K]`)
- But declare `matrix_b` as `col_major`
- This expects K^T layout in memory (`[K][N]`)
- Result: We're multiplying `Q @ K` instead of `Q @ K^T`!

**Fix Options**:
1. **Option A**: Load K tile transposed in shared memory
2. **Option B**: Change `matrix_b` to `row_major` and adjust pointer arithmetic
3. **Option C**: Use `matrix_a` for K^T (flipped dimensions)

### P·V Kernel (`v7_1_wmma_pv.cu`)

**Problem**: Accumulation across tiles

**Current code** (lines 177-179):
```cuda
mma_pv(p_tile, v_tile, &shared.accum[0]);
```

**Issue**:
- Each tile writes to `accum` with `store_matrix_sync`
- This **overwrites** previous tile results instead of accumulating!
- We need to:
  1. Load existing accum into fragment
  2. MMA into that fragment
  3. Store back

**Fix**: Proper accumulation loop

---

## 🛠️ Fix Plan

### Priority 1: QK^T (Highest Impact)
**Target**: Get error < 0.05

**Fix**: Use `row_major` for `matrix_b` and adjust pointers

```cuda
// Change line 80:
nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, kWmmaM, kWmmaN, kWmmaK, half, nvcuda::wmma::row_major> b_frag;

// Change K pointer arithmetic (line 86):
const half* k_ptr = k_tile + k * kTileN + tile_n;  // K^T indexing
```

### Priority 2: P·V (Moderate Impact)
**Target**: Get error < 0.05

**Fix**: Proper accumulation

```cuda
// Load existing accumulator
nvcuda::wmma::fragment<nvcuda::wmma::accumulator, kWmmaM, kWmmaN, kWmmaK, float> c_frag;
if (tile == 0) {
    nvcuda::wmma::fill_fragment(c_frag, 0.0f);
} else {
    nvcuda::wmma::load_matrix_sync(c_frag, dst, kTileD, nvcuda::wmma::mem_row_major);
}

// Accumulate
for (int k = 0; k < kTileN; k += kWmmaK) {
    // ... mma_sync adds to c_frag
}

// Store back
nvcuda::wmma::store_matrix_sync(dst, c_frag, kTileD, nvcuda::wmma::mem_row_major);
```

---

## 📈 Expected Results After Fix

### Correctness
```
QK^T: Max error < 0.05 ✅
P·V:  Max error < 0.05 ✅
```

### Performance (Estimates)
```
QK^T:  ~80-120 μs (WMMA + cp.async + vectorized)
P·V:   ~80-120 μs (WMMA + cp.async + vectorized)
Total: ~160-240 μs (unfused, but still 2-3× faster than v5's 2122 μs)
```

### Path to <40 μs
1. Fix correctness first (current step)
2. Profile with NCU (identify bottlenecks)
3. Optimize tile sizes (64×64 → 128×128?)
4. Add fused softmax (eliminate P matrix write)
5. Warp specialization (producer/consumer)

---

## 🚀 Next Steps

1. **Fix QK^T**: Change `matrix_b` to `row_major` + pointer arithmetic
2. **Fix P·V**: Proper accumulation across tiles
3. **Test**: Re-run `test_wmma.py` on L4
4. **Profile**: Once correct, measure actual latency
5. **Optimize**: Iterate to <40 μs

---

**Status**: Ready for fixes 
**Confidence**: High (root causes identified)  
**ETA**: 30-60 minutes to validated correctness

