# BSR Sparse GEMM - Progress Report

## Performance Journey (H100 PCIe 80GB)

**Matrix: 8192×8192×8192, 87.5% sparse (BSR format)**

| Version | TFLOPS | Efficiency | Key Optimization |
|---------|--------|-----------|------------------|
| v1 (Atomics) | 30.7 | 5.0% | Naive atomic accumulation |
| v2 (Registers) | 61.5 | 10.0% | Register-based accumulation |
| **v3 (Optimized)** | **68.8** | **11.2%** | **512 threads, vectorized loads** |

**Progress: 2.24× improvement from baseline (30 → 68.8 TFLOPS)**

## Current Status (Nov 1, 2025)

✅ **Correct implementation** (validated vs CPU reference)  
✅ **2.24× faster than naive** (meaningful optimization)  
✅ **Arbitrary block-sparse** (fills gap vs CUTLASS 2:4-only)  
⚠️ **Not yet competitive** (need 150+ TFLOPS for 25% efficiency)

## What's Working

1. **Vectorized memory access** (`float4` loads, 8 halfs at once)
2. **Register accumulation** (no atomic overhead)
3. **512 threads per block** (better occupancy than 256)
4. **Half2 compute hints** (compiler vectorization)
5. **Aggressive unrolling** (`#pragma unroll 16`)

## What's Missing (Path to 150+ TFLOPS)

### Hardware Limitations
- **No tensor core utilization** - using scalar FP16 ops
- **WMMA accumulation bug** - store_matrix_sync overwrites instead of accumulating
- **Shared memory constraints** - 48KB limit prevents output buffering

### Potential Next Steps
1. **Fix WMMA properly** - load existing C, accumulate, store (complex)
2. **Use cuBLASLt per block** - proven fast, architectural fit
3. **Extend CUTLASS CollectiveBuilder** - leverage proven infrastructure
4. **Wait for TMA/WGMMA** - Blackwell features may help

## Validated Baselines (H100)

| Implementation | TFL OPS | Notes |
|----------------|---------|-------|
| cuBLAS Dense | 615 | FP16 tensor core ceiling |
| CUTLASS Ex 62 | 270 | 2:4 structured only |
| **Our BSR** | **69** | Arbitrary patterns, correct |
| PyTorch BSR | CRASH | Beta, non-square blocks fail |

## Key Learnings

1. **Progress is iterative** - 30 → 61 → 69 TFLOPS in 3 versions
2. **Correctness first** - multiple "fast" versions were wrong
3. **Architectural fit matters** - H100 optimized for dense/2:4
4. **Compiler help limited** - half2 hints don't auto-use tensor cores

## Next Session Goals

- [ ] Try persistent kernels (reduce launch overhead)
- [ ] Test cudaBLASLt per block approach
- [ ] Profile with actual NCU if possible
- [ ] Consider CUTLASS extension path

---

**Status:** Research in progress, meaningful optimization achieved  
**Hardware:** H100 PCIe 80GB, CUDA 13.0.2, CUTLASS 4.3.0  
**Date:** November 1, 2025
