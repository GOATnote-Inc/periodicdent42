# BSR Sparse GEMM - Honest Status (Nov 1, 2025)

## Current State

**H100 PCIe 80GB, 8192×8192×8192, 87.5% sparse**

| Implementation | TFLOPS | Efficiency | Status |
|----------------|--------|-----------|---------|
| cuBLAS Dense | 614 | 100% | ✅ Gold standard |
| CUTLASS Ex 62 | 270 | 44% | ✅ 2:4 structured only |
| **Our BSR** | **61.5** | **10%** | ✅ Correct, ❌ Slow |

## What Works

✅ **Correctness validated** (max error: 0.0 vs CPU reference)  
✅ **Arbitrary block-sparse** (not just 2:4 structured)  
✅ **No atomics** (register accumulation)  
✅ **Proper shared memory usage** (32KB, within limits)

## What's Missing

❌ **Tensor core utilization** - using FP16 scalar ops, not WMMA/WGMMA  
❌ **Memory coalescing** - suboptimal global memory access  
❌ **Occupancy** - not maximizing SM utilization  
❌ **Pipeline** - no overlapping compute/memory

## The Real Gap

**CUTLASS limitation is confirmed:**
- CUTLASS: Only 2:4 structured sparsity (270 TFLOPS)
- PyTorch: BSR crashes (beta, broken)
- Our BSR: Arbitrary patterns but slow (61 TFLOPS)

**To provide value, need:**
- 150+ TFLOPS (25% efficiency) = competitive
- 200+ TFLOPS (33% efficiency) = good
- 300+ TFLOPS (50% efficiency) = excellent

## Why Current Implementation is Slow

1. **No tensor cores:** Each thread does scalar FP16 multiply-adds  
   - Need: WMMA fragments or WGMMA instructions  
   - Impact: 5-10× speedup potential

2. **Poor memory patterns:** Random access to B matrix  
   - Need: Vectorized loads (float4), better tiling  
   - Impact: 2-3× speedup potential

3. **Low occupancy:** Only using 256 threads per block  
   - Need: Multiple blocks per SM, more parallelism  
   - Impact: 1.5-2× speedup potential

## Path Forward

### Option 1: Keep Optimizing Custom Kernel (High effort, uncertain)
- Add WMMA/WGMMA  
- Optimize memory access  
- Tune tile sizes  
- **Risk:** Weeks of work, may never reach 200+ TFLOPS

### Option 2: Extend CUTLASS (Smart, hard)
- Study CUTLASS `CollectiveMainloop`  
- Create `BlockSparseConfig` for BSR  
- Leverage proven infrastructure  
- **Benefit:** Likely to be correct and fast

### Option 3: Document Gap (Immediate value)
- Publish reproducer for PyTorch BSR crash  
- Document CUTLASS 2:4 limitation  
- File feature request with NVIDIA  
- **Benefit:** Community recognizes need

## Recommendation

**Document the gap honestly, don't claim false victory.**

The gap is real (arbitrary BSR needed), but filling it requires either:
1. Significant optimization effort (weeks), or  
2. Deep CUTLASS expertise (extension), or  
3. Waiting for NVIDIA to add BSR support

Current 61 TFLOPS kernel is **correct but not competitive**.

---

**Validated:** Nov 1, 2025  
**Hardware:** H100 PCIe 80GB  
**Status:** Research prototype, not production-ready
