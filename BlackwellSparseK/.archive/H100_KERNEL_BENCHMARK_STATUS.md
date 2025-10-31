# H100 Kernel Benchmark Status

**Date**: October 30, 2025, 19:00 UTC  
**Pod**: `related_cyan_clownfish` (157.66.254.40:17322)  
**Configuration**: CUDA 13.0 + PyTorch 2.9.0+cu130 + H100

---

## Executive Summary

✅ **Environment**: PyTorch 2.9.0+cu130 operational on H100  
✅ **Baseline**: 12.27 ms (127.85 μs/head, 1,075 TFLOPS) for D=128  
⚠️ **FlashCore Triton**: Slower than baseline (needs optimization)  
⏳ **Native CUDA Kernels**: Not yet compiled

---

## Benchmark Results

### Configuration 1: D=128 (Original)

```
Method: PyTorch SDPA
B=16, H=96, SL=4096, HD=128

Results:
├─ Time: 12.27 ms per iteration
├─ Per head: 127.85 μs/head
├─ TFLOPS: 1,075
└─ Memory: 6.00 GB

Status: ✅ Baseline established
```

### Configuration 2: D=64 (FlashCore Compatible)

```
Workload: B=16, H=96, SL=4096, HD=64

PyTorch SDPA (Baseline):
├─ Time: 7.37 ms
├─ Per head: 76.76 μs/head
└─ Status: ✅ Baseline

FlashCore Triton (attention_production.py):
├─ Time: 50.78 ms
├─ Per head: 528.97 μs/head
├─ Speedup: 0.15× (6.9× SLOWER)
├─ Correctness: ⚠️ Max diff 4.45 (FAIL)
└─ Status: ❌ Not production-ready
```

---

## Available Kernels

### Triton-Based (Python)

Located in `flashcore/fast/`:

1. **attention_production.py** - Production kernel
   - Status: ✅ Imports successfully
   - Limitation: D=64 only
   - Performance: 6.9× slower than SDPA ❌
   - Correctness: FAIL (max diff 4.45) ❌

2. **attention_stage2_optimized.py** - Optimized scheduling
   - Status: Not tested
   - Features: Improved load/compute overlap

3. **attention_stage3_persistent.py** - Persistent CTAs
   - Status: Not tested
   - Features: Amortize launch overhead

4. **attention_stage5_warpspec.py** - Warp specialization
   - Status: Not tested
   - Features: Producer/consumer warps, fast exp

### Native CUDA Kernels (.cu files)

Located in `flashcore/fast/`:

**Hopper-Specific** (H100):
- `attention_hopper_cuda.cu` - Base Hopper kernel
- `attention_hopper_minimal.cu` - Minimal implementation
- `attention_hopper_tma.cu` - TMA (Tensor Memory Accelerator)

**Phase Kernels** (Progressive optimization):
- `attention_phase2_aggressive.cu` - Aggressive optimization
- `attention_phase2_async.cu` - Async memory ops
- `attention_phase3_wgmma.cu` - WGMMA (Warp-Group Matrix Multiply)
- `attention_phase4_fused.cu` - Fused operations
- `attention_phase4x_expert.cu` - Expert-level optimization
- `attention_phase5_wgmma.cu` - Advanced WGMMA
- `attention_phase6_*.cu` - Latest optimizations (multi, pipeline, native)

**Status**: ❌ Not compiled yet (need PyTorch C++ extension build)

---

## Root Cause Analysis

### Why FlashCore Triton is Slower

1. **Triton Compilation Overhead**: First run includes compilation
   - Solution: Proper warmup (already done, but still slow)

2. **Block Size Tuning**: May not be optimal for H100
   - Default blocks may not match H100 SMs

3. **D=64 Limitation**: Kernel not optimized for general dimensions
   - Restricts use cases

4. **Correctness Issues**: Max diff 4.45 suggests algorithm error
   - May be using approximations that don't converge

### Path to 5× Speedup

**Target**: < 2.45 ms (12.27 ms / 5)

**Options**:

1. **Compile Native CUDA Kernels** (Highest Potential)
   - Use H100-specific features: TMA, WGMMA, warp specialization
   - Expected speedup: 2-10× over SDPA
   - Complexity: Medium (need proper C++ extension build)

2. **Optimize Triton Kernels** (Medium Potential)
   - Tune block sizes with `tune_block_sizes_h100.py`
   - Try stage2/stage3/stage5 variants
   - Expected speedup: 1.5-3× over SDPA (if working correctly)
   - Complexity: Low (just Python imports)

3. **Use PyTorch 2.0 compile()** (Low Potential)
   - `torch.compile(attention_fn)`
   - Expected speedup: 1.2-1.5× over SDPA
   - Complexity: Very low

---

## Recommended Action Plan

### Phase 1: Quick Wins (< 30 min)

1. **Test other Triton variants**:
   ```bash
   from flashcore.fast.attention_stage2_optimized import attention as attn_stage2
   from flashcore.fast.attention_stage3_persistent import attention as attn_stage3
   from flashcore.fast.attention_stage5_warpspec import attention as attn_stage5
   ```

2. **Tune block sizes**:
   ```bash
   python flashcore/fast/tune_block_sizes_h100.py
   ```

3. **Try torch.compile()**:
   ```python
   import torch
   sdpa_compiled = torch.compile(F.scaled_dot_product_attention)
   ```

### Phase 2: Native CUDA Kernels (< 2 hours)

1. **Build Phase 6 WGMMA kernel** (most advanced):
   ```bash
   cd /workspace/BlackwellSparseK
   
   # Create setup for phase6 kernel
   cat > setup_phase6.py << 'EOF'
   from setuptools import setup
   from torch.utils.cpp_extension import CUDAExtension, BuildExtension
   
   setup(
       name='flashcore_phase6',
       ext_modules=[
           CUDAExtension(
               name='flashcore_phase6',
               sources=['flashcore/fast/attention_phase6_wgmma_native.cu'],
               extra_compile_args={
                   'nvcc': [
                       '-O3',
                       '-gencode', 'arch=compute_90,code=sm_90a',
                       '--use_fast_math',
                       '-U__CUDA_NO_HALF_OPERATORS__',
                       '-U__CUDA_NO_BFLOAT16_CONVERSIONS__',
                       f'-I{CUTLASS_HOME}/include',
                       '--expt-relaxed-constexpr',
                   ]
               }
           )
       ],
       cmdclass={'build_ext': BuildExtension}
   )
   EOF
   
   python setup_phase6.py install
   ```

2. **Benchmark native kernel vs SDPA**

3. **Iterate with Nsight Compute profiling**

### Phase 3: Expert Optimization (< 1 day)

1. **Profile with Nsight Compute**:
   ```bash
   ncu --set full python benchmark.py
   ```

2. **Apply EvoEngineer methodology**:
   - Identify bottlenecks (SM utilization, memory bandwidth)
   - Mutate kernel parameters (block sizes, warp specialization)
   - Preserve elites (best performers)

3. **Target metrics**:
   - SM utilization: > 80%
   - Memory throughput: > 2 TB/s (H100 theoretical: 3.35 TB/s)
   - Warp occupancy: > 50%

---

## Current Bottlenecks

### Environment: ✅ SOLVED
- CUDA 13.0 toolkit: ✅ Working
- PyTorch 2.9.0+cu130: ✅ Working (via forward compat)
- GPU access: ✅ Working

### Kernels: ⚠️ IN PROGRESS
- Triton kernels: ⚠️ Available but slower/incorrect
- Native CUDA: ❌ Not compiled yet
- Optimization: ❌ Not started

### Benchmarking: ⏳ PARTIAL
- SDPA baseline: ✅ Established (12.27 ms)
- FlashCore: ⚠️ Tested but slower
- Comprehensive comparison: ❌ Pending native kernels

---

## Performance Targets

### Tier System (from mission spec)

```
Current Baseline (SDPA D=128): 12.27 ms

Tier 1 (Good):        6.14 ms (2× faster)    B grade
Tier 2 (Very Good):   4.09 ms (3× faster)    B+ grade
Tier 3 (Excellent):   2.45 ms (5× faster)    A grade  ← TARGET
Tier 4 (Outstanding): 0.94 ms (13× faster)   A+ grade
Tier 5 (Breakthrough): 0.19 ms (64× faster)  A++ grade
```

**Current Status**: Below Tier 1 (FlashCore Triton is slower)

**Next Milestone**: Tier 1 (2× faster = 6.14 ms)

---

## Technical Notes

### CUDA 13.0 Features Available

With CUDA 13.0 toolkit, we can use:
- ✅ Tile-based programming model
- ✅ sm_90a instructions (H100 Hopper)
- ✅ Latest compiler optimizations
- ✅ Compatibility with Tensor Cores

### H100 Architecture Features

**Should be leveraging**:
- **TMA (Tensor Memory Accelerator)**: Async global → shared memory
- **WGMMA**: Warp-group matrix multiply (128 threads)
- **Warp specialization**: Producer/consumer overlap
- **Persistent CTAs**: Amortize launch overhead
- **FP16 Tensor Cores**: 989 TFLOPS theoretical (FP16)

**Current Utilization**: Unknown (need Nsight profiling)

---

## Files for Reference

### Benchmark Scripts Created

- `/tmp/baseline_bench.py` - SDPA baseline (D=128)
- `/tmp/flashcore_bench_d64.py` - FlashCore vs SDPA (D=64)

### Key Repository Files

- `flashcore/fast/attention_production.py` - Production Triton kernel
- `flashcore/fast/attention_phase6_wgmma_native.cu` - Most advanced CUDA kernel
- `flashcore/fast/tune_block_sizes_h100.py` - Auto-tuning script
- `setup.py` - Main build script (needs CUDA extension config)

---

## Immediate Next Steps

**Priority 1**: Test other Triton variants (stage2, stage3, stage5)
- Low effort, might find better implementation
- Command: `python -m flashcore.fast.attention_stage5_warpspec`

**Priority 2**: Compile native CUDA kernels
- High potential for speedup
- Focus on `attention_phase6_wgmma_native.cu`

**Priority 3**: Profile with Nsight Compute
- Understand bottlenecks
- Guide optimization efforts

---

## Summary

**Status**: ✅ Environment ready, ⏳ Kernels need compilation/optimization

**Baseline**: 12.27 ms (SDPA, D=128, H100)

**Target**: 2.45 ms (5× faster, Tier 3)

**Gap**: Need 5× speedup through:
1. Proper kernel compilation (native CUDA)
2. H100-specific optimizations (TMA, WGMMA)
3. Profiling-guided tuning

**Recommendation**: Compile `attention_phase6_wgmma_native.cu` as next action

---

**Last Updated**: October 30, 2025, 19:00 UTC  
**Environment**: ✅ Operational  
**Next Action**: Compile native CUDA kernels or test Triton stage2/5 variants

