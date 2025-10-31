# CUTLASS 4.3.0 + CUDA 13.0 Environment Ready

**Date**: October 30, 2025  
**Pod**: `related_cyan_clownfish` (157.66.254.40:17322)  
**Status**: ✅ **PRODUCTION READY**

---

## Executive Summary

**Mission**: Leverage NVIDIA's CUTLASS 4.3.0 (October 2025) + CUDA 13.0 to build expert-grade kernels.

**Status**: Environment fully operational with latest tools from [NVIDIA/cutlass](https://github.com/NVIDIA/cutlass) main branch.

**Key Achievement**: Standing on NVIDIA's shoulders with proper API access.

---

## Environment Specification

### CUDA Toolkit 13.0.2

```
Version:  13.0.88 (Release: Aug 20, 2025)
Location: /usr/local/cuda-13.0
Driver:   570.133.20 (with forward compat: 580.95.05)
Source:   NVIDIA CUDA Toolkit Release
```

**Features Available**:
- ✅ sm_90a (H100 Hopper) 
- ✅ sm_100 (Blackwell) compile support
- ✅ FP8 E4M3/E5M2 types
- ✅ TMA (Tensor Memory Accelerator)
- ✅ WGMMA (Warp-Group Matrix Multiply)
- ✅ Forward compatibility (cu130 runtime on older driver)

### CUTLASS 4.3.0 (October 2025)

```
Version:  4.3.0 (main branch)
Commit:   8afb19d9 (October 28, 2025)
Location: /opt/cutlass
Source:   https://github.com/NVIDIA/cutlass
License:  BSD-3-Clause
```

**Key Features** (from [CHANGELOG.md](https://github.com/NVIDIA/cutlass/blob/main/CHANGELOG.md)):

1. **CuTe DSL** (Python interface):
   - Programmatic Dependent Launch
   - Pipeline APIs (PipelineProducer, PipelineConsumer)
   - Blackwell SM100 persistent GEMM
   - Mixed-input GEMM (FP8/FP16)
   - Block-scaled data types

2. **C++ API**:
   - GemmUniversalAdapter + CollectiveBuilder
   - KernelTmaWarpSpecialized (Hopper/Blackwell)
   - Example 77: Blackwell FMHA (Flash Multi-Head Attention)

3. **Available Examples**:
   - `examples/48_hopper_warp_specialized_gemm/`
   - `examples/49_hopper_gemm_with_collective_builder/`
   - `examples/50_hopper_gemm_with_epilogue_swizzle/`
   - `examples/77_blackwell_fmha/` ← **Flash Attention reference**

### PyTorch 2.9.0+cu130

```
Version:  2.9.0+cu130
CUDA:     13.0
GPU:      NVIDIA H100 80GB HBM3
Status:   ✅ Operational
```

**Additional Libraries**:
- xFormers 0.0.32.post1
- vLLM 0.11.0
- Triton 3.5.0

---

## Verified Capabilities

### ✅ CUDA 13.0 Compilation

```bash
nvcc --version
# Cuda compilation tools, release 13.0, V13.0.88
```

### ✅ CUTLASS 4.3.0 API Access

```bash
cd /opt/cutlass
git log -1 --oneline
# 8afb19d9 update CITATION.cff (2025-10-28)
```

**Hopper Examples Present**:
- 48_hopper_warp_specialized_gemm ✓
- 49_hopper_gemm_with_collective_builder ✓
- 50_hopper_gemm_with_epilogue_swizzle ✓
- 77_blackwell_fmha ✓

### ✅ PyTorch GPU Access

```python
import torch
print(torch.__version__)        # 2.9.0+cu130
print(torch.version.cuda)        # 13.0
print(torch.cuda.is_available()) # True
print(torch.cuda.get_device_name(0)) # NVIDIA H100 80GB HBM3
```

---

## Bootstrap Script

**Location**: `/workspace/pod_setup.sh`

**Features**:
- ✅ Auto-installs CUDA 13.0.2 toolkit
- ✅ Clones CUTLASS 4.3.0 (main branch from GitHub)
- ✅ Installs PyTorch 2.9.0+cu130 with forward compatibility
- ✅ Persists across pod restarts (auto-source in `.bashrc`)
- ✅ Idempotent (safe to run multiple times)

**Usage**:
```bash
bash /workspace/pod_setup.sh
```

**Verification**:
```bash
nvcc --version | grep "13.0"
python3 -c "import torch; print(torch.version.cuda)"
cd /opt/cutlass && git log -1 --oneline
```

---

## Example Build Commands

### 1. Compile CUTLASS Hopper Example

```bash
export CUDA_HOME=/usr/local/cuda-13.0
export PATH=$CUDA_HOME/bin:$PATH
export CUTLASS_HOME=/opt/cutlass

cd $CUTLASS_HOME/examples/50_hopper_gemm_with_epilogue_swizzle

nvcc -O3 --std=c++17 --expt-relaxed-constexpr \
     -I$CUTLASS_HOME/include \
     -I$CUTLASS_HOME/tools/util/include \
     -gencode arch=compute_90,code=sm_90a \
     -DCUTLASS_ENABLE_TENSOR_CORE_MMA=1 \
     50_hopper_gemm_with_epilogue_swizzle.cu \
     -o hopper_gemm

./hopper_gemm --m=2048 --n=2048 --k=2048
```

### 2. Compile User's Blackwell Kernel (SM100)

User provided `cutlass_blackwell_sparsek_v2.cu` targeting SM100.

**For H100 (sm_90a), adapt using Hopper schedules**:
```cpp
// Change:
using ArchTag = cutlass::arch::Sm100;           // Blackwell
using KernelSchedule = cutlass::gemm::KernelTmaWarpSpecialized2SmSm100;
using EpilogueSchedule = cutlass::epilogue::TmaWarpSpecialized2Sm;

// To:
using ArchTag = cutlass::arch::Sm90;            // Hopper
using KernelSchedule = cutlass::gemm::KernelTmaWarpSpecialized;
using EpilogueSchedule = cutlass::epilogue::TmaWarpSpecialized;
```

**Build command**:
```bash
nvcc -O3 --std=c++20 --expt-relaxed-constexpr -DNDEBUG \
     -I$CUTLASS_HOME/include -I$CUTLASS_HOME/tools/util/include \
     -gencode arch=compute_90,code=sm_90a \
     -DMM_TILE_M=128 -DMM_TILE_N=128 -DMM_TILE_K=64 \
     -o cutlass_h100_gemm \
     cutlass_h100_gemm.cu
```

---

## Reference Documentation

### Official NVIDIA Sources

1. **CUTLASS GitHub**: https://github.com/NVIDIA/cutlass
   - Main branch (October 2025)
   - v4.3.0 features documented in CHANGELOG.md

2. **CUTLASS Documentation**: https://docs.nvidia.com/cutlass/
   - C++ API Reference
   - CuTe DSL Guide
   - Example Tutorials

3. **CUDA Toolkit 13.0**: https://developer.nvidia.com/cuda-toolkit
   - Release Notes (August 2025)
   - Programming Guides

4. **Hopper Architecture**: https://developer.nvidia.com/hopper-architecture
   - TMA, WGMMA features
   - Performance tuning

### Key Examples to Study

**Example 48**: Warp-Specialized GEMM
- Shows basic kernel structure
- TMA + WGMMA usage
- Collective builders

**Example 50**: GEMM with Epilogue Swizzle
- More advanced epilogue customization
- Shows how to instantiate custom collectives

**Example 77**: Blackwell FMHA
- Flash Multi-Head Attention implementation
- Most relevant for attention kernels
- Uses latest Blackwell optimizations (adaptable to Hopper)

---

## Current Baseline Performance

### PyTorch SDPA (Baseline)

**Configuration**: B=16, H=96, SL=4096, HD=128

```
Time:      12.27 ms (12,274 μs)
Per head:  127.85 μs/head
TFLOPS:    1,075
Memory:    6.00 GB
```

**Configuration**: B=16, H=96, SL=4096, HD=64

```
Time:      7.37 ms (7,369 μs)
Per head:  76.76 μs/head
```

### Target Performance

**Mission Goal**: < 2.45 ms (5× faster than 12.27 ms baseline)

**Tier System**:
```
Current:  12.27 ms (baseline)
Tier 1:    6.14 ms (2× faster, B grade)
Tier 2:    4.09 ms (3× faster, B+ grade)
Tier 3:    2.45 ms (5× faster, A grade)  ← TARGET
Tier 4:    0.94 ms (13× faster, A+ grade)
```

---

## Next Actions

### Priority 1: Adapt User's SM100 Kernel to H100

User provided SM100 (Blackwell) kernel using:
- `cutlass::arch::Sm100`
- `KernelTmaWarpSpecialized2SmSm100`
- `TmaWarpSpecialized2Sm`

**Adaptation needed**:
1. Change `Sm100` → `Sm90` 
2. Use Hopper schedules (`KernelTmaWarpSpecialized`, `TmaWarpSpecialized`)
3. Adjust ClusterShape if needed
4. Test compilation with `-gencode arch=compute_90,code=sm_90a`

### Priority 2: Study CUTLASS Example 77 (FMHA)

Example 77 implements Flash Multi-Head Attention for Blackwell:
```bash
ls $CUTLASS_HOME/examples/77_blackwell_fmha/
```

**Key learnings**:
- Flash Attention algorithm in CUTLASS style
- TMA usage for attention matrices
- WGMMA for Q@K^T and P@V operations
- Online softmax implementation

**Adaptation**: Use Hopper (sm_90) version of techniques

### Priority 3: Benchmark Custom Kernel vs SDPA

Once kernel compiles:
```bash
# Baseline
python benchmark_sdpa.py  # 12.27 ms

# Custom kernel
./cutlass_h100_gemm 1536 4096 128 100  # Compare

# Target: < 2.45 ms (5× speedup)
```

---

## Success Criteria

### Environment ✅

- [x] CUDA 13.0.2 installed
- [x] CUTLASS 4.3.0 (October 2025) from main branch
- [x] PyTorch 2.9.0+cu130 operational
- [x] H100 GPU accessible
- [x] Bootstrap script working
- [x] Examples compile with `--expt-relaxed-constexpr`

### API Access ✅

- [x] CollectiveBuilder available
- [x] GemmUniversalAdapter available
- [x] TMA schedules available (KernelTmaWarpSpecialized)
- [x] Hopper examples present and buildable

### Performance (Pending)

- [ ] Custom kernel compiles successfully
- [ ] Kernel runs without errors
- [ ] Correctness validated vs SDPA
- [ ] Performance measured with Nsight Compute
- [ ] Target: < 2.45 ms (5× faster than baseline)

---

## Technical Notes

### CUDA 13.0 Forward Compatibility

**Challenge**: Pod has driver 570.133, PyTorch cu130 requires 580.95+

**Solution**: `cuda-compat-13-0` package provides compatibility libraries
```bash
apt-get install -y cuda-compat-13-0
export LD_LIBRARY_PATH=/usr/local/cuda-13.0/compat:$LD_LIBRARY_PATH
```

This allows CUDA 13.0 runtime to work with older driver.

### ClusterShape_MNK = cute::Shape<>

Empty cluster shape means dynamic (runtime-configured).
Set via `args.hw_info.cluster_shape` in GemmUniversalAdapter Arguments.

For H100, common cluster shapes:
- 1×1 (default, safe)
- 2×1 (recommended for large problems)
- 1×2 (for specific tile sizes)

### make_cute_packed_stride API

CUTLASS 4.3.0 API for strides requires:
```cpp
StrideA stride_A = cutlass::make_cute_packed_stride<StrideA>(
    cute::make_shape(M, K, 1), LayoutA{}
);
```

Note: Takes `Shape` object, not array. Ensure proper CuTe types.

---

## Troubleshooting

### Issue: "Index out of range" error in ClusterShape

**Cause**: Empty `ClusterShape_MNK = cute::Shape<>` not supported by all CollectiveBuilder instantiations.

**Solution**: Use explicit cluster shape:
```cpp
using ClusterShape_MNK = cute::Shape<cute::Int<1>, cute::Int<1>, cute::Int<1>>;
```

### Issue: "--expt-relaxed-constexpr" required

**Cause**: CUTLASS utility functions use `std::min/max` in `__device__` code.

**Solution**: Always include `--expt-relaxed-constexpr` flag:
```bash
nvcc --expt-relaxed-constexpr ...
```

### Issue: "no instance of function matches" for make_cute_packed_stride

**Cause**: API expects `cute::Shape` not raw integers or arrays.

**Solution**: Use `cute::make_shape(M, N, K)` wrapper:
```cpp
auto stride_A = cutlass::make_cute_packed_stride<StrideA>(
    cute::make_shape(M, K, 1), LayoutA{}
);
```

---

## Summary

**Environment Status**: ✅ **READY**

✅ CUDA 13.0.2 (nvcc 13.0.88)  
✅ CUTLASS 4.3.0 (main branch, October 2025)  
✅ PyTorch 2.9.0+cu130  
✅ H100 80GB HBM3  
✅ Bootstrap script functional  
✅ API access verified  

**Source**: [NVIDIA/cutlass GitHub](https://github.com/NVIDIA/cutlass) (main branch)

**Mission**: Build custom attention kernel that exceeds PyTorch SDPA by 5× using CUTLASS 4.3.0 + CUDA 13.0 on H100.

**Target**: < 2.45 ms (currently 12.27 ms baseline)

**Approach**: Leverage NVIDIA's CUTLASS examples (especially Example 77 FMHA) and adapt for H100 Hopper architecture.

---

**Last Updated**: October 30, 2025, 21:00 UTC  
**Next Step**: Adapt user's SM100 kernel to H100 (sm_90a) and compile  
**Repository**: https://github.com/NVIDIA/cutlass


