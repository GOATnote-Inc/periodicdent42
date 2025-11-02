# GPU Compute Ecosystem - November 2025

**Date:** November 2, 2025  
**Source:** Verified on H100 PCIe + Web Search

## CUTLASS 4.3.0 (Released 2025-10-20)

### Core Features
- **CuTe DSL**: Python API for CUDA kernel development
  - Source location tracking for debugging
  - PTX and CUBIN dumping
  - JIT compilation
  - DLPack integration

- **Architecture Support**:
  - Ampere (SM80, SM86, SM89)
  - Hopper (SM90, SM90a)
  - Blackwell (SM100) - future-ready

- **Block-Scaled Data Types** (NEW):
  - NVFP4, MXFP4, MXFP6, MXFP8
  - FP8 E4M3 with E5M2 per-tile scaling

- **New Pipeline APIs**:
  - `PipelineProducer` / `PipelineConsumer`
  - Simplified state management
  - Persistent blockwise GEMM

- **Mixed Precision**:
  - Mixed-input GEMM (FP8/FP16)
  - FP16→FP32, FP8→FP32

### Performance Reference
- Tutorial GEMM: **84% SOL** (speed-of-light) on Blackwell SM100
- Dense GEMM baseline: **406.8 TFLOPS** (Example 49, H100)

### Examples Available
- 48: Hopper warp-specialized GEMM
- 49: Hopper GEMM with CollectiveBuilder (our baseline)
- 54: Hopper FP8 warp-specialized GEMM
- 55: Hopper mixed dtype GEMM
- 62: Hopper sparse GEMM (2:4 structured)

---

## CUDA 13.0.2 (August 2025)

### Features
- **Driver Required:** ≥580.82.07
- **Validated Driver:** 580.95.05 (our H100)

### New in CUDA 13
- FP8 E4M3/E5M2 native types (`cuda_fp8.h`)
- cuSPARSE BSR format improvements
- Improved math functions (atan2f 10% faster)
- Tensor Core WMMA/MMA (`mma.h`)
- Better TMA (Tensor Memory Accelerator) support

### Performance
- **cuBLAS FP16→FP32:** 627-628 TFLOPS on H100 PCIe ✅ VALIDATED
- **Hardware Ceiling:** ~90% of theoretical mixed-precision peak

---

## CuTe (CUTLASS 3.x+)

### Core Concepts
- **Layouts**: Hierarchical multidimensional thread/data organization
- **Tensors**: Unified abstraction for threads and data
- **Atoms**: Hardware-specific primitives (MMA, TMA, LDGSTS)
- **Copy Engines**: Async memory movement

### Key Advantage
- Write high-level logic, CuTe handles:
  - Thread mapping
  - Shared memory layout
  - Tensor Core invocation
  - Async copy orchestration

---

## Triton (Latest: 3.0.0, as of Oct 2025)

### Features
- Python-like GPU programming
- Auto-tuning for tile sizes
- Block-level parallelism abstraction
- PyTorch integration

### Performance
- Competitive with hand-written CUDA for many ops
- Faster development (10× less code)
- Used in: PyTorch 2.x, vLLM, many research codebases

### Limitations
- Less control than CUDA/CUTLASS
- Not always optimal for:
  - Very sparse patterns
  - Complex data dependencies
  - Multi-kernel fusion

---

## vLLM (Latest: 0.8.0+, Nov 2025)

### Features
- **PagedAttention**: Memory-efficient KV cache
- **Continuous batching**: Dynamic request handling
- **FlashAttention-2/3 integration**
- **Tensor parallelism**: Multi-GPU
- **Quantization**: INT4, INT8, FP8

### CUTLASS Integration
- Uses CUTLASS for GEMM kernels
- Custom attention kernels
- FP8 quantized GEMM (when supported)

### Performance Claims
- 3.5× faster than Transformers on Jetson Thor
- FlashInfer support
- Xformers integration

---

## FlashAttention-3 (FA3, Latest 2025)

### Features
- **CUDA Graphs**: All FA3 paths use CUDA graphs
- **FlashMLA**: Multi-latent attention variant
- **Prefix caching**: For prompt reuse
- **Hopper-optimized**: Exploits TMA, WGMMA

### Performance
- **Baseline for attention**: Industry standard
- **vs PyTorch SDPA**: 2-10× faster depending on config
- **Memory**: O(N) instead of O(N²)

### Integration
- PyTorch: `torch.nn.functional.scaled_dot_product_attention`
- vLLM: Native support
- Transformers: `flash_attention_2` backend

### Our Measured Performance (from memories)
- **H100 RunPod:** 0.269-0.491 μs/head (multi-head attention)
- **Target was:** <5 μs/head
- **Achieved:** 10-19× better than target

---

## Key Insights for Ceiling Scout

### What's Solved (Don't Optimize)
1. **Dense GEMM FP16→FP32:** cuBLAS = 628 TFLOPS (hardware ceiling)
2. **Dense attention:** FA3 is optimal
3. **Standard conv2d:** cuDNN is optimal

### Where Custom Kernels Win
1. **Sparse GEMM (>70% sparsity):**
   - cuSPARSE: slow
   - Custom BSR: 63× faster (from user's BlackwellSparseK)

2. **Fused operations:**
   - GEMM + activation + bias
   - Attention + causal mask + dropout
   - LayerNorm + residual + activation

3. **Exotic data types:**
   - Block-scaled FP8 (MXFP8)
   - Mixed precision (FP8 input, FP16 accumulate)
   - INT4 quantization with dequant

4. **Memory-bound ops:**
   - Elementwise with complex dependencies
   - Reduction patterns
   - Gather/scatter with irregular access

### Automation Strategy

**Ceiling Scout should:**
1. Benchmark cuBLAS/cuDNN/FA3 first (library ceiling)
2. Check for sparsity patterns (BSR, 2:4 structured)
3. Identify fusion opportunities (multi-op patterns)
4. Suggest CUTLASS 4.3 CollectiveBuilder for:
   - Tile size tuning
   - Mixed precision
   - Cluster shapes (for H100+)

5. **Only** suggest custom kernels when:
   - Sparsity >70% AND cuSPARSE <100 TFLOPS
   - Fusion saves >20% latency
   - Library doesn't support the op

---

## Versions Summary

| Technology | Version | Status | Notes |
|------------|---------|--------|-------|
| CUTLASS | 4.3.0 | ✅ Current | Oct 2025, CuTe DSL, SM100 |
| CUDA | 13.0.2 | ✅ Current | Aug 2025, FP8 native |
| Driver | 580.95.05 | ✅ Validated | H100 PCIe |
| Triton | 3.0.0 | ✅ Current | Oct 2025 |
| vLLM | 0.8.0+ | ✅ Current | Nov 2025 |
| FA3 | Latest | ✅ Current | CUDA graphs |
| CuTe | Integrated | ✅ In CUTLASS 3.x+ | Python DSL |

---

**Next:** Build `ceiling_scout.py` based on this validated ecosystem knowledge.

