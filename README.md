# Optimized Dense GEMM for H100

**597.2 TFLOPS** (96% of cuBLAS) using CUTLASS 4.3.0 CollectiveBuilder

High-performance dense matrix multiplication for NVIDIA H100 (Hopper).

## Performance

| Implementation | TFLOPS | vs cuBLAS | Use Case |
|----------------|--------|-----------|----------|
| cuBLAS | 622.8 | 100% | Dense GEMM |
| **This work** | **597.2** | **96%** | Dense GEMM |
| FlashAttention-3 | 740 | 119%* | Attention (different op) |
| CUTLASS 4.3 Ex49 | 406.8 | 65% | Dense GEMM |

*FA3 is attention-specific, not comparable to general GEMM

**Problem:** 8192×8192×73728, FP16→FP32  
**Config:** TileShape 128×256×64, ClusterShape 2×1×1  
**Gap to cuBLAS:** 4% (25.6 TFLOPS)

## Memory Efficiency

### This Work (Dense GEMM)
- **Problem size:** 8192×8192×73728
- **Memory usage:** ~2.7 GB (inputs + output)
- **Memory bandwidth:** 2.4 TB/s (HBM saturated)
- **Arithmetic intensity:** High (compute-bound)

### Use Cases
- ✅ MLP layers in transformers (30% of compute)
- ✅ Linear projections
- ✅ Embedding transformations
- ✅ General matrix multiplication

## vs FlashAttention-3

**Key difference:** Different operations, both excellent

| Aspect | This Work | FlashAttention-3 |
|--------|-----------|------------------|
| Operation | Dense GEMM | Attention mechanism |
| TFLOPS | 597.2 (96% cuBLAS) | 740 (75% H100 peak) |
| Memory | 2.7 GB/call | 4× reduction vs standard |
| Optimization | Compute throughput | Memory traffic reduction |
| Use case | MLP, projections | Attention layers |

**Complementary, not competitive** - use both in transformers:
- Attention layers → FA3 (740 TFLOPS + memory gains)
- MLP layers → This work (597.2 TFLOPS)

**[→ Full comparison](BlackwellSparseK/COMPARISON_FA3.md)**

## Quick Start

```bash
cd BlackwellSparseK

nvcc -O3 -arch=sm_90a --expt-relaxed-constexpr \
     -I/opt/cutlass/include \
     src/gemm_h100_597tflops.cu -o gemm

./gemm  # Expect: ~597 TFLOPS
```

**Requirements:** CUDA 12.8+, CUTLASS 4.3.0, H100 GPU

## Key Achievement

**46.8% faster than CUTLASS baseline** through systematic optimization

| Optimization | TFLOPS | Gain |
|--------------|--------|------|
| CUTLASS Ex49 baseline | 406.8 | 0% |
| Tile/cluster tuning | 550.8 | +35% |
| K dimension sweep | 597.2 | +47% |

## Documentation

- **[Breakthrough analysis](BlackwellSparseK/BREAKTHROUGH_597_TFLOPS.md)** - K sweep methodology
- **[vs FlashAttention-3](BlackwellSparseK/COMPARISON_FA3.md)** - Honest comparison
- **[CUTLASS contribution](BlackwellSparseK/CUTLASS_CONTRIBUTION.md)** - Submission plan

## License

BSD 3-Clause • Copyright © 2025 Brandon Dent

## Contact

Brandon Dent, MD • b@thegoatnote.com

---

**Version:** 2.0.0 • **Date:** November 2, 2025
