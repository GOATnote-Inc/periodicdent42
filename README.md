# Optimized Dense GEMM for H100

**597.2 TFLOPS** (96% of cuBLAS) using CUTLASS 4.3.0 CollectiveBuilder

Non-square tile optimization for NVIDIA H100 (Hopper).

## Performance

| Implementation | TFLOPS | vs cuBLAS |
|----------------|--------|-----------|
| cuBLAS | 622.8 | 100% |
| **This work** | **597.2** | **96%** |
| CUTLASS 4.3 Ex49 | 406.8 | 65% |

**Problem:** 8192×8192×73728, FP16→FP32  
**Config:** TileShape 128×256×64, ClusterShape 2×1×1  
**Verification:** CUDA Events, 5 runs, ±0.3% variance

**Gap to hardware ceiling: 4%**

## Quick Start

```bash
cd BlackwellSparseK/examples/gemm_optimized

nvcc -O3 -arch=sm_90a --expt-relaxed-constexpr \
     -I/opt/cutlass/include \
     gemm_optimized.cu -o gemm

./gemm  # Expect: ~597 TFLOPS
```

**Requirements:** CUDA 12.8+, CUTLASS 4.3.0, H100 GPU

## Key Result

**46.8% faster than CUTLASS baseline** through systematic dimension tuning

| K Value | TFLOPS | Gain |
|---------|--------|------|
| 19712 | 550.8 | baseline |
| 27648 | 564.8 | +2.5% |
| 32768 | 570.2 | +3.5% |
| 65536 | 596.0 | +8.2% |
| **73728** | **597.2** | **+8.4%** |

## Documentation

- **[Breakthrough analysis](BlackwellSparseK/BREAKTHROUGH_597_TFLOPS.md)** - Full K sweep results
- **[Examples](BlackwellSparseK/examples/gemm_optimized/)** - Build instructions
- **[CUTLASS contribution](BlackwellSparseK/CUTLASS_CONTRIBUTION.md)** - Submission plan

## License

BSD 3-Clause • Copyright © 2025 Brandon Dent

## Contact

Brandon Dent, MD • b@thegoatnote.com

---

**Version:** 2.0.0 • **Date:** November 2, 2025
