# Optimized Dense GEMM for NVIDIA H100

High-performance FP16 dense matrix multiplication using CUTLASS 4.3.0 CollectiveBuilder.

## Performance (H100 80GB)

| Problem Size | Time (ms) | TFLOPS | vs cuBLAS |
|--------------|-----------|--------|-----------|
| 8192³ | 2.10 | 523.6 | 84% |
| 8192×8192×19712 | 4.80 | **550.8** | **88%** |

**Configuration:** TileShape 128×256×64, ClusterShape 2×1×1, FP16→FP32

**Verification:** CUDA Events, 5 independent runs, ±0.3% variance

## Comparison

| Implementation | TFLOPS | Relative |
|----------------|--------|----------|
| cuBLAS | 622.8 | 1.00× |
| **This kernel** | **550.8** | **0.88×** |
| CUTLASS 4.3 (Ex49) | 406.8 | 0.65× |
| CUTLASS Ex62 (sparse 2:4) | 269.1 | 0.43× |

## Quick Start

```bash
# Requirements: CUDA 12.8+, H100 GPU, CUTLASS 4.3.0

# Compile
nvcc -O3 -std=c++17 -arch=sm_90a --expt-relaxed-constexpr \
     --maxrregcount=255 \
     -I/opt/cutlass/include \
     optimized_gemm.cu -o gemm -lcudart

# Run
./gemm
# Expected: ~550 TFLOPS
```

## Key Optimizations

1. **TileShape 128×256×64** - Larger N dimension vs standard 128×128×128
2. **ClusterShape 2×1×1** - Better SM alignment than default 1×2×1
3. **Non-square problems** - K=19712 optimal for this tile config

## Technical Details

- **Input:** FP16 matrices (half precision)
- **Output:** FP32 accumulation
- **API:** CUTLASS CollectiveBuilder (modern CUTLASS 4.x)
- **Features:** TMA + WGMMA (Hopper native instructions)
- **Timing:** CUDA Events (industry standard)

## Limitations

- Tested on H100 only (sm_90a)
- Specific problem sizes validated
- NCU profiling unavailable (cloud restrictions)

## Files

```
production_gemm_550tflops.cu  # Main kernel
VERIFIED_551_TFLOPS.md        # Full validation report
```

## Citation

```bibtex
@software{optimized_gemm_h100,
  title={Optimized Dense GEMM for NVIDIA H100},
  author={Dent, Brandon},
  year={2025},
  note={88\% of cuBLAS performance using CUTLASS 4.3.0}
}
```

## License

BSD 3-Clause

## Contact

Brandon Dent, MD • b@thegoatnote.com

---

**Status:** Research code, verified performance  
**Date:** November 2, 2025
