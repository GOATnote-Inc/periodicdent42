# Optimized Dense GEMM for H100

**550.8 TFLOPS** (88% of cuBLAS) using CUTLASS 4.3.0 CollectiveBuilder

Non-square tile optimization for NVIDIA H100 (Hopper).

## Performance

| Implementation | TFLOPS | vs cuBLAS |
|----------------|--------|-----------|
| cuBLAS | 622.8 | 100% |
| **This work** | **550.8** | **88%** |
| CUTLASS 4.3 Ex49 | 406.8 | 65% |

**Problem:** 8192×8192×19712, FP16→FP32  
**Config:** TileShape 128×256×64, ClusterShape 2×1×1  
**Verification:** CUDA Events, 5 runs, ±0.3% variance

## Quick Start

```bash
cd BlackwellSparseK/examples/gemm_optimized

nvcc -O3 -arch=sm_90a --expt-relaxed-constexpr \
     -I/opt/cutlass/include \
     gemm_optimized.cu -o gemm

./gemm  # Expect: ~550 TFLOPS
```

**Requirements:** CUDA 12.8+, CUTLASS 4.3.0, H100 GPU

## Repository Structure

```
BlackwellSparseK/
├── examples/gemm_optimized/    # Main implementation
│   ├── gemm_optimized.cu       # Kernel source
│   └── README.md               # Build & usage
├── src/                        # Alternative entry point
├── CUTLASS_CONTRIBUTION.md     # Contribution roadmap
└── README.md                   # This file
```

## Documentation

- **[Examples](BlackwellSparseK/examples/gemm_optimized/)** - Build instructions
- **[Full report](BlackwellSparseK/VERIFIED_551_TFLOPS.md)** - Verification details
- **[CUTLASS contribution](BlackwellSparseK/CUTLASS_CONTRIBUTION.md)** - Submission plan

## Status

**Code:** Complete and verified  
**Performance:** Measured with CUDA Events (industry standard)  
**NCU profiling:** Pending (requires bare metal access)

## License

BSD 3-Clause

Copyright © 2025 Brandon Dent

## Contact

Brandon Dent, MD • b@thegoatnote.com

---

**Version:** 1.0.0  
**Date:** November 2, 2025
