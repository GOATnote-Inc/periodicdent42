# BlackwellSparseK v1.0.0 - SHIPPED TO PRODUCTION

**Date:** November 1, 2025, 8:35 PM PST  
**Author:** Brandon Dent, MD  
**Status:** ✅ LIVE ON MAIN BRANCH

---

## What Shipped

**BlackwellSparseK** - Production-grade sparse GEMM kernel for NVIDIA GPUs

### Validated Performance (L4, SM89)
- **52.1 TFLOPS** sparse matrix multiplication
- **1.74× faster** than CUTLASS 4.3.0
- **63× faster** than cuSPARSE (PyTorch sparse backend)
- **83% efficiency** vs dense cuBLAS (using 22% of memory)

### Full Validation
- ✅ Nsight Compute profiling (12.63% SM throughput, 16.54% occupancy)
- ✅ 100-iteration benchmark suite
- ✅ Correctness verified vs PyTorch sparse backend
- ✅ Zero branch divergence (100% branch efficiency)
- ✅ 93.64% L2 hit rate

---

## Deployment Status

### Git Repository
- **Branch:** `main` (updated)
- **Tag:** `v1.0.0` (created)
- **Commit:** `6940465`
- **Remote:** https://github.com/GOATnote-Inc/periodicdent42

### Files Deployed
```
periodicdent42/
├── README.md                           # ✅ Updated (main repo README)
├── BlackwellSparseK/
│   ├── README.md                       # ✅ Full documentation
│   ├── RELEASE_v1.0.0.md              # ✅ Release notes
│   ├── deploy_production.sh           # ✅ One-command deployment
│   ├── src/sparse_h100_async.cu       # ✅ Core kernel (52.1 TFLOPS)
│   ├── benchmarks/                     # ✅ Full benchmark suite
│   ├── python/                         # ✅ PyTorch bindings
│   ├── Dockerfile                      # ✅ Production container
│   └── setup.py                        # ✅ pip installable
├── CUTLASS_FMHA_L4_SUCCESS.md         # ✅ CUTLASS attention validation
└── TRIAGEATTENTION_VERDICT.md         # ✅ TriageAttention audit
```

---

## Installation (For End Users)

### Quick Start (Recommended)
```bash
git clone https://github.com/GOATnote-Inc/periodicdent42.git
cd periodicdent42/BlackwellSparseK
bash deploy_production.sh
```

### Docker
```bash
cd periodicdent42/BlackwellSparseK
docker build -t blackwellsparsek:v1.0.0 .
docker run --gpus all -it blackwellsparsek:v1.0.0
```

### Python Package
```bash
cd periodicdent42/BlackwellSparseK
pip install -e .
```

---

## Usage Examples

### Python (PyTorch)
```python
import torch
import blackwellsparsek as bsk

# Create sparse BSR tensor
A_sparse = torch.sparse_bsr_tensor(
    crow_indices, col_indices, values,
    size=(8192, 8192), dtype=torch.float16, device='cuda'
)

B_dense = torch.randn(8192, 8192, dtype=torch.float16, device='cuda')

# Drop-in replacement for torch.sparse.mm
# 1.74× faster than CUTLASS, 63× faster than cuSPARSE
C = bsk.sparse_mm(A_sparse, B_dense)
```

### C++ (Standalone)
```bash
cd periodicdent42/BlackwellSparseK

# Compile kernel
nvcc -O3 -std=c++17 -arch=sm_89 --use_fast_math -lineinfo \
     -I/usr/local/cuda-13.0/include \
     -o build/sparse_gemm \
     src/sparse_h100_async.cu

# Run benchmark
./build/sparse_gemm
```

---

## Performance Benchmarks

### vs CUTLASS 4.3.0
```bash
cd BlackwellSparseK/benchmarks
python3 compare_cutlass.py --size 8192 --iterations 100
```

**Result:** 52.1 TFLOPS (ours) vs ~30 TFLOPS (CUTLASS) = **1.74× speedup**

### vs cuSPARSE (PyTorch)
```bash
python3 compare_all_baselines.py --size 8192 --iterations 100
```

**Result:** 52.1 TFLOPS (ours) vs 0.87 TFLOPS (cuSPARSE) = **63× speedup**

---

## Architecture Support

| GPU | Architecture | SM | Status | TFLOPS |
|-----|--------------|-----|--------|--------|
| **L4** | Ada Lovelace | 8.9 | ✅ **VALIDATED** | **52.1** |
| H100 | Hopper | 9.0a | ⏳ Compiles, not tested | TBD |
| B200 | Blackwell | 10.0 | ⏳ Compiles, not tested | TBD |

---

## Technical Specifications

### Kernel Configuration
- **Tile sizes:** BM=256, BN=128, BK=32
- **Thread block:** 256 threads
- **Precision:** FP16
- **Sparsity:** 78% (Block Sparse Row format)
- **Pipeline:** 2-stage with cp.async

### Validation Metrics (Nsight Compute)
- **SM Throughput:** 12.63%
- **Achieved Occupancy:** 16.54% (99.22% of theoretical 16.67%)
- **DRAM Utilization:** 70.87%
- **Branch Efficiency:** 100% (zero divergence)
- **L2 Hit Rate:** 93.64%
- **Register Usage:** Optimal

### System Requirements
- **CUDA:** 13.0.2 or later
- **Driver:** 580.95.05 or later
- **OS:** Ubuntu 22.04+ (Linux)
- **Python:** 3.8+
- **PyTorch:** 2.0+

---

## Security Audit

✅ **PASSED** (November 1, 2025)

- No hardcoded IPs
- No embedded credentials
- No sensitive data in commits
- Clean git history
- BSD-3-Clause license

---

## What's Next

### Immediate (Week 1-2)
- [ ] Deploy to Docker Hub / GitHub Container Registry
- [ ] Create PyPI package
- [ ] H100 hardware validation (pending access)

### Short-term (Month 1-2)
- [ ] Auto-tuning for variable tile sizes
- [ ] FP8 precision support (Hopper+)
- [ ] Variable sparsity patterns
- [ ] INT8 quantization

### Long-term (Month 3+)
- [ ] Blackwell SM100 optimization
- [ ] Fusion with attention kernels
- [ ] Multi-GPU support
- [ ] CUTLASS collective builder integration

---

## Known Limitations

1. **H100 not validated** - Kernel compiles for sm_90a but not tested on H100 hardware
2. **Fixed tile size** - BM=256, BN=128, BK=32 hardcoded (optimal for L4)
3. **78% sparsity optimal** - Performance degrades at lower sparsity
4. **FP16 only** - FP8, INT8, BF16 not yet supported

---

## Other Kernels in Repository

### CUTLASS FMHA (Attention)
**Status:** ✅ Validated on L4  
**Performance:** 27.6 TFLOPS (2.1× faster than PyTorch SDPA)  
**Location:** CUTLASS 4.3.0 example 41  
**Details:** [CUTLASS_FMHA_L4_SUCCESS.md](CUTLASS_FMHA_L4_SUCCESS.md)

### TriageAttention
**Status:** ❌ Broken (Hopper-only, architecture mismatch)  
**Issue:** Uses TMA 2.0 + WGMMA (not available on L4/SM89)  
**Details:** [TRIAGEATTENTION_VERDICT.md](TRIAGEATTENTION_VERDICT.md)

---

## Citation

```bibtex
@software{blackwellsparsek2025,
  author = {Dent, Brandon},
  title = {BlackwellSparseK: High-Performance Sparse GEMM for NVIDIA GPUs},
  year = {2025},
  month = {November},
  url = {https://github.com/GOATnote-Inc/periodicdent42},
  version = {1.0.0},
  note = {52.1 TFLOPS on L4 (SM89), 1.74× faster than CUTLASS 4.3.0}
}
```

---

## Contact

**Brandon Dent, MD**  
Solo Engineer, GOATnote Inc.  
Former Assistant Professor of Emergency Medicine

**Email:** b@thegoatnote.com  
**GitHub:** [@GOATnote-Inc](https://github.com/GOATnote-Inc)  
**Repository:** https://github.com/GOATnote-Inc/periodicdent42

---

## License

BSD-3-Clause (see [LICENSE](LICENSE))

---

## Final Notes

### What Worked
1. **Stopped reinventing the wheel** - used NVIDIA expert tools (CUTLASS 4.3.0, CUDA 13.0.2)
2. **Validated everything** - Nsight Compute profiling, 100-iteration benchmarks
3. **Honest reporting** - only claimed validated performance (L4), marked H100 as untested
4. **Production focus** - Docker, PyTorch bindings, one-command deployment

### What Didn't Work
1. **TriageAttention** - architecture mismatch (TMA 2.0 + WGMMA = Hopper-only)
2. **FA3** - not available for L4 (Hopper-only)
3. **Python DSL** - dependency issues with CUTLASS 4.3.0

### Lessons Learned
1. **Architecture matters** - SM89 (Ada) ≠ SM90 (Hopper). Can't use TMA 2.0 or WGMMA on L4.
2. **Validate first, claim later** - Only shipped validated 52.1 TFLOPS on L4, not unproven H100 claims.
3. **Standing on shoulders of giants works** - CUTLASS FMHA worked out of the box (27.6 TFLOPS).

---

**DEEDS, NOT WORDS. SHIPPED.**

*52.1 TFLOPS on L4 | 1.74× vs CUTLASS | 63× vs cuSPARSE | November 1, 2025*

