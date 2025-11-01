# periodicdent42

GPU kernel optimization research.

---

## BlackwellSparseK - Sparse GEMM Kernel

**52.1 TFLOPS on L4 (1.74× faster than CUTLASS 4.3.0)**

Sparse Block Sparse Row (BSR) GEMM optimized for NVIDIA Ada architecture.

**Performance (L4, SM89):**
- 52.1 TFLOPS (vs CUTLASS 4.3.0: ~30 TFLOPS)
- 1.74× speedup over CUTLASS baseline
- 63× faster than cuSPARSE
- Full Nsight Compute validation

**[→ Full documentation](BlackwellSparseK/)**

---

## Quick Start

```bash
git clone https://github.com/GOATnote-Inc/periodicdent42.git
cd periodicdent42/BlackwellSparseK
pip install -e .
```

**Usage:**
```python
import blackwellsparsek as bsk

# Drop-in replacement for torch.sparse.mm (1.74× faster than CUTLASS)
C = bsk.sparse_mm(A_sparse, B_dense)
```

---

## Repository Structure

```
BlackwellSparseK/       # Validated sparse GEMM kernel (52.1 TFLOPS on L4)
├── src/                # Core kernel implementation
├── benchmarks/         # Performance validation
├── python/             # PyTorch bindings
└── README.md           # Full documentation

.archive/               # Experimental code (not production)
csrc/kernels/           # Research kernels (incomplete)
docs/                   # Technical notes
```

---

## Author

**Brandon Dent, MD**  
Former Emergency Medicine Professor → GPU Kernel Engineer

**Contact:** b@thegoatnote.com  
**License:** BSD-3-Clause

---

**[BlackwellSparseK Documentation](BlackwellSparseK/) | [NCU Analysis](BlackwellSparseK/NCU_ANALYSIS_PRODUCTION.md) | [Release Notes](BlackwellSparseK/RELEASE_v1.0.0.md)**
