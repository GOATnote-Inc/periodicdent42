# periodicdent42

GPU kernel optimization research by Brandon Dent, MD.

---

## Optimized GEMM (H100)

**550.8 TFLOPS** - 88% of cuBLAS using CUTLASS 4.3.0 CollectiveBuilder

Dense FP16 GEMM optimized for NVIDIA H100 (Hopper).

**[→ Documentation](BlackwellSparseK/)**

---

## Results

| Kernel | TFLOPS | Hardware |
|--------|--------|----------|
| This work | 550.8 | H100 80GB |
| cuBLAS | 622.8 | H100 80GB |
| CUTLASS 4.3 | 406.8 | H100 80GB |

**Method:** TileShape 128×256×64, ClusterShape 2×1×1

**Verification:** CUDA Events, 5 runs, ±0.3% variance

---

## Contact

Brandon Dent, MD  
b@thegoatnote.com

---

**License:** BSD 3-Clause
