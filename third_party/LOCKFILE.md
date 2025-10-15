# Third-Party Tool Lockfile

**Purpose**: Record exact versions/commits of integrated tools for reproducibility.

**Last Updated**: 2025-10-15 (Phase 1 - Staff-Level CUDA Optimization)

---

## EvoEngineer

**Type**: Internal Custom Tool  
**Path**: `third_party/evoengineer/`  
**Description**: Evolutionary kernel parameter optimizer for CUDA kernels  
**Version**: Internal v1.0 (2025-10-15)  
**Commit/Hash**: N/A (internal development)  

**Components**:
- `optimizer.py` - Core evolutionary optimization loop
- `evaluator.py` - Correctness testing & performance benchmarking
- `mutator.py` - Parameter mutation strategies

**Integration Method**: Direct Python package (not submodule)  
**Dependencies**: `torch`, `numpy`, `json`, `subprocess`

---

## robust-kbench

**Type**: Internal Custom Tool  
**Path**: `third_party/robust_kbench/`  
**Description**: Statistical micro-benchmarking framework for CUDA kernels  
**Version**: Internal v1.0 (2025-10-15)  
**Commit/Hash**: N/A (internal development)

**Components**:
- `config.py` - YAML configuration loading & validation
- `runner.py` - Benchmark execution engine
- `reporter.py` - Results aggregation & comparison reports

**Integration Method**: Direct Python package (not submodule)  
**Dependencies**: `torch`, `pyyaml`, `json`

---

## CUTLASS

**Type**: External Open-Source Library (Git Submodule)  
**Repository**: https://github.com/NVIDIA/cutlass  
**Path**: `third_party/cutlass/`  
**Description**: CUDA Templates for Linear Algebra Subroutines (Tensor Core GEMM operations)  
**Version**: v3.5.1  
**Commit**: f7b19de3 (December 2024)  
**Added**: 2025-10-15 (Phase 3 Tensor Core prototype)

**Purpose**:
- Tensor Core GEMM operations for FlashAttention TC kernel
- High-performance matrix multiply primitives (QK^T, P@V)
- Templates for sm_89 (NVIDIA L4) architecture

**Integration Method**: Headers-only use (no separate CUTLASS build)  
**Dependencies**: CUDA 12.2+, C++17  
**Include Paths**: `third_party/cutlass/include`, `third_party/cutlass/tools/util/include`

**Status**: Submodule added, kernel prototype created, compilation pending

---

## Notes

**Why Internal Tools?**  
EvoEngineer and robust-kbench are custom-built frameworks designed specifically for this project's CUDA kernel optimization workflow. They are not publicly available open-source projects.

**Reproducibility Strategy**:
- Tools are version-controlled directly in the repository under `third_party/`
- All dependencies are pinned in `requirements.lock` and `requirements-full.lock`
- Bootstrap script (`scripts/bootstrap_tools.sh`) handles environment setup
- Git commit history provides full audit trail

**Alternative Public Tools** (if needed in future):
- For benchmarking: [KernelBench](https://github.com/andravin/KernelBench), [nsys](https://developer.nvidia.com/nsight-systems)
- For optimization: Manual Nsight Compute + iterative refinement, [CUTLASS profiler](https://github.com/NVIDIA/cutlass)
