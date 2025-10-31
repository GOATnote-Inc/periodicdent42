# Contributing to TriageAttention

Thank you for your interest in contributing to TriageAttention! This document outlines the guidelines and processes for contributing to this project.

---

## Code of Conduct

We are committed to providing a welcoming and harassment-free experience for everyone. All contributors are expected to adhere to professional and respectful conduct.

---

## How to Contribute

### Reporting Issues

If you encounter a bug or have a feature request:

1. **Search existing issues** to avoid duplicates
2. **Create a new issue** with:
   - Clear, descriptive title
   - Detailed description of the problem/feature
   - Steps to reproduce (for bugs)
   - Expected vs actual behavior
   - System information (GPU, CUDA version, OS)
   - Relevant code snippets or error messages

### Submitting Pull Requests

1. **Fork the repository**
2. **Create a feature branch** from `main`:
   ```bash
   git checkout -b feature/your-feature-name
   ```
3. **Make your changes** following our coding standards
4. **Add tests** for new functionality
5. **Run the test suite**:
   ```bash
   mkdir build && cd build
   cmake .. -DTRIAGEATTENTION_BUILD_TESTS=ON
   make -j
   ctest --output-on-failure
   ```
6. **Commit with clear messages**:
   ```bash
   git commit -m "feat: Add sparse attention for GQA

   - Implement grouped query attention support
   - Add correctness tests for GQA patterns
   - Benchmark on H100 with standard LLM configs
   
   Performance: 580 TFLOPS on H100 (H=8, D=128)"
   ```
7. **Push to your fork** and **create a pull request**

---

## Coding Standards

### C++ / CUDA

- **Style:** Follow [Google C++ Style Guide](https://google.github.io/styleguide/cppguide.html)
- **Formatting:** Use `clang-format` (config provided in `.clang-format`)
- **Naming:**
  - Classes: `PascalCase`
  - Functions: `snake_case`
  - Constants: `UPPER_SNAKE_CASE`
  - Templates: `PascalCase`
- **Documentation:** Use Doxygen-style comments for public APIs

**Example:**
```cpp
/**
 * @brief Sparse BSR GEMM kernel for Hopper architecture
 * 
 * @param M Number of rows in output matrix
 * @param N Number of columns in output matrix
 * @param K Inner dimension
 * @param block_size Block size for BSR format
 * @param topk Number of non-zero blocks per row
 * 
 * @return Kernel execution status
 */
template<typename T>
cudaError_t sparse_bsr_gemm_h100(
    const T* A, const T* B, T* C,
    int M, int N, int K,
    int block_size, int topk,
    cudaStream_t stream
);
```

### Python

- **Style:** Follow [PEP 8](https://pep8.org/)
- **Formatting:** Use `black` (line length 100)
- **Type Hints:** Required for public APIs
- **Docstrings:** Use Google-style docstrings

**Example:**
```python
def sparse_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    block_size: int = 16,
    topk: int = 16,
    device: str = "cuda"
) -> torch.Tensor:
    """
    Sparse attention with block-sparse pattern.
    
    Args:
        query: Query tensor [B, H, S, D]
        key: Key tensor [B, H, S, D]
        value: Value tensor [B, H, S, D]
        block_size: Block size for sparsity pattern
        topk: Number of top-k blocks to attend to
        device: Target device
        
    Returns:
        Attention output tensor [B, H, S, D]
        
    Raises:
        ValueError: If tensor shapes are incompatible
        RuntimeError: If CUDA kernel execution fails
    """
    pass
```

---

## Testing Requirements

### Unit Tests

All new features must include unit tests:

```cpp
// tests/test_sparse_gemm.cu
TEST(SparseBSRGEMM, CorrectnessMediumMatrix) {
    const int M = 2048, N = 2048, K = 2048;
    const int block_size = 16;
    const int topk = 16;
    
    // Allocate and initialize test data
    // ...
    
    // Run kernel
    auto result = sparse_bsr_gemm_h100(...);
    
    // Validate against reference
    EXPECT_LT(max_error, 1e-3);
    EXPECT_EQ(result, cudaSuccess);
}
```

### Performance Tests

Include benchmarks for performance-critical changes:

```python
# benchmarks/performance/bench_new_feature.py
def bench_sparse_attention_gqa():
    """Benchmark sparse attention with grouped query attention."""
    for config in [
        (B=1, H=32, S=4096, D=128),
        (B=4, H=32, S=8192, D=128),
    ]:
        # Run benchmark
        # Report TFLOPS, latency, memory bandwidth
```

---

## Commit Message Format

We follow [Conventional Commits](https://www.conventionalcommits.org/):

```
<type>(<scope>): <subject>

<body>

<footer>
```

### Types

- `feat`: New feature
- `fix`: Bug fix
- `perf`: Performance improvement
- `refactor`: Code refactoring
- `test`: Adding/updating tests
- `docs`: Documentation changes
- `build`: Build system changes
- `ci`: CI/CD changes
- `style`: Code style changes (formatting)

### Examples

```
feat(attention): Add GQA support for sparse attention

Implements grouped query attention pattern for sparse kernels.
Tested on Llama-3.1 configurations (H=32, GQA groups=8).

Performance: 580 TFLOPS on H100 (vs 610 for MHA)
Correctness: <1e-3 max error vs PyTorch SDPA

Closes #42
```

```
fix(memory): Resolve memory leak in TMA pipeline

Fixed missing cudaFree in error path of TMA pipeline.
Added valgrind test to catch future leaks.

Fixes #58
```

---

## Performance Requirements

### New Kernels

All new kernels must:

1. **Meet baseline performance:**
   - ≥60% of cuBLAS performance for GEMM-like operations
   - ≥80% of FlashAttention-3 for attention kernels
   
2. **Include roofline analysis:**
   - SM utilization
   - Memory bandwidth utilization
   - Compute vs memory bound classification
   
3. **Provide reproducibility:**
   - Deterministic results (<1% variance)
   - SHA-256 checksums for correctness
   - Benchmark scripts included

### Example Benchmark Output

```
========================================
Sparse BSR GEMM Benchmark
========================================
Configuration: M=8192, N=8192, K=8192, block_size=16, topk=16
GPU: H100 SXM5 80GB
CUDA: 13.0.2
Iterations: 100

Results:
  Latency:    1.23 ms (±0.01 ms)
  Throughput: 610 TFLOPS
  DRAM BW:    1.2 TB/s
  SM Util:    72%
  
Comparison:
  vs cuBLAS:   72% (baseline: 843 TFLOPS)
  vs CUTLASS:  +47% (baseline: 414 TFLOPS)
  
Reproducibility:
  SHA-256: a3f2b9c4d1e5f6...
  Variance: 0.3%
========================================
```

---

## Review Process

1. **Automated Checks:**
   - All tests pass (`ctest`)
   - Code formatting (`clang-format`, `black`)
   - Static analysis (`cppcheck`, `mypy`)
   - No new compiler warnings
   
2. **Manual Review:**
   - Code quality and readability
   - Test coverage
   - Performance validation
   - Documentation completeness
   
3. **Maintainer Approval:**
   - At least one maintainer must approve
   - No unresolved review comments
   - CI/CD pipeline passes

---

## Development Workflow

### Setting Up Development Environment

```bash
# Clone repository
git clone https://github.com/GOATnote-Inc/periodicdent42.git
cd periodicdent42

# Install dependencies (Ubuntu 22.04)
sudo apt-get update
sudo apt-get install -y cmake ninja-build clang-format cppcheck

# Install CUDA 13.0.2 (if not already installed)
wget https://developer.download.nvidia.com/compute/cuda/13.0.2/...
sudo sh cuda_13.0.2_linux.run

# Install CUTLASS 4.3.0
git clone https://github.com/NVIDIA/cutlass.git third_party/cutlass
cd third_party/cutlass && git checkout v4.3.0

# Build
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Debug \
         -DTRIAGEATTENTION_BUILD_TESTS=ON \
         -DTRIAGEATTENTION_BUILD_BENCHMARKS=ON
make -j$(nproc)
```

### Running Pre-Commit Checks

```bash
# Format code
clang-format -i csrc/**/*.cu csrc/**/*.cpp include/**/*.h
black python/

# Static analysis
cppcheck --enable=all --inconclusive csrc/
mypy python/

# Run tests
cd build
ctest --output-on-failure

# Run benchmarks
./benchmarks/performance/bench_sparse_gemm
```

---

## Getting Help

- **Technical Questions:** Open a GitHub Discussion
- **Bug Reports:** Open a GitHub Issue
- **Security Concerns:** Email b@thegoatnote.com directly
- **General Inquiries:** Email b@thegoatnote.com

---

## Recognition

Contributors will be acknowledged in:
- `CONTRIBUTORS.md`
- Release notes
- Academic citations (for significant contributions)

---

## License

By contributing to TriageAttention, you agree that your contributions will be licensed under the Apache License 2.0.

---

Thank you for helping make TriageAttention better! 🚀

**Brandon Dent, MD**  
Project Maintainer  
b@thegoatnote.com

