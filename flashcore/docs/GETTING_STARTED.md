# Getting Started with FlashCore

This guide walks you through setting up FlashCore and running your first benchmarks.

---

## Prerequisites

### Hardware
- NVIDIA GPU with CUDA support (tested on L4, Ada architecture)
- Minimum 8GB GPU memory

### Software
- Linux or Windows with WSL2
- CUDA Toolkit 12.2+ ([download](https://developer.nvidia.com/cuda-downloads))
- Python 3.10+
- PyTorch 2.1+ with CUDA support

---

## Installation

### Step 1: Verify CUDA

```bash
# Check CUDA compiler
nvcc --version
# Expected: Cuda compilation tools, release 12.2 or higher

# Check GPU
nvidia-smi
# Expected: Your GPU model (e.g., NVIDIA L4)
```

### Step 2: Create Python Environment

```bash
# Create virtualenv
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install PyTorch with CUDA
pip install torch --index-url https://download.pytorch.org/whl/cu121

# Verify PyTorch CUDA
python -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}')"
# Expected: PyTorch: 2.x.x, CUDA: True
```

### Step 3: Install Dependencies

```bash
cd flashcore/
pip install -r requirements.txt
```

---

## Quick Start

### Build Baseline Kernel

```bash
python build.py
```

**Expected output**:
```
================================================================================
FlashCore Baseline Kernel Build
================================================================================
  CUDA Arch:      sm_89
  Debug Mode:     No
  PyTorch:        2.5.1
  CUDA Version:   12.1
  Kernel:         flashcore_baseline.cu
  Bindings:       bindings.cpp
================================================================================

[Compilation progress...]

================================================================================
✅ Build Complete
================================================================================
✅ CUDA available: NVIDIA L4
Running quick sanity check...
✅ Sanity check passed!
```

---

## Run Tests

### Correctness Tests (15 test cases)

```bash
pytest tests/test_correctness.py -v
```

**Expected output**:
```
================================================================================
FlashCore Correctness Tests
================================================================================
  Shapes: 5
  Seeds per shape: 3
  Total tests: 15
  Max error threshold: 0.06
  Mean error threshold: 0.02
================================================================================

tests/test_correctness.py::test_correctness[tiny-0] 
✅ PASS | tiny         (seed=    0) | max=0.0324 | mean=0.0112 | rel=0.0087 | bad=0.0%
PASSED

[... 14 more tests ...]

================ 15 passed in 12.34s ================
```

---

## Run Benchmarks

### Single Shape

```bash
python benchmarks/benchmark_latency.py --shape mission --iters 100
```

**Expected output**:
```
================================================================================
FlashCore Latency Benchmark (iters=100, warmup=20)
================================================================================
Shape        | Config               | FlashCore (p50) | PyTorch (p50)   | Speedup   
----------------------------------------------------------------------------------------------------
mission      | B=1,H=8,S=512,D=64   |   1500.0 µs     |     25.9 µs     |   0.02×   
================================================================================
```

**Interpretation**: Baseline is 58× slower than PyTorch (expected). This is the starting point for optimization.

### All Shapes

```bash
python benchmarks/benchmark_latency.py --all --out results.json
```

**Output**: Console table + JSON file with detailed stats

---

## Development Workflow

### 1. Make Changes to Kernel

Edit `kernels/flashcore_baseline.cu` or create new kernel file.

### 2. Rebuild

```bash
python build.py
```

### 3. Test

```bash
pytest tests/test_correctness.py -v
```

### 4. Benchmark

```bash
python benchmarks/benchmark_latency.py --shape mission
```

### 5. Profile (Optional)

```bash
# NCU profiling (requires sudo on some systems)
ncu --set full --launch-skip 10 --launch-count 1 \
    python benchmarks/benchmark_latency.py --shape mission --iters 1 \
    > profiling/ncu_baseline.txt
```

---

## Troubleshooting

### Build Fails: "nvcc not found"

**Solution**: Add CUDA to PATH
```bash
export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
```

### Build Fails: "architecture mismatch"

**Solution**: Set CUDA_ARCH for your GPU
```bash
# For Tesla T4 (sm_75)
CUDA_ARCH=7.5 python build.py

# For A100 (sm_80)
CUDA_ARCH=8.0 python build.py
```

### Tests Fail: "CUDA out of memory"

**Solution**: Reduce batch size or sequence length
```python
# In test_correctness.py, modify SHAPES:
SHAPES = [
    ("tiny", {"B": 1, "H": 1, "S": 32, "D": 64}),  # Start with tiny
    # Comment out larger shapes temporarily
]
```

### Performance Much Worse Than Expected

**Check GPU utilization**:
```bash
# In another terminal
nvidia-smi -l 1
```

**Enable verbose profiling**:
```bash
VERBOSE=1 python build.py
```

---

## Next Steps

### Phase 1: Implement WMMA (Tensor Cores)

See [Phase 1 Guide](PHASE1_WMMA_GUIDE.md) for detailed instructions.

**Goal**: 10× speedup (1500 µs → 150 µs)

### Phase 2: FlashAttention Fusion

See [Phase 2 Guide](PHASE2_FUSION_GUIDE.md) (coming soon).

**Goal**: 26× speedup (1500 µs → 58 µs) → **PROJECT SUCCESS**

---

## Resources

- **[Architecture](ARCHITECTURE.md)**: Technical design details
- **[Launch Plan](../FLASHCORE_LAUNCH_PLAN.md)**: Project overview and roadmap
- **[Implementation Plan](../FLASHCORE_IMPLEMENTATION_PLAN.md)**: Detailed execution plan
- **[CUDA Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/)**: Official NVIDIA docs

---

## Support

- **Issues**: [GitHub Issues](https://github.com/yourusername/flashcore/issues)
- **Discussions**: [GitHub Discussions](https://github.com/yourusername/flashcore/discussions)

---

**Congratulations! You're ready to optimize GPU kernels. 🚀**

