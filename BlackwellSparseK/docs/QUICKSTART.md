# BlackwellSparseK Quick Start

Get started with BlackwellSparseK in 5 minutes.

---

## Prerequisites

- **GPU**: NVIDIA H100 (sm_90a) or Blackwell B200 (sm_100)
- **CUDA**: 13.0.2
- **Python**: 3.11+
- **Docker** (recommended) or local CUDA toolkit

---

## Option 1: Docker (Recommended)

### Pull Pre-Built Image

```bash
docker pull ghcr.io/yourusername/blackwell-sparsek:latest
```

### Run Interactive Session

```bash
docker run --gpus all -it --rm \
    ghcr.io/yourusername/blackwell-sparsek:latest
```

### Quick Test

```python
import torch
from blackwell_sparsek import attention_forward

# Create inputs
Q = torch.randn(1, 8, 512, 64, dtype=torch.float16, device='cuda')
K = torch.randn(1, 8, 512, 64, dtype=torch.float16, device='cuda')
V = torch.randn(1, 8, 512, 64, dtype=torch.float16, device='cuda')

# Compute attention
output = attention_forward(Q, K, V)
print(f"✅ Success! Output shape: {output.shape}")
```

---

## Option 2: Build from Source

### Clone Repository

```bash
git clone https://github.com/yourusername/periodicdent42.git
cd periodicdent42/BlackwellSparseK
```

### Set Environment Variables

```bash
export CUDA_HOME=/usr/local/cuda-13.0
export CUTLASS_PATH=/opt/cutlass
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
```

### Install

```bash
pip install -e .[dev]
```

This will:
1. Build CUDA kernels with dual-arch support (sm_90a, sm_100)
2. Install Python package and dependencies
3. Install development tools (pytest, ruff, etc.)

### Verify Installation

```bash
python -c "import blackwell_sparsek; print(blackwell_sparsek.__version__)"
```

---

## Option 3: Docker from Source

### Build Containers

```bash
cd BlackwellSparseK
bash scripts/build_containers.sh
```

This builds:
- `blackwell-sparsek:dev` - Development environment
- `blackwell-sparsek:prod` - Production runtime
- `blackwell-sparsek:bench` - Benchmarking tools
- `blackwell-sparsek:ci` - CI testing

Build time: ~20-30 minutes

### Launch Development Environment

```bash
bash scripts/quick_start.sh 0  # GPU 0
```

Or use docker-compose:

```bash
docker-compose up dev
```

---

## Basic Usage Examples

### 1. Simple Attention

```python
import torch
from blackwell_sparsek import attention_forward

# Create random inputs [B, H, S, D]
B, H, S, D = 1, 8, 512, 64
Q = torch.randn(B, H, S, D, dtype=torch.float16, device='cuda')
K = torch.randn(B, H, S, D, dtype=torch.float16, device='cuda')
V = torch.randn(B, H, S, D, dtype=torch.float16, device='cuda')

# Compute attention
output = attention_forward(Q, K, V)
```

### 2. With Custom Scale

```python
# Use custom softmax scale
scale = 1.0 / (D ** 0.5)  # Standard: 1/sqrt(D)
output = attention_forward(Q, K, V, scale=scale)
```

### 3. Benchmark Performance

```python
from blackwell_sparsek.utils import benchmark_latency

stats = benchmark_latency(attention_forward, Q, K, V, num_iters=100)
print(f"Latency: {stats['median_us']:.2f} μs")
```

### 4. Compare to PyTorch SDPA

```python
from blackwell_sparsek.utils import compare_to_sdpa, print_comparison_summary

comparison = compare_to_sdpa(Q, K, V, attention_forward)
print_comparison_summary(comparison)
```

---

## xFormers Integration

```python
from blackwell_sparsek.backends import SparseKAttention

# Create attention module
attention = SparseKAttention()

# xFormers uses [B, S, H, D] layout
q = torch.randn(1, 512, 8, 64, dtype=torch.float16, device='cuda')
k = torch.randn(1, 512, 8, 64, dtype=torch.float16, device='cuda')
v = torch.randn(1, 512, 8, 64, dtype=torch.float16, device='cuda')

# Forward pass
output = attention(q, k, v)
```

---

## vLLM Server

### Using Docker Compose

```bash
docker-compose --profile production up vllm-server
```

Server runs on http://localhost:8000

### Manual Start

```bash
python -m vllm.entrypoints.openai.api_server \
    --model meta-llama/Llama-3.1-70B \
    --attention-backend SPARSEK_XFORMERS \
    --gpu-memory-utilization 0.9
```

### Test API

```bash
curl http://localhost:8000/v1/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "meta-llama/Llama-3.1-70B",
        "prompt": "Once upon a time",
        "max_tokens": 50
    }'
```

---

## Running Tests

### All Tests

```bash
pytest tests/ -v
```

### GPU Tests Only

```bash
pytest tests/test_kernels.py -v --gpu
```

### Specific Test

```bash
pytest tests/test_kernels.py::test_kernel_correctness -v
```

---

## Running Benchmarks

### Performance Benchmark

```bash
python benchmarks/perf.py --save-results
```

Output:
```
BlackwellSparseK Performance Benchmark
========================================
Device: NVIDIA H100 80GB HBM3
CUDA Version: 13.0
PyTorch Version: 2.5.0

Config: B=1, H=8, S=512, D=64
========================================
PyTorch SDPA:       24.83 μs
BlackwellSparseK:   4.50 μs  ← Target <5 μs
Speedup:            5.52×

✅ TARGET ACHIEVED!
```

### Compare to SDPA

```bash
python benchmarks/compare_sdpa.py
```

### Profile with Nsight Compute

```bash
bash benchmarks/ncu_roofline.sh
```

---

## Common Issues

### "CUDA not available"

```python
import torch
print(torch.cuda.is_available())  # Should be True
print(torch.cuda.get_device_name(0))  # Should show H100
```

**Fix**: Ensure `--gpus all` flag when running Docker

### "Unsupported architecture"

```
RuntimeError: Unsupported architecture: sm_89
```

**Cause**: BlackwellSparseK requires sm_90a (H100) or sm_100 (Blackwell)

**Fix**: Use H100 or newer GPU

### "CUTLASS not found"

```
FileNotFoundError: CUTLASS not found at /opt/cutlass
```

**Fix**: Set `CUTLASS_PATH` environment variable:
```bash
export CUTLASS_PATH=/path/to/cutlass
```

### Build Fails

**Check CUDA version**:
```bash
nvcc --version  # Should be 13.0.2
```

**Check CUTLASS**:
```bash
ls $CUTLASS_PATH/include/cutlass  # Should exist
```

**Clean build**:
```bash
rm -rf build/ *.so
pip install -e . --force-reinstall --no-cache-dir
```

---

## Next Steps

- **Explore Examples**: `examples/` directory
- **Read Architecture**: [ARCHITECTURE.md](ARCHITECTURE.md)
- **API Reference**: [API_REFERENCE.md](API_REFERENCE.md)
- **Migrate from FlashCore**: [MIGRATION_FROM_FLASHCORE.md](MIGRATION_FROM_FLASHCORE.md)

---

## Quick Reference

### Commands

```bash
# Build containers
bash scripts/build_containers.sh

# Start dev environment
bash scripts/quick_start.sh 0

# Run tests
pytest tests/ -v

# Run benchmarks
python benchmarks/perf.py

# Start vLLM server
docker-compose --profile production up
```

### Import Paths

```python
# Main API
from blackwell_sparsek import attention_forward

# Utilities
from blackwell_sparsek.utils import benchmark_latency, validate_correctness

# xFormers backend
from blackwell_sparsek.backends import SparseKAttention

# vLLM backend (auto-registered on import)
from blackwell_sparsek.backends import SparseKBackend
```

### Configuration

```python
from blackwell_sparsek.core import Config, get_default_config

# Get default config
config = get_default_config()

# Customize
config.block_m = 128
config.use_fast_math = True
config.debug_mode = False
```

---

**Need Help?** Open an issue: https://github.com/yourusername/periodicdent42/issues

