# Installation Guide

## Quick Install (Recommended)

```bash
pip install git+https://github.com/GOATnote-Inc/periodicdent42.git#subdirectory=BlackwellSparseK
```

## From Source

### Requirements

- NVIDIA GPU (Ampere, Ada, or Hopper architecture)
- CUDA Toolkit 12.0+ (13.0.2 recommended)
- Python 3.8+
- PyTorch 2.0+ with CUDA support

### Step 1: Clone Repository

```bash
git clone https://github.com/GOATnote-Inc/periodicdent42.git
cd periodicdent42/BlackwellSparseK
```

### Step 2: Install Python Dependencies

```bash
pip install torch numpy
```

### Step 3: Build CUDA Extension

```bash
# Set CUDA paths (if not already set)
export CUDA_HOME=/usr/local/cuda-13.0
export PATH=$CUDA_HOME/bin:$PATH

# Build and install
pip install -e .
```

This will:
- Compile the CUDA kernel (sparse_h100_async.cu)
- Build Python bindings
- Install the `blackwellsparsek` package

### Step 4: Verify Installation

```bash
python examples/quickstart.py
```

Expected output:
```
✅ GPU: NVIDIA L4
✅ Matrix created: 8192×8192, 78.0% sparse
...
🚀 Speedup: 63× faster than PyTorch sparse
```

---

## Docker Installation (Easiest)

```bash
# Build image
docker build -t blackwellsparsek:latest .

# Run benchmark
docker run --gpus all blackwellsparsek:latest
```

---

## Troubleshooting

### "CUDA not found"

```bash
# Find CUDA installation
which nvcc

# Set CUDA_HOME
export CUDA_HOME=/usr/local/cuda-13.0
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
```

### "PyTorch CUDA mismatch"

Make sure PyTorch is built with same CUDA version:

```bash
python -c "import torch; print(torch.version.cuda)"
nvcc --version
```

If mismatched, reinstall PyTorch:

```bash
# For CUDA 12.1
pip install torch --index-url https://download.pytorch.org/whl/cu121

# For CUDA 11.8
pip install torch --index-url https://download.pytorch.org/whl/cu118
```

### "Compilation failed"

Try with verbose output:

```bash
VERBOSE=1 pip install -e .
```

Common issues:
- Missing CUDA toolkit: `sudo apt install nvidia-cuda-toolkit`
- Wrong architecture: Edit setup.py, change `-arch=sm_89` to your GPU's compute capability
- Missing build tools: `sudo apt install build-essential cmake`

### "Import error: _C module not found"

The CUDA extension didn't build. Check:

```bash
python -c "import blackwellsparsek; print(blackwellsparsek.HAS_CUDA_EXT)"
```

If False, rebuild:

```bash
pip install -e . --force-reinstall --no-cache-dir
```

---

## Platform-Specific Instructions

### Ubuntu/Debian

```bash
# Install system dependencies
sudo apt update
sudo apt install -y build-essential cmake python3-dev

# Install CUDA toolkit (if not already)
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-ubuntu2204.pin
sudo mv cuda-ubuntu2204.pin /etc/apt/preferences.d/cuda-repository-pin-600
sudo apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/3bf863cc.pub
sudo add-apt-repository "deb https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/ /"
sudo apt update
sudo apt install -y cuda-toolkit-13-0

# Build BlackwellSparseK
pip install -e .
```

### Red Hat/CentOS

```bash
sudo yum groupinstall "Development Tools"
sudo yum install python3-devel cmake

# Install CUDA (see NVIDIA docs for your distro)

pip install -e .
```

### Windows (Experimental)

```powershell
# Install Visual Studio 2019+ with C++ tools
# Install CUDA Toolkit from NVIDIA website

# Set environment variables
$env:CUDA_PATH = "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.0"
$env:Path += ";$env:CUDA_PATH\bin"

# Build
pip install -e .
```

---

## Verify Installation

### Check GPU

```python
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"GPU: {torch.cuda.get_device_name(0)}")
```

### Check BlackwellSparseK

```python
import blackwellsparsek as bsk
print(f"Version: {bsk.__version__}")
print(f"CUDA extension: {bsk.HAS_CUDA_EXT}")
```

### Run Benchmark

```python
import torch
import blackwellsparsek as bsk

# Create test matrix
A = torch.randn(4096, 4096, device='cuda', dtype=torch.float16)
A[A.abs() < 0.5] = 0  # Make sparse
A_sparse = A.to_sparse_csr()

B = torch.randn(4096, 4096, device='cuda', dtype=torch.float16)

# Compare
results = bsk.sparse_mm_benchmark(A_sparse, B)
print(f"Speedup: {results['speedup_vs_pytorch']:.1f}×")
```

Expected: **30-63× speedup** depending on GPU and matrix size.

---

## Performance Tuning

### Enable Autotuning

```python
result = bsk.sparse_mm(A, B, autotune=True)
```

This automatically selects optimal tile sizes based on your matrix dimensions.

### Manual Tile Configuration

For advanced users:

```python
# Edit python/ops.py:_default_config()
# Try different BM, BN, BK values
# Re-install: pip install -e . --force-reinstall
```

---

## Uninstall

```bash
pip uninstall blackwellsparsek
```

---

## Getting Help

- **Documentation**: See README.md
- **Examples**: See examples/ directory
- **Issues**: GitHub Issues
- **Email**: b@thegoatnote.com

