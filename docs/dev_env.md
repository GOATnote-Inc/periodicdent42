# Development Environment Setup

**Target**: NVIDIA L4 (SM_89, Ada Lovelace) with PyTorch 2.2+  
**Purpose**: Fast, reproducible CUDA kernel builds with environment locking

---

## Quick Setup (L4 GPU Instance)

```bash
# Clone repository
git clone https://github.com/GOATnote-Inc/periodicdent42.git
cd periodicdent42

# Run setup script
bash scripts/setup_dev_env.sh

# Verify
python3 -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"
```

---

## Manual Setup

### 1. System Dependencies

```bash
# Ubuntu 22.04 LTS
sudo apt-get update
sudo apt-get install -y \
    build-essential \
    ccache \
    git \
    python3-pip \
    python3-venv

# Optional: CUDA toolkit (if not pre-installed)
# wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb
# sudo dpkg -i cuda-keyring_1.1-1_all.deb
# sudo apt-get update
# sudo apt-get install -y cuda-toolkit-12-1
```

### 2. Python Environment

```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt

# Install build accelerators
pip install ninja ccache
```

### 3. Environment Variables

Add to `~/.bashrc` or `~/.zshrc`:

```bash
# CUDA build optimization (L4 = SM_89)
export TORCH_CUDA_ARCH_LIST="8.9"
export MAX_JOBS=$(nproc)
export CUDAFLAGS="--use_fast_math -O3"

# Build cache (speeds up recompilation)
export CCACHE_DIR=$HOME/.ccache
export CCACHE_MAXSIZE=5G

# PyTorch build cache
export TORCH_CUDA_BUILD_CACHE=$HOME/.torch_cuda_cache

# CUDA compiler threads
export NVCC_THREADS=$(nproc)

# Path (ensure CUDA in PATH)
export PATH=/usr/local/cuda/bin:$HOME/.local/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
```

Apply changes:
```bash
source ~/.bashrc
```

### 4. Verify Setup

```bash
# Check CUDA
nvcc --version
nvidia-smi

# Check PyTorch
python3 -c "import torch; print(f'PyTorch: {torch.__version__}')"
python3 -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
python3 -c "import torch; print(f'GPU: {torch.cuda.get_device_name(0)}')"

# Check build tools
which ninja
which ccache
ccache -s  # Show cache statistics

# Check Nsight Compute (profiling)
which ncu
ncu --version
```

Expected output:
```
CUDA: 12.1
PyTorch: 2.2.1+cu121
CUDA available: True
GPU: NVIDIA L4
Ninja: 1.11.1
ccache: 4.8
Nsight Compute: 2024.1.0
```

---

## Environment Locking (Reproducibility)

For benchmarking and correctness testing, we enforce strict environment settings:

```python
import torch
import os

# Lock dtype to FP16
torch.set_default_dtype(torch.float16)

# Disable TF32 (use true FP32 precision)
torch.set_float32_matmul_precision("highest")
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

# Enable deterministic algorithms
torch.use_deterministic_algorithms(True, warn_only=True)
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'

# Set random seeds
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)
import numpy as np
np.random.seed(42)
os.environ.setdefault("PYTHONHASHSEED", "0")
```

This is automatically applied by `cudadent42.bench.common.env_lock.lock_environment()`.

---

## Build Performance Optimization

### Ninja (Parallel Build)

Ninja parallelizes build steps and tracks dependencies efficiently.

**Before Ninja** (setuptools):
```
Build time: 3-5 minutes
Parallelism: Limited
```

**After Ninja**:
```
Build time: 30-60 seconds
Parallelism: Full (MAX_JOBS cores)
```

**Verify Ninja is used**:
```python
import torch.utils.cpp_extension
print(f"Ninja available: {torch.utils.cpp_extension.is_ninja_available()}")
```

### ccache (Compilation Cache)

ccache caches compiler outputs, avoiding recompilation of unchanged files.

**First build**: 30-60 seconds  
**Subsequent builds** (cached): 5-10 seconds

**Check cache stats**:
```bash
ccache -s
```

Example output:
```
cache hit rate: 92.3%
files in cache: 847
cache size: 1.2 GB / 5.0 GB
```

### Persistent Build Cache

PyTorch extensions use a build directory that can be reused:

```python
module = torch.utils.cpp_extension.load(
    name="my_kernel",
    sources=["kernel.cu"],
    build_directory=".torch_build",  # Persistent cache
    verbose=False
)
```

**Benefits**:
- Reuse compiled objects across Python sessions
- Avoid full recompilation on code changes
- Share cache across experiments

---

## GPU Clock Locking (Stability)

For reproducible benchmarks, lock GPU clocks to maximum:

```bash
# Check current clocks
nvidia-smi --query-gpu=clocks.sm,clocks.mem --format=csv

# Lock to maximum (requires sudo)
sudo nvidia-smi -lgc 2100  # Lock GPU clock to 2100 MHz
sudo nvidia-smi -lmc 6251  # Lock memory clock to 6251 MHz

# Verify
nvidia-smi --query-gpu=clocks.sm,clocks.mem --format=csv

# Reset to auto (when done)
sudo nvidia-smi -rgc
sudo nvidia-smi -rmc
```

**Note**: Clock locking requires root access. On GCP instances, you may need to request elevated permissions.

---

## Troubleshooting

### Issue: `ModuleNotFoundError: No module named 'ninja'`

**Fix**:
```bash
pip install ninja
export PATH=$HOME/.local/bin:$PATH
```

### Issue: `CUDA out of memory`

**Fix 1**: Reduce batch size
```python
batch = 16  # instead of 32
```

**Fix 2**: Clear cache
```python
import torch
torch.cuda.empty_cache()
```

**Fix 3**: Check available memory
```bash
nvidia-smi
```

### Issue: Slow builds even with Ninja

**Check 1**: Verify Ninja is detected
```python
import torch.utils.cpp_extension
print(torch.utils.cpp_extension.is_ninja_available())  # Should be True
```

**Check 2**: Verify architecture list is limited
```bash
echo $TORCH_CUDA_ARCH_LIST  # Should be "8.9" only
```

**Check 3**: Check ccache hit rate
```bash
ccache -s  # Should show >50% hit rate after first build
```

### Issue: `RuntimeError: CUDA error: invalid device function`

**Cause**: Kernel compiled for wrong architecture

**Fix**: Ensure `TORCH_CUDA_ARCH_LIST="8.9"` for L4
```bash
export TORCH_CUDA_ARCH_LIST="8.9"
```

### Issue: Non-deterministic results

**Fix**: Verify environment lock
```python
import torch
print(f"TF32 matmul: {torch.backends.cuda.matmul.allow_tf32}")  # Should be False
print(f"TF32 cuDNN: {torch.backends.cudnn.allow_tf32}")  # Should be False
print(f"Deterministic: {torch.are_deterministic_algorithms_enabled()}")  # Should be True
```

---

## Performance Checklist

Before benchmarking, verify:

- ✅ `TORCH_CUDA_ARCH_LIST="8.9"` (L4 only)
- ✅ `MAX_JOBS=$(nproc)` (parallel builds)
- ✅ Ninja installed and detected
- ✅ ccache configured and warming up
- ✅ GPU clocks locked (if possible)
- ✅ No other processes using GPU (`nvidia-smi`)
- ✅ Environment locked (FP16, TF32 off, deterministic)

---

## Recommended Workflow

### 1. One-Time Setup
```bash
bash scripts/setup_dev_env.sh
source ~/.bashrc
```

### 2. Start Session
```bash
source venv/bin/activate
cd /path/to/periodicdent42

# Verify environment
python3 -c "from cudadent42.bench.common.env_lock import lock_environment; lock_environment()"
```

### 3. Build Kernel
```bash
# Use helper (auto Ninja + ccache)
python3 bench/_build.py --kernel fa_s512 --config "BLOCK_M=64,BLOCK_N=64"
```

### 4. Benchmark
```bash
# Baseline
python3 bench/baseline_comprehensive.py --only s512

# Correctness
python3 bench/correctness_fuzz.py

# Profile
S=512 B=32 H=8 D=64 bash scripts/profile_sdpa.sh
```

### 5. Clean Up
```bash
# Clear build cache (if needed)
rm -rf .torch_build

# Clear CUDA cache
python3 -c "import torch; torch.cuda.empty_cache()"

# Stop GPU (GCP)
gcloud compute instances stop $(hostname) --zone=$(curl -H "Metadata-Flavor: Google" http://metadata.google.internal/computeMetadata/v1/instance/zone | cut -d'/' -f4)
```

---

## Resources

- **PyTorch CUDA Extension Docs**: https://pytorch.org/tutorials/advanced/cpp_extension.html
- **Ninja Build**: https://ninja-build.org/
- **ccache**: https://ccache.dev/
- **CUDA Toolkit**: https://developer.nvidia.com/cuda-toolkit
- **Nsight Compute**: https://developer.nvidia.com/nsight-compute

---

**Last Updated**: 2025-10-14  
**Maintainer**: GOATnote Autonomous Research Lab Initiative  
**Contact**: b@thegoatnote.com

