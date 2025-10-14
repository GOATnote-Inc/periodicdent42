# CUDA Development Environment Setup

**Purpose**: Lock environment to hermetic, reproducible settings for CUDA kernel development on NVIDIA L4 (SM_89).

**Target GPU**: NVIDIA L4 (Ada Lovelace, SM_89, 23GB HBM2e, 242 TFLOPS FP16)

---

## Required Environment Variables

Add to your `~/.bashrc` or `~/.zshrc`:

```bash
# CUDA Architecture (L4 = Ada = SM_89)
export TORCH_CUDA_ARCH_LIST="8.9"

# Parallel Build Jobs
export MAX_JOBS=$(nproc)

# CUDA Compiler Flags
export CUDAFLAGS="-O3 --use_fast_math -lineinfo"

# Compilation Cache
export CCACHE_DIR="$HOME/.ccache"

# Python Path (if needed)
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
```

**After editing, reload**: `source ~/.bashrc` (or `~/.zshrc`)

---

## Required Tools

### 1. Ninja (Fast Parallel Builds)

```bash
pip install --user ninja
```

**Verification**:
```bash
which ninja  # Should show ~/.local/bin/ninja
ninja --version  # Should show 1.11+
```

### 2. ccache (Compilation Caching)

```bash
# Ubuntu/Debian
sudo apt-get install -y ccache

# macOS
brew install ccache

# Or via pip
pip install --user ccache
```

**Verification**:
```bash
ccache --version  # Should show 3.7+
ccache -s  # Show cache statistics
```

### 3. PyTorch with CUDA 12.1

```bash
pip install torch==2.2.1 torchvision --index-url https://download.pytorch.org/whl/cu121
```

**Verification**:
```bash
python -c "import torch; print(f'PyTorch {torch.__version__}, CUDA {torch.version.cuda}')"
# Expected: PyTorch 2.2.1+cu121, CUDA 12.1
```

### 4. Nsight Compute (Profiling)

```bash
# Already installed on GPU (see NSIGHT_COMPUTE_BASELINE_OCT14_2025.md)
/opt/nvidia/nsight-compute/2024.1.1/ncu --version
```

---

## Build Directory Structure

```
.torch_build/          # PyTorch JIT builds (gitignored)
build/                 # setuptools builds (gitignored)
~/.ccache/            # Compilation cache (persistent)
~/.torch_cuda_cache/  # Persistent kernel cache (gitignored)
```

**Clean caches** (if needed):
```bash
rm -rf .torch_build build/
ccache -C  # Clear ccache
```

---

## Environment Verification Script

Create `scripts/verify_env.sh`:

```bash
#!/usr/bin/env bash
set -euo pipefail

echo "🔍 Verifying CUDA Development Environment"
echo "=========================================="
echo ""

# Check environment variables
echo "📦 Environment Variables:"
echo "  TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST:-NOT SET}"
echo "  MAX_JOBS=${MAX_JOBS:-NOT SET}"
echo "  CUDAFLAGS=${CUDAFLAGS:-NOT SET}"
echo "  CCACHE_DIR=${CCACHE_DIR:-NOT SET}"
echo ""

# Check tools
echo "🔧 Tools:"
which ninja && ninja --version | head -1 || echo "  ❌ ninja not found"
which ccache && ccache --version | head -1 || echo "  ❌ ccache not found"
which ncu && ncu --version | head -1 || echo "  ⚠️  ncu not found (optional)"
echo ""

# Check PyTorch
echo "🐍 PyTorch:"
python -c "import torch; print(f'  PyTorch {torch.__version__}'); print(f'  CUDA {torch.version.cuda}'); print(f'  Device: {torch.cuda.get_device_name(0)}')" || echo "  ❌ PyTorch/CUDA not available"
echo ""

# Check GPU
echo "🎮 GPU:"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader || echo "  ❌ nvidia-smi failed"
echo ""

echo "✅ Verification complete"
```

**Run verification**:
```bash
chmod +x scripts/verify_env.sh
bash scripts/verify_env.sh
```

---

## Build Optimizations

### 1. Use Ninja for Parallel Builds

PyTorch will automatically use Ninja if available. Verify with:

```python
import torch.utils.cpp_extension
print(f"Ninja available: {torch.utils.cpp_extension.is_ninja_available()}")
```

### 2. Use ccache for Compilation Caching

**First build**: 5-15 minutes (cold cache)  
**Subsequent builds**: 10-30 seconds (hot cache)

**View cache stats**:
```bash
ccache -s  # Show hits/misses
ccache -z  # Reset stats
```

### 3. Persistent Build Cache

Use `build_directory` parameter in `torch.utils.cpp_extension.load()`:

```python
module = load(
    name="my_kernel",
    sources=["kernel.cu"],
    build_directory=".torch_build",  # Persistent across runs
    verbose=True
)
```

---

## Troubleshooting

### Issue: "Ninja not found"

**Solution**:
```bash
pip install --user ninja
export PATH="$HOME/.local/bin:$PATH"
echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.bashrc
```

### Issue: "ccache not working"

**Check if enabled**:
```bash
ccache -s  # Should show stats
```

**Enable ccache**:
```bash
export PATH="/usr/lib/ccache:$PATH"  # Ubuntu
export PATH="/usr/local/opt/ccache/libexec:$PATH"  # macOS
```

### Issue: "CUDA out of memory"

**Clear PyTorch cache**:
```python
import torch
torch.cuda.empty_cache()
```

**Reduce batch size** or **tile sizes** in kernel config.

### Issue: "Build still slow"

1. Check `MAX_JOBS`: `echo $MAX_JOBS` (should be CPU count)
2. Check Ninja: `torch.utils.cpp_extension.is_ninja_available()` (should be True)
3. Check ccache stats: `ccache -s` (should show hits on rebuild)
4. Verify arch list: `echo $TORCH_CUDA_ARCH_LIST` (should be "8.9" only)

---

## Performance Best Practices

### 1. Limit Architecture Targets

**Don't**:
```bash
export TORCH_CUDA_ARCH_LIST="7.0;7.5;8.0;8.6;8.9"  # Builds 5 architectures
```

**Do**:
```bash
export TORCH_CUDA_ARCH_LIST="8.9"  # Builds 1 architecture (L4 only)
```

**Impact**: 5× faster compilation

### 2. Use Fast Math

```bash
export CUDAFLAGS="-O3 --use_fast_math"
```

**Trade-off**: Slightly reduced numerical precision for 10-20% speedup

### 3. Enable Line Info for Profiling

```bash
export CUDAFLAGS="-O3 --use_fast_math -lineinfo"
```

**Benefit**: Nsight Compute can map performance to source lines

---

## Environment Lock for Reproducibility

**Lock file**: `bench/artifacts/env.json`

**Generate**:
```python
from cudadent42.bench.common.env_lock import lock_environment, write_env

lock_environment()  # Set FP16, disable TF32, enable deterministic
write_env("bench/artifacts/env.json")  # Save fingerprint
```

**Verify**:
```bash
cat bench/artifacts/env.json
```

**Expected**:
```json
{
  "dtype": "float16",
  "tf32": false,
  "deterministic": true,
  "cuda_version": "12.1",
  "pytorch_version": "2.2.1+cu121",
  "gpu": "NVIDIA L4",
  "driver": "570.172.08"
}
```

---

## Quick Start Checklist

- [ ] Add environment variables to `~/.bashrc`
- [ ] Install `ninja` and `ccache`
- [ ] Verify PyTorch CUDA 12.1
- [ ] Run `scripts/verify_env.sh`
- [ ] Test build with `bench/_build.py`
- [ ] Clear caches: `rm -rf .torch_build build/`
- [ ] Lock environment: `python -c "from cudadent42.bench.common.env_lock import lock_environment; lock_environment()"`

---

**Status**: Environment ready for hermetic CUDA kernel development

**Next**: Build pre-compiled extension (`ext/setup_fa_s512.py`)
