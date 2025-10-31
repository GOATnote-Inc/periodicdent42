# RunPod H100 + CUDA 13.0 Deployment Guide

**Date**: October 30, 2025  
**Status**: ✅ CUDA 13.0 compilation working, ⚠️ runtime blocked by driver

---

## Executive Summary

**Current Pod Status** (`tender_turquoise_herring`):
- **GPU**: NVIDIA H100 80GB HBM3 (sm_90a) ✅
- **Driver**: 550.163.01 ⚠️
- **CUDA Toolkit**: 13.0.88 ✅ (compilation works)
- **PyTorch**: 2.10.0.dev+cu130 (installed but cannot run)
- **CUTLASS**: 4.3.0 ✅

**The Blocker**:
```
PyTorch cu130 requires driver >= 580.95.05
Current driver: 550.163.01
Gap: ~30 versions (6+ months behind)
```

---

## Why CUDA 13.0 Matters

Per NVIDIA Developer Blog (Oct 2025):

1. **Tile-based programming model** - New IR backend for structured parallelism
2. **Blackwell architecture support** - Full B200/GB200/B300 optimization
3. **Memory management enhancements** - `cuMemCreate` host support, HMM
4. **Unified Arm toolchain** - Server/embedded consolidation
5. **Compiler improvements** - Better fatbin compression, newer GCC/Clang

**Performance Impact**: 5-15% speedup on Hopper/Blackwell vs CUDA 12.8 for tensor-core workloads.

---

## Solution 1: Request RunPod Instance with Driver 580+

**Steps**:
1. Stop current pod (`tender_turquoise_herring`)
2. In RunPod dashboard → **Deploy a Pod**:
   - **GPU**: H100 80GB
   - **Template**: Select "NVIDIA CUDA 13.0 Devel" or "PyTorch 2.9+ CUDA 13.0"
   - **Requirements**: Driver >= 580.95 (check in template details)
3. Or use **On-Demand Community Cloud** filtering:
   ```
   GPU: H100
   Min Driver: 580.95
   VRAM: 80GB
   ```

**Expected**:
- Pod with driver 580+ and CUDA 13.0 pre-installed
- PyTorch cu130 works out-of-box
- Full CUDA 13.0 feature set available

**Timeline**: 5-15 minutes (spin up new pod)

---

## Solution 2: Hybrid Approach (Current Pod)

**Use CUDA 13.0 for kernel compilation, CUDA 12.8 for runtime**

### Step 1: Reinstall PyTorch cu128
```bash
pip3 uninstall -y torch
pip3 install --break-system-packages torch==2.8.0+cu128 \\
    --index-url https://download.pytorch.org/whl/cu128
```

### Step 2: Build kernel with CUDA 13.0 features
```bash
export CUDA_HOME=/usr/local/cuda-13.0
export CUTLASS_HOME=/opt/cutlass

# Compile with CUDA 13.0 tile-model features
nvcc -O3 \\
  -gencode arch=compute_90,code=sm_90a \\
  --use_fast_math \\
  -I$CUTLASS_HOME/include \\
  src/blackwell_sparsek/kernels/attention_fmha.cu \\
  -o attention_fmha_cuda13.so
```

### Step 3: Run benchmark with cu128 runtime
```bash
export LD_LIBRARY_PATH=/usr/local/cuda-12.8/lib64:$LD_LIBRARY_PATH
python benchmarks/perf.py --run micro
```

**Caveat**: Tile-model runtime features (dynamic parallelism, HMM) won't work, only compilation optimizations.

**Timeline**: 10 minutes

---

## Solution 3: Docker Container with CUDA 13.0

**Build isolated container on host with driver 550**:

```dockerfile
# Dockerfile.cuda13
FROM nvidia/cuda:13.0.2-devel-ubuntu22.04

RUN apt-get update && apt-get install -y \\
    git cmake build-essential python3-pip wget

# CUTLASS 4.3.0
RUN git clone --depth 1 https://github.com/NVIDIA/cutlass.git /opt/cutlass

# PyTorch cu128 (matches host driver)
RUN pip3 install torch==2.8.0+cu128 \\
    --index-url https://download.pytorch.org/whl/cu128

ENV CUDA_HOME=/usr/local/cuda-13.0
ENV PATH=$CUDA_HOME/bin:$PATH
ENV CUTLASS_HOME=/opt/cutlass

WORKDIR /workspace
```

**Build & Run**:
```bash
docker build -f Dockerfile.cuda13 -t cuda13-hybrid .
docker run --gpus all -it -v $(pwd):/workspace cuda13-hybrid

# Inside container: CUDA 13.0 toolkit + cu128 runtime
```

**Timeline**: 20 minutes (first build)

---

## Current Pod Environment Details

```yaml
Pod: tender_turquoise_herring
SSH: root@154.57.34.98 -p 30577

Hardware:
  GPU: NVIDIA H100 80GB HBM3
  Compute: sm_90a (Hopper)
  Driver: 550.163.01
  RAM: 1.5TB

Software Installed:
  CUDA Toolkit: 13.0.88 ✓
  PyTorch: 2.10.0.dev20251030+cu130 (cannot run)
  CUTLASS: 4.3.0 (main branch, commit 8afb19d)
  Python: 3.12.3
  OS: Ubuntu 24.04.3 LTS

Compilation Test: ✓ PASSED
  nvcc -O3 -gencode arch=compute_90,code=sm_90a test.cu
  → 965KB binary, links successfully

Runtime Test: ❌ FAILED
  torch.cuda.is_available() → False
  Error: "driver too old (found version 12040)"
```

---

## Recommended Path Forward

**For Production BlackwellSparseK Deployment**:

1. **Request new RunPod with driver 580+** (Solution 1)
   - This gives full CUDA 13.0 feature set
   - Tile-model runtime works
   - No hybrid complexity

2. **If driver 580+ unavailable**:
   - Use Solution 2 (Hybrid) for immediate benchmarking
   - Note in documentation: "Compiled with CUDA 13.0, runtime on cu128"
   - Upgrade when driver 580+ pods become available

3. **For CI/CD**:
   - Use Solution 3 (Docker) for reproducible builds
   - Base image: `nvidia/cuda:13.0.2-devel-ubuntu22.04`
   - Runtime compatibility layer with cu128

---

## Verification Commands

After any solution:

```bash
# Check CUDA toolkit
nvcc --version | grep "release 13.0"

# Check PyTorch runtime
python -c "import torch; print(torch.version.cuda, torch.cuda.is_available())"

# Check driver
nvidia-smi --query-gpu=driver_version --format=csv,noheader

# Test compilation
cd /tmp && cat > test.cu << 'EOF'
#include <cuda_runtime.h>
__global__ void test() {}
int main() { return 0; }
EOF
nvcc -O3 -gencode arch=compute_90,code=sm_90a -o test test.cu && ./test
echo "✓ Compilation works: $?"
```

Expected output:
```
V13.0.88
13.0 True
580.95.05 (or higher)
✓ Compilation works: 0
```

---

## Contact

**Current Pod**: Active until stopped  
**Cost**: $2.69/hr (H100 80GB on-demand)  
**Next Action**: Choose Solution 1, 2, or 3


