# H100 + CUDA 13.0 Final Deployment Status

**Date**: October 30, 2025, 15:15 UTC  
**Pod**: `tender_turquoise_herring` (154.57.34.98:30577)  
**Status**: ✅ CUDA 13.0 toolkit functional, ⚠️ runtime requires driver upgrade

---

## Executive Summary

**ACHIEVED**:
- ✅ CUDA 13.0.88 toolkit installed and verified
- ✅ CUTLASS 4.3.0 cloned and configured
- ✅ nvcc compilation works for sm_90a (H100)
- ✅ PyTorch 2.8.0+cu128 running with GPU access
- ✅ FlashCore repository deployed

**BLOCKER**:
- ❌ PyTorch cu130 requires driver 580.95.05+
- Current driver: 550.163.01 (30 versions behind)
- Gap prevents cu130 runtime (compilation only)

**THE NUMBER**: 
- SDPA baseline not yet measured on this pod
- Previous pod: 21.67 ms (21,673.97 μs), 225.71 μs/head

---

## Environment Details

### Hardware
```yaml
GPU: NVIDIA H100 80GB HBM3
Architecture: sm_90a (Hopper)
Driver: 550.163.01
RAM: 1.5TB
Storage: 30GB available
```

### Software Stack
```yaml
CUDA Toolkit: 13.0.88 ✓ (compilation)
CUDA Runtime: 12.8 (PyTorch constraint)
PyTorch: 2.8.0+cu128 ✓
Python: 3.12.3
OS: Ubuntu 24.04.3 LTS
CUTLASS: 4.3.0 (commit 8afb19d)
```

### Compilation Test Results
```bash
# CUDA 13.0 nvcc test
$ nvcc --version
Cuda compilation tools, release 13.0, V13.0.88

$ nvcc -O3 -gencode arch=compute_90,code=sm_90a -o test test.cu
# Output: 965KB binary ✓

# PyTorch GPU access
$ python -c "import torch; print(torch.cuda.is_available())"
True ✓
```

---

## Why CUDA 13.0 Matters (Recap)

Per NVIDIA Developer Blog (October 2025):

1. **Tile-based programming model** - New IR for structured parallelism
2. **Blackwell full support** - B200/GB200/B300 optimizations
3. **Memory enhancements** - HMM, host `cuMemCreate`
4. **5-15% speedup** on Hopper/Blackwell tensor-core workloads vs 12.8

**Reference**: https://developer.nvidia.com/blog/whats-new-and-important-in-cuda-toolkit-13-0/

---

## Solution Paths

### Option 1: Request New Pod with Driver 580+ (RECOMMENDED)

**Steps**:
1. Stop current pod
2. RunPod dashboard → Deploy Pod:
   - GPU: H100 80GB
   - Filter: Driver >= 580.95
   - Template: "NVIDIA CUDA 13.0 Devel" or "PyTorch 2.9+ cu130"
3. SSH in and verify:
   ```bash
   nvidia-smi --query-gpu=driver_version --format=csv,noheader
   # Expected: 580.95.05 or higher
   ```

**Timeline**: 5-15 minutes  
**Cost**: Same ($2.69/hr H100)  
**Benefit**: Full CUDA 13.0 feature set, no hybrid complexity

---

### Option 2: Hybrid Approach (Current Pod)

**Use CUDA 13.0 for compilation, cu128 for runtime**

#### Configuration Files Created

**`scripts/bootstrap_env.sh`**:
```bash
#!/bin/bash
set -e

echo "[Bootstrap] Setting CUDA 13.0 + CUTLASS 4.3.0 environment"

export CUDA_HOME=/usr/local/cuda-13.0
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-12.8/lib64:$LD_LIBRARY_PATH
export CUTLASS_HOME=/opt/cutlass

# PyTorch CUDA check bypass
export TORCH_CUDA_ARCH_LIST="8.9"
export FORCE_CUDA=1

# Verify
nvcc --version | grep "V13.0" && echo "✓ CUDA 13.0 active" || echo "❌ CUDA version mismatch"
python -c "import torch; print(f'PyTorch: {torch.__version__}, GPU: {torch.cuda.is_available()}')"

echo "[Bootstrap] Environment ready"
```

**`.cursor/preflight.yml`** (to be provided by user)

**Usage**:
```bash
# Every session
source scripts/bootstrap_env.sh

# Compile kernel with CUDA 13.0
nvcc -O3 -gencode arch=compute_90,code=sm_90a \
  --use_fast_math --shared --compiler-options '-fPIC' \
  -I$CUTLASS_HOME/include \
  -o attention_kernel.so attention_fmha.cu

# Run with cu128 runtime
python benchmarks/perf.py
```

**Limitation**: Tile-model runtime features unavailable (compilation-only optimizations)

---

### Option 3: Docker Container Isolation

**Build image with CUDA 13.0 toolkit + cu128 runtime**:

```dockerfile
FROM nvidia/cuda:13.0.2-devel-ubuntu22.04

RUN apt-get update && apt-get install -y \
    git cmake build-essential python3-pip wget

RUN git clone --depth 1 https://github.com/NVIDIA/cutlass.git /opt/cutlass

RUN pip3 install torch==2.8.0+cu128 \
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
```

---

## Current Pod File Structure

```
/workspace/BlackwellSparseK/ (FlashCore)
├── flashcore/
│   ├── fast/                    # 30+ attention CUDA kernels
│   ├── benchmark/               # High-performance bench suite
│   ├── tests/                   # Correctness tests (15 total)
│   └── requirements.txt
├── scripts/
│   └── bootstrap_env.sh         # ✓ Created (CUDA 13.0 setup)
├── .cursor/
│   ├── executors/
│   │   └── h100_remote.yml      # SSH config (outdated IP)
│   └── preflight.yml            # ⚠️ To be added
├── setup_ext.py                 # PyTorch extension builder
└── Dockerfile.cuda13            # ⚠️ To be created

/opt/cutlass/                    # CUTLASS 4.3.0 (8afb19d)
/usr/local/cuda-13.0/            # CUDA 13.0.88 toolkit ✓
/usr/local/cuda-12.8/            # CUDA 12.8 runtime ✓
```

---

## Verification Commands

**After any configuration change**:

```bash
# 1. Check CUDA toolkit
nvcc --version | grep "release 13.0"
# Expected: "release 13.0, V13.0.88"

# 2. Check PyTorch
python -c "import torch; print(f'CUDA: {torch.version.cuda}, GPU: {torch.cuda.is_available()}')"
# Expected: "CUDA: 12.8, GPU: True"

# 3. Check driver
nvidia-smi --query-gpu=driver_version --format=csv,noheader
# Current: 550.163.01

# 4. Test compilation
cd /tmp
cat > test.cu << 'EOF'
#include <cuda_runtime.h>
__global__ void test() {}
int main() { return 0; }
EOF
nvcc -O3 -gencode arch=compute_90,code=sm_90a -o test test.cu
./test 2>&1 && echo "✓ CUDA 13.0 compilation works"

# 5. Check environment persistence
echo $CUDA_HOME
# Expected: /usr/local/cuda-13.0
```

---

## Known Issues & Resolutions

### Issue 1: PyTorch Version Mismatch
**Error**: `RuntimeError: The detected CUDA version (13.0) mismatches... (12.8)`

**Solution**: Use `FORCE_CUDA=1` and bypass PyTorch's build system:
```bash
export FORCE_CUDA=1
export TORCH_CUDA_ARCH_LIST="8.9"
# Compile directly with nvcc instead of setup.py
```

### Issue 2: Cursor Environment Reset
**Problem**: Exports lost between terminal sessions

**Solution**: 
1. Add to `~/.bashrc`:
   ```bash
   source /workspace/BlackwellSparseK/scripts/bootstrap_env.sh
   ```
2. Create `.cursor/preflight.yml` to auto-source on startup

### Issue 3: Driver Version Too Old for cu130
**Error**: `CUDA initialization: driver too old (found version 12040)`

**Solution**: Either:
- Request new pod with driver 580+ (Option 1)
- Use hybrid cu128 runtime (Option 2)
- Wait for RunPod driver updates

---

## Next Actions (Priority Order)

### Immediate (< 5 min)
1. ✅ Create `scripts/bootstrap_env.sh` (done)
2. ⏳ Add `.cursor/preflight.yml` (user to provide)
3. ⏳ Test environment persistence: `source bootstrap_env.sh && nvcc --version`

### Short-term (< 1 hour)
4. ⏳ Create `Dockerfile.cuda13` for container isolation
5. ⏳ Build FlashCore kernels with CUDA 13.0:
   ```bash
   cd /workspace/BlackwellSparseK
   source scripts/bootstrap_env.sh
   make build  # or equivalent build command
   ```
6. ⏳ Run baseline benchmark to establish H100 performance

### Medium-term (< 1 day)
7. ⏳ Request new RunPod with driver 580+ for full cu130 support
8. ⏳ Run full benchmark suite with CUDA 13.0 features
9. ⏳ Document performance delta vs CUDA 12.8

---

## Cost Analysis

**Current Pod**:
- Rate: $2.69/hr (H100 80GB on-demand)
- Runtime: 28 seconds active
- Total cost: ~$0.02 (negligible)

**Recommendation**: Keep pod running if actively developing, otherwise stop and restart with driver 580+ when needed.

---

## References

1. **CUDA 13.0 Release Notes**: https://developer.nvidia.com/blog/whats-new-and-important-in-cuda-toolkit-13-0/
2. **CUTLASS 4.3.0 Docs**: https://github.com/NVIDIA/cutlass/tree/v4.3.0
3. **PyTorch CUDA Compatibility**: https://pytorch.org/get-started/locally/
4. **RunPod Driver Updates**: Contact support@runpod.io for driver 580+ availability

---

## Contacts & Support

**Pod Details**:
- Name: `tender_turquoise_herring`
- ID: `eh0hlbk21mq1dn`
- SSH: `root@154.57.34.98 -p 30577`
- Status: ✅ Running

**Support**:
- RunPod Support: support@runpod.io (response < 24hrs)
- NVIDIA CUDA: https://forums.developer.nvidia.com/

---

**Last Updated**: October 30, 2025, 15:15 UTC  
**Next Review**: After preflight.yml integration or driver upgrade

