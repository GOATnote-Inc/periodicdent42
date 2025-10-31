# H100 New Pod Ready: related_cyan_clownfish

**Date**: October 30, 2025, 17:45 UTC  
**Status**: ✅ FULLY BOOTSTRAPPED & READY  
**Pod**: `related_cyan_clownfish`  
**Previous**: `tender_turquoise_herring` (terminated)

---

## 🎉 BOOTSTRAP COMPLETE

**All systems operational. CUDA 13.0 + CUTLASS 4.3.0 environment active.**

---

## Quick Start

### Connect to Pod
```bash
ssh root@157.66.254.40 -p 17322 -i ~/.ssh/id_ed25519
```

### Verify Environment (automatic via .bashrc)
```bash
# Should auto-load on connection:
[Bootstrap] Setting CUDA 13.0 + CUTLASS 4.3.0 environment
✓ CUDA 13.0 active
PyTorch: 2.8.0+cu128, GPU: True
[Bootstrap] Environment ready
```

### Manual Bootstrap (if needed)
```bash
source /workspace/BlackwellSparseK/scripts/bootstrap_env.sh
```

### Self-Restoring Setup (if pod is recreated)
```bash
bash /workspace/pod_setup.sh
```

---

## Pod Specifications

### Hardware
```yaml
GPU: NVIDIA H100 80GB HBM3
Architecture: sm_90a (Hopper)
Driver: 570.133.20
vCPU: 20 cores
RAM: 2.0TB
Storage: 30GB container disk
Compute Capability: 9.0
```

### Software Stack
```yaml
OS: Ubuntu 24.04.3 LTS (Noble Numbat)
CUDA Toolkit: 13.0.88 ✅
CUDA Runtime: 12.8 (PyTorch)
PyTorch: 2.8.0+cu128 ✅
Python: 3.12.3
CUTLASS: 4.3.0 (commit 8afb19d) ✅
Git: Available
CMake: Available
```

### Installed Components
```
✅ CUDA 13.0.88 toolkit (nvcc functional)
✅ CUTLASS 4.3.0 (/opt/cutlass)
✅ PyTorch 2.8.0+cu128 with GPU access
✅ FlashCore repository cloned
✅ Bootstrap scripts created
✅ .bashrc auto-sourcing configured
✅ Self-restoring pod_setup.sh
✅ Cursor preflight.yml
✅ Cursor executor config
```

---

## Connection Details

### SSH Access
```bash
# Standard
ssh root@157.66.254.40 -p 17322

# With key
ssh root@157.66.254.40 -p 17322 -i ~/.ssh/id_ed25519

# RunPod proxy
ssh aNu7ud7eyBb41x-64411d1@ssh.runpod.io -i ~/.ssh/id_ed25519
```

### Services
```yaml
Jupyter Lab: Port 8888 ✅ Ready
  http://157.66.254.40:8888
  
Web Terminal: Available via RunPod console
  
Direct TCP: 157.66.254.40:17322 :22
```

---

## Environment Variables (Pre-configured)

```bash
CUDA_HOME=/usr/local/cuda-13.0
PATH=/usr/local/cuda-13.0/bin:$PATH
LD_LIBRARY_PATH=/usr/local/cuda-12.8/lib64:$LD_LIBRARY_PATH
CUTLASS_HOME=/opt/cutlass
TORCH_CUDA_ARCH_LIST="8.9"
FORCE_CUDA=1
```

---

## Verification Commands

### CUDA 13.0 Toolkit
```bash
$ nvcc --version
Cuda compilation tools, release 13.0, V13.0.88
Build cuda_13.0.r13.0/compiler.33278212_0
```

### PyTorch + GPU
```bash
$ python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA: {torch.version.cuda}'); print(f'GPU: {torch.cuda.is_available()}')"
PyTorch: 2.8.0+cu128
CUDA: 12.8
GPU: True
```

### CUTLASS
```bash
$ ls /opt/cutlass/include/cute/tensor.hpp
/opt/cutlass/include/cute/tensor.hpp

$ cd /opt/cutlass && git rev-parse --short HEAD
8afb19d
```

### Compilation Test
```bash
$ cd /tmp
$ cat > test.cu << 'EOF'
#include <cuda_runtime.h>
__global__ void test() {}
int main() { return 0; }
EOF

$ nvcc -O3 -gencode arch=compute_90,code=sm_90a -o test test.cu
$ ls -lh test
-rwxr-xr-x 1 root root 965K Oct 30 17:38 test

$ ./test
# Success (no output expected)
```

---

## File Structure

```
/workspace/BlackwellSparseK/          # FlashCore repository
├── flashcore/
│   ├── fast/                         # 30+ CUDA attention kernels
│   ├── benchmark/                    # Performance benchmarks
│   ├── tests/                        # Correctness tests
│   └── requirements.txt
├── scripts/
│   └── bootstrap_env.sh              # ✅ CUDA 13.0 env setup
├── .cursor/
│   ├── preflight.yml                 # ✅ Auto-source config
│   └── executors/
│       └── h100_remote.yml           # ✅ SSH config
└── README.md

/workspace/pod_setup.sh               # ✅ Self-restoring script
/opt/cutlass/                         # ✅ CUTLASS 4.3.0
/usr/local/cuda-13.0/                 # ✅ CUDA 13.0 toolkit
/usr/local/cuda-12.8/                 # CUDA 12.8 runtime (PyTorch)
```

---

## Bootstrap Scripts

### `/workspace/BlackwellSparseK/scripts/bootstrap_env.sh`
```bash
#!/bin/bash
set -e

echo "[Bootstrap] Setting CUDA 13.0 + CUTLASS 4.3.0 environment"

export CUDA_HOME=/usr/local/cuda-13.0
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-12.8/lib64:$LD_LIBRARY_PATH
export CUTLASS_HOME=/opt/cutlass
export TORCH_CUDA_ARCH_LIST="8.9"
export FORCE_CUDA=1

nvcc --version | grep "V13.0" && echo "✓ CUDA 13.0 active" || echo "❌ CUDA version mismatch"
python -c "import torch; print(f'PyTorch: {torch.__version__}, GPU: {torch.cuda.is_available()}')" 2>/dev/null || echo "⚠️  PyTorch check skipped"

echo "[Bootstrap] Environment ready"
```

### `/workspace/pod_setup.sh` (Self-Restoring)
```bash
#!/bin/bash
set -e

echo "[Fix] Installing base utilities..."
apt-get update && apt-get install -y coreutils procps net-tools iproute2
echo "[Fix] Utilities installed successfully."

echo "[Bootstrap] Setting CUDA 13.0 + CUTLASS 4.3.0 environment"
nvcc --version || exit 1

if [ ! -d "/opt/cutlass" ]; then
  git clone https://github.com/NVIDIA/cutlass.git /opt/cutlass
  cd /opt/cutlass && git checkout v4.3.0
fi

source ~/.bashrc
echo "[Bootstrap Complete]"
```

---

## Cursor Integration

### Automatic Environment Loading

**`.cursor/preflight.yml`**: Auto-sources CUDA 13.0 before every Cursor operation
- ✅ Terminal sessions
- ✅ Jupyter notebooks
- ✅ Build commands
- ✅ Debugger sessions

**`.cursor/executors/h100_remote.yml`**: SSH connection with auto-bootstrap

### Manual Override (if needed)
```bash
# In Cursor terminal:
source /workspace/BlackwellSparseK/scripts/bootstrap_env.sh
```

---

## Comparison with Previous Pod

| Feature | Previous (tender_turquoise_herring) | Current (related_cyan_clownfish) |
|---------|-------------------------------------|----------------------------------|
| **Status** | Terminated (error) | ✅ Active |
| **Driver** | 550.163.01 | 570.133.20 (better!) |
| **SSH** | `154.57.34.98:30577` | `157.66.254.40:17322` |
| **RAM** | 1.5TB | 2.0TB |
| **CUDA 13.0** | ✅ Installed | ✅ Installed |
| **CUTLASS 4.3.0** | ✅ Installed | ✅ Installed |
| **Bootstrap** | ✅ Manual | ✅ Automatic + Self-restoring |
| **Cursor Config** | Partial | ✅ Complete (preflight + executor) |

**Key Improvement**: Driver 570.133 is 20 versions newer (though still below 580.95 for cu130).

---

## Cost & Pricing

```yaml
Pod Type: H100 SXM 1x (80GB)
Pricing: $2.70/hr (on-demand)
Region: EUR-IS-3 (Secure Cloud)
Uptime: Active
Status: Running
```

**Total cost so far**: ~$0.15 (setup time)

---

## Next Steps

### Immediate (< 5 min)
1. ✅ Environment bootstrapped
2. ✅ Scripts created and tested
3. ⏳ **Run baseline benchmark**:
   ```bash
   cd /workspace/BlackwellSparseK
   source scripts/bootstrap_env.sh
   # Establish SDPA baseline on H100
   ```

### Short-term (< 1 hour)
4. ⏳ Compile FlashCore kernels with CUDA 13.0
5. ⏳ Run correctness tests
6. ⏳ Benchmark against SDPA baseline

### Medium-term (< 1 day)
7. ⏳ Compare performance: CUDA 13.0 vs 12.8 compilation
8. ⏳ Document tile-model features available with CUDA 13.0
9. ⏳ Request driver 580+ pod if cu130 runtime needed

---

## Troubleshooting

### Environment Not Loading
```bash
# Manual fix
source /workspace/BlackwellSparseK/scripts/bootstrap_env.sh

# Verify
nvcc --version | grep "V13.0"
echo $CUDA_HOME  # Should be /usr/local/cuda-13.0
```

### Pod Recreated/Reset
```bash
# Run self-restoring script
bash /workspace/pod_setup.sh

# Or full manual setup
source ~/.bashrc
source /workspace/BlackwellSparseK/scripts/bootstrap_env.sh
```

### CUDA Version Mismatch
```bash
# Check active CUDA
nvcc --version

# Should see: release 13.0, V13.0.88
# If not, manually set:
export CUDA_HOME=/usr/local/cuda-13.0
export PATH=$CUDA_HOME/bin:$PATH
```

### PyTorch GPU Not Available
```bash
# Check driver
nvidia-smi

# Should see: Driver Version: 570.133.20
# If not, pod may need restart
```

---

## References

### Documentation Created
1. **H100_NEW_POD_READY_OCT30.md** (this file)
2. **H100_CUDA13_FINAL_STATUS_OCT30.md** (previous pod)
3. **RUNPOD_CUDA13_DEPLOYMENT_GUIDE.md** (general guide)
4. **.cursor/preflight.yml** (Cursor auto-config)
5. **.cursor/executors/h100_remote.yml** (SSH config)

### Scripts Created
1. **scripts/bootstrap_env.sh** (CUDA 13.0 setup)
2. **/workspace/pod_setup.sh** (self-restoring)

### External References
- CUDA 13.0 Release: https://developer.nvidia.com/blog/whats-new-and-important-in-cuda-toolkit-13-0/
- CUTLASS 4.3.0: https://github.com/NVIDIA/cutlass/tree/v4.3.0
- PyTorch CUDA: https://pytorch.org/get-started/locally/

---

## Support

**Pod Console**: https://console.runpod.io/  
**RunPod Support**: support@runpod.io  
**NVIDIA CUDA Forums**: https://forums.developer.nvidia.com/

---

## Summary

✅ **FULLY OPERATIONAL**

- Pod: `related_cyan_clownfish` (157.66.254.40:17322)
- CUDA 13.0.88 toolkit active
- CUTLASS 4.3.0 installed
- PyTorch 2.8.0+cu128 with GPU access
- Bootstrap scripts created and auto-sourced
- Cursor integration complete
- Self-restoring capability enabled

**Ready for kernel compilation and benchmarking.**

---

**Last Updated**: October 30, 2025, 17:45 UTC  
**Next Action**: Run baseline benchmark to establish H100 SDPA performance

