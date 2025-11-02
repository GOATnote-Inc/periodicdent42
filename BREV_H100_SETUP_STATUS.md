# Brev H100 Environment Setup Status

**Date:** November 2, 2025  
**Pod:** awesome-gpu-name (Brev H100 PCIe 85GB)

## ✅ Completed

### Software Installed
- **CUDA 13.0.2** (`/usr/local/cuda-13.0`)
  - nvcc 13.0.88
  - cuBLAS 13.1.0.3
  - All development libraries
- **Nsight Compute 2025.3.1** (installed with CUDA)
- **CUTLASS 4.3.0** (`/opt/cutlass`, main branch)
- **Rust nightly** (1.93.0-nightly)
- **CMake, Ninja, build-essential** (all dev tools)

### Environment
- `/workspace` directory configured
- Preflight check script: `/workspace/preflight.sh`
- Test programs compiled and ready

## ⚠️ Requires Reboot

### Current Driver: 570.172.08
### Required Driver: ≥580.82.07

**Issue:** CUDA 13.0.2 runtime requires driver 580+. Current driver (570) is too old.

### Solution Options

**Option 1: Reboot Pod (Recommended)**
```bash
sudo reboot
# After reboot, driver 580.95.05 should be active
```

**Option 2: Fresh Pod**
Request new Brev H100 pod with driver 580+

## Next Steps After Reboot

1. Verify driver:
```bash
nvidia-smi --query-gpu=driver_version --format=csv,noheader
# Should show: 580.95.05
```

2. Test CUDA 13:
```bash
cd /workspace
export LD_LIBRARY_PATH=/usr/local/cuda-13.0/lib64:$LD_LIBRARY_PATH
export PATH=/usr/local/cuda-13.0/bin:$PATH
./cuda13_test
```

3. Baseline cuBLAS:
```bash
# Should achieve ~591 TFLOPS on H100 PCIe
```

4. Build modern CUTLASS kernel:
```bash
/usr/local/cuda-13.0/bin/nvcc -O3 -std=c++20 -arch=sm_90a \
  -I/opt/cutlass/include \
  your_kernel.cu -o kernel -lcudart
```

## Environment Variables

Add to `~/.bashrc`:
```bash
export CUDA_HOME=/usr/local/cuda-13.0
export PATH=$CUDA_HOME/bin:$HOME/.cargo/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
export CUTLASS_HOME=/opt/cutlass
export CPATH=$CUTLASS_HOME/include:$CPATH
```

## Verified Baseline (CUDA 12.8)

Before upgrade, confirmed H100 PCIe baseline:
- **cuBLAS (CUDA 12.8):** 591.6 TFLOPS
- **Target (Our Kernel):** 598.9 TFLOPS (1.01× cuBLAS)

---

**Status:** Ready for cooking after driver reboot 🔥
