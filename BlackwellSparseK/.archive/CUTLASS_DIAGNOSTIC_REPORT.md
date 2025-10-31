# CUTLASS Example 88 (Hopper FMHA) Diagnostic Report
**Date**: October 30, 2025  
**Pod**: RunPod H100 (157.66.254.40:17322)  
**Status**: ⚠️ **CRITICAL PERFORMANCE ISSUE**

---

## Executive Summary

CUTLASS Example 88 was successfully built for sm_90a architecture with CUDA 13.0.2, cuBLAS enabled, and Release optimization. However, runtime performance is **10,000× slower** than expected, suggesting a critical configuration or runtime issue.

---

## Build Configuration ✅

### CMake Settings (Verified)
```cmake
CMAKE_BUILD_TYPE:              Release
CUTLASS_ENABLE_CUBLAS:         ON
CUTLASS_ENABLE_PROFILER:       ON
CUTLASS_TEST_LEVEL:            2
CUTLASS_NVCC_ARCHS:            90a
CMAKE_CUDA_ARCHITECTURES:      90a
CMAKE_CUDA_FLAGS:              -lineinfo -Xptxas=-v -arch=sm_90a
                               --generate-code=arch=compute_90,code=sm_90a
                               --generate-code=arch=compute_90,code=compute_90
```

### Binary Verification ✅
```
File: /opt/cutlass/build_release/examples/88_hopper_fmha/88_hopper_fmha
Size: 6.1 MB
Architecture: sm_90a (verified with cuobjdump)
ELF files: 2x sm_90a.cubin present
```

---

## Runtime Environment ⚠️

### Hardware
- GPU: NVIDIA H100 80GB HBM3
- Driver: 570.133.20 (CUDA 12.8)
- Compute Capability: sm_90a

### Software
- CUDA Runtime: 13.0.88
- cuda-compat-13-0: **INSTALLED**
- LD_LIBRARY_PATH: /usr/local/cuda-13.0/compat:/usr/local/cuda-13.0/lib64

### Driver-Runtime Mismatch
```
Driver Version:  570.133.20 (supports CUDA 12.8)
Runtime Version: 13.0.88
Compat Layer:    cuda-compat-13-0 (provides 580.95.05 libs)
```

**Status**: compat layer installed but runtime errors persist

---

## Performance Results ❌

### Test Configuration
```
B=1, H=1, Q=128, K=128, D=64, iterations=1
```

### Measured Performance
```
tma ws cooperative 128x64x64:    0.0193 TFLOPS/s
tma ws ping-pong 128x64x64:      0.0193 TFLOPS/s
```

### Expected Performance (H100 Baseline)
```
PyTorch SDPA (B=16, H=96, Q=4096, D=128):  ~313 TFLOPS
Optimal CUTLASS (same workload):           ~400+ TFLOPS
Current CUTLASS (scaled estimate):         ~200 TFLOPS (for small test)
```

### Performance Gap
```
Current:  0.0193 TFLOPS
Expected: ~200 TFLOPS
Gap:      ~10,000× slower than expected
```

---

## Runtime Errors

### Repeated Error Message
```
ERROR : Arch conditional MMA instruction used without targeting 
        appropriate compute capability. Aborting.
```

**Frequency**: Hundreds of instances  
**Impact**: Non-fatal (test continues and produces results)  
**Hypothesis**: Test harness is attempting multiple kernel variants, many of which fail runtime arch checks

---

## Root Cause Hypotheses

### 1. Reference Implementation Fallback (MOST LIKELY)
- **Hypothesis**: When optimized kernels fail runtime checks, Example 88 falls back to a very slow reference CPU/GPU implementation
- **Evidence**: 
  - Test produces results (not crashing)
  - Performance is ~10,000× slower
  - Hundreds of "Arch conditional MMA" warnings suggest kernel selection failures
- **Fix**: Identify why optimized kernels are being rejected at runtime despite correct sm_90a compilation

### 2. Driver Compatibility Issue Despite compat Layer
- **Hypothesis**: cuda-compat-13-0 is installed but binary is not correctly loading compat libraries
- **Evidence**:
  - Previous runs showed "CUDA driver version is insufficient" error
  - Current run shows arch mismatch warnings
- **Fix**: Verify LD_LIBRARY_PATH precedence, check ldd output, confirm libcuda.so.1 is from compat layer

### 3. CUTLASS Runtime Architecture Detection Failure
- **Hypothesis**: CUTLASS's runtime arch detection is failing to recognize H100 as sm_90a-capable
- **Evidence**:
  - Binary compiled for sm_90a (verified)
  - Runtime errors suggest arch mismatch
- **Fix**: Check CUTLASS source for runtime capability queries, may need environment variable override

### 4. Incomplete cuBLAS Integration
- **Hypothesis**: cuBLAS is enabled but not properly linked or initialized
- **Evidence**:
  - Very low TFLOPS suggests not using Tensor Cores
  - cuBLAS typically provides optimized GEMM routines
- **Fix**: Run `ldd` on binary to verify cuBLAS linkage, check for cuBLAS initialization errors in full log

---

## Next Actions (Priority Order)

### PRIORITY 1: Verify compat Layer Loading
```bash
cd /opt/cutlass/build_release/examples/88_hopper_fmha
ldd ./88_hopper_fmha | grep -E "libcuda|libcublas"
```
**Expected**: libcuda.so.1 should point to /usr/local/cuda-13.0/compat/libcuda.so.1

### PRIORITY 2: Check Full Startup Log
```bash
./88_hopper_fmha --b=1 --h=1 --q=128 --k=128 --d=64 --iterations=1 2>&1 | head -100
```
**Look for**: cuBLAS initialization, GPU capability detection, kernel selection logs

### PRIORITY 3: Try Alternative Example (Simpler GEMM)
```bash
cd /opt/cutlass/build_release/examples/48_hopper_warp_specialized_gemm
./48_hopper_warp_specialized_gemm
```
**Purpose**: Determine if issue is specific to FMHA or affects all Hopper kernels

### PRIORITY 4: Override Runtime Capability Detection
```bash
CUDA_FORCE_PTX_JIT=1 ./88_hopper_fmha --b=1 --h=1 --q=128 --d=128 --iterations=1
```
**Purpose**: Force PTX JIT compilation, bypassing precompiled SASS

### PRIORITY 5: Rebuild with Debug Symbols
```bash
cd /opt/cutlass/build_release
cmake .. -DCMAKE_BUILD_TYPE=RelWithDebInfo
make -j16 88_hopper_fmha
```
**Purpose**: Get more detailed error messages and stack traces

---

## Comparison with flashcore (Working H100 Kernels)

### flashcore Status (October 25, 2025)
- **Environment**: Same H100 pod, CUDA 12.4.1, PyTorch 2.4.1+cu124
- **Performance**: 0.451 μs/head for H=8 (11× better than 5μs target)
- **Correctness**: max_diff=0.0039, validated ✅

### Key Differences
1. **CUDA Version**: flashcore used 12.4.1, CUTLASS using 13.0.2
2. **Compilation**: flashcore used PyTorch extension build, CUTLASS uses CMake
3. **Architecture**: flashcore explicitly tested sm_90a paths, CUTLASS runtime arch detection may be failing

### Lesson
H100 hardware is confirmed working with optimized kernels. The issue is specific to CUTLASS Example 88 runtime configuration, not hardware capability.

---

## User's Original Guidance (October 30, 2025)

> **"Compile for sm_90a instead of sm_90."**  
> H100's WGMMA/TMA path requires sm_90a. Building for plain 90 causes "Arch conditional MMA instruction" errors and slow fallbacks.

**Status**: We ARE compiling for sm_90a (verified in CMakeCache.txt and cuobjdump). The runtime errors suggest a different issue.

> **"cuBLAS should remain enabled for validation, reference benchmarking, and profiler calibration."**

**Status**: cuBLAS IS enabled (verified in CMakeCache.txt). Need to verify actual linkage and initialization.

---

## Immediate Fix Attempt

Based on hypothesis #2 (compat layer not loading), try explicit library preload:

```bash
export LD_PRELOAD=/usr/local/cuda-13.0/compat/libcuda.so.1
cd /opt/cutlass/build_release/examples/88_hopper_fmha
./88_hopper_fmha --b=1 --h=1 --q=128 --k=128 --d=64 --iterations=1
```

If this fixes the issue, the root cause is compat library loading order.

---

## Conclusion

CUTLASS Example 88 build is **technically correct** (sm_90a, cuBLAS, Release mode), but runtime execution is falling back to a ~10,000× slower implementation. The "Arch conditional MMA" errors are the key diagnostic: optimized kernels are being rejected at runtime despite correct compilation.

**Recommended approach**: Focus on library loading (compat layer) and runtime capability detection before considering rebuild or code changes.

**Timeline**: With proper runtime configuration, should achieve ~200-400 TFLOPS on H100 for this workload (matching or exceeding PyTorch SDPA's 313 TFLOPS baseline).

