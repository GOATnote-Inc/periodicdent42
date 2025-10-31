#!/bin/bash
# CUTLASS 4.3 + CUDA 13.0 STRICT BUILD SCRIPT
# NO HEURISTICS, NO AUTO-FIX, LITERAL EXECUTION ONLY
#
# Purpose: Build Example 88 (Hopper FMHA) with verified sm_90a targeting
# Author: Expert CUDA Engineer
# Date: October 30, 2025

set -euo pipefail

# =============================================================================
# SECTION 1: ENVIRONMENT LOCK
# =============================================================================
echo "======================================================================"
echo "  STRICT-EXECUTION MODE: Architecture Lock for H100 (sm_90a)"
echo "======================================================================"
echo ""

export CUDA_HOME=/usr/local/cuda-13.0
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/compat:/usr/local/lib/python3.12/dist-packages/nvidia/cuda_runtime/lib:$CUDA_HOME/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}

# Lock architecture to H100 sm_90a (Hopper)
export TORCH_CUDA_ARCH_LIST="9.0"
export CUDAARCHS="90"

echo "Environment Configuration:"
echo "  CUDA_HOME:              $CUDA_HOME"
echo "  CUDA Version:           $(nvcc --version | grep release | awk '{print $5}' | cut -d, -f1)"
echo "  Target Architecture:    sm_90a (H100 Hopper)"
echo "  TORCH_CUDA_ARCH_LIST:   $TORCH_CUDA_ARCH_LIST"
echo "  CUDAARCHS:              $CUDAARCHS"
echo ""

# Verify GPU is H100
GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
if [[ ! "$GPU_NAME" =~ "H100" ]]; then
    echo "❌ FATAL: Expected H100, found: $GPU_NAME"
    echo "   This script is locked to sm_90a. Aborting."
    exit 1
fi
echo "✅ GPU Verification: $GPU_NAME"
echo ""

# =============================================================================
# SECTION 2: CLEAN BUILD
# =============================================================================
echo "======================================================================"
echo "  SECTION 2: Clean Build Directory"
echo "======================================================================"
cd /opt/cutlass
rm -rf build_release
mkdir -p build_release
cd build_release
echo "✅ Clean build directory created"
echo ""

# =============================================================================
# SECTION 3: CMAKE CONFIGURATION (LOCKED)
# =============================================================================
echo "======================================================================"
echo "  SECTION 3: CMake Configuration (Release + cuBLAS + Profiler)"
echo "======================================================================"
echo ""
echo "Configuration Matrix:"
echo "  CMAKE_BUILD_TYPE:        Release"
echo "  CUTLASS_ENABLE_CUBLAS:   ON"
echo "  CUTLASS_ENABLE_PROFILER: ON"
echo "  CUTLASS_TEST_LEVEL:      2"
echo "  CUTLASS_NVCC_ARCHS:      90"
echo "  CMAKE_CUDA_ARCHITECTURES: 90"
echo ""

cmake .. \
  -DCMAKE_BUILD_TYPE=Release \
  -DCUTLASS_ENABLE_CUBLAS=ON \
  -DCUTLASS_ENABLE_PROFILER=ON \
  -DCUTLASS_TEST_LEVEL=2 \
  -DCUTLASS_NVCC_ARCHS=90 \
  -DCMAKE_CUDA_ARCHITECTURES=90 \
  -DCUTLASS_ENABLE_EXAMPLES=ON \
  -DCMAKE_CUDA_COMPILER=$CUDA_HOME/bin/nvcc \
  -DCMAKE_CUDA_FLAGS="-lineinfo -Xptxas=-v" \
  > /tmp/cutlass_cmake.log 2>&1

if [ $? -ne 0 ]; then
    echo "❌ CMake configuration failed. Log:"
    tail -50 /tmp/cutlass_cmake.log
    exit 1
fi

echo "✅ CMake configuration complete"
echo ""
echo "Verification of key settings:"
grep -E "CMAKE_BUILD_TYPE|CUTLASS_ENABLE_CUBLAS|CUTLASS_ENABLE_PROFILER|CUDA.*Arch" /tmp/cutlass_cmake.log | head -10
echo ""

# =============================================================================
# SECTION 4: BUILD
# =============================================================================
echo "======================================================================"
echo "  SECTION 4: Build Example 88 + Profiler"
echo "======================================================================"
echo ""

echo "Building (parallel, ~3 minutes)..."
make -j$(nproc) 88_hopper_fmha cutlass_profiler > /tmp/cutlass_build.log 2>&1

if [ $? -ne 0 ]; then
    echo "❌ Build failed. Last 100 lines of log:"
    tail -100 /tmp/cutlass_build.log
    exit 1
fi

echo "✅ Build complete"
echo ""

# =============================================================================
# SECTION 5: ARTIFACT VERIFICATION
# =============================================================================
echo "======================================================================"
echo "  SECTION 5: Artifact Verification"
echo "======================================================================"
echo ""

# Check Example 88 binary
if [ ! -f examples/88_hopper_fmha/88_hopper_fmha ]; then
    echo "❌ FATAL: Example 88 binary not found"
    echo "Expected: examples/88_hopper_fmha/88_hopper_fmha"
    exit 1
fi

echo "✅ Example 88 binary found:"
ls -lh examples/88_hopper_fmha/88_hopper_fmha

# Check CUTLASS profiler
if [ ! -f tools/profiler/cutlass_profiler ]; then
    echo "❌ FATAL: CUTLASS profiler not found"
    echo "Expected: tools/profiler/cutlass_profiler"
    echo "This indicates CUTLASS_ENABLE_PROFILER=ON was ignored."
    exit 1
fi

echo "✅ CUTLASS profiler found:"
ls -lh tools/profiler/cutlass_profiler
echo ""

# Verify architecture targeting
echo "Binary architecture check:"
cuobjdump -arch examples/88_hopper_fmha/88_hopper_fmha 2>&1 | grep -E "sm_90|sm_80|sm_70" | head -5

if cuobjdump -arch examples/88_hopper_fmha/88_hopper_fmha 2>&1 | grep -q "sm_90"; then
    echo "✅ Binary compiled for sm_90 (Hopper)"
else
    echo "❌ FATAL: Binary NOT compiled for sm_90"
    echo "   This will cause 'Arch conditional MMA' errors."
    cuobjdump -arch examples/88_hopper_fmha/88_hopper_fmha 2>&1 | head -20
    exit 1
fi
echo ""

# =============================================================================
# SECTION 6: CORRECTNESS VERIFICATION
# =============================================================================
echo "======================================================================"
echo "  SECTION 6: Correctness Verification"
echo "======================================================================"
echo ""
cd examples/88_hopper_fmha

echo "Running small test (B=1, H=8, Q=512, D=64)..."
./88_hopper_fmha --b=1 --h=8 --q=512 --k=512 --d=64 --iterations=5 --verify 2>&1 | tee /tmp/verify.log

if grep -q "ERROR" /tmp/verify.log; then
    echo ""
    echo "❌ FATAL: Correctness verification failed"
    cat /tmp/verify.log
    exit 1
fi

echo ""
echo "✅ Correctness verification passed"
echo ""

# =============================================================================
# SECTION 7: TARGET WORKLOAD BASELINE
# =============================================================================
echo "======================================================================"
echo "  SECTION 7: Target Workload Baseline (B=16, H=96, Q=4096, D=128)"
echo "======================================================================"
echo ""

echo "Running 100 iterations for stable measurements..."
./88_hopper_fmha --b=16 --h=96 --q=4096 --k=4096 --d=128 --iterations=100 2>&1 | tee /tmp/baseline.log

echo ""
echo "======================================================================"
echo "  BUILD AND BASELINE COMPLETE"
echo "======================================================================"
echo ""
echo "Next steps:"
echo "  1. Review baseline performance in /tmp/baseline.log"
echo "  2. Run Nsight Compute profiling"
echo "  3. Compare with PyTorch SDPA baseline"
echo ""
echo "Artifacts:"
echo "  - Binary:       /opt/cutlass/build_release/examples/88_hopper_fmha/88_hopper_fmha"
echo "  - Profiler:     /opt/cutlass/build_release/tools/profiler/cutlass_profiler"
echo "  - CMake log:    /tmp/cutlass_cmake.log"
echo "  - Build log:    /tmp/cutlass_build.log"
echo "  - Verify log:   /tmp/verify.log"
echo "  - Baseline log: /tmp/baseline.log"

