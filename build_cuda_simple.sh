#!/bin/bash
# Simple nvcc build (no CMake required)

set -e

echo "========================================="
echo "BUILDING PHASE 1 KERNEL (nvcc)"
echo "========================================="
echo ""

# Setup CUDA environment
export PATH=/usr/local/cuda-12.4/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-12.4/lib64:$LD_LIBRARY_PATH

# Check CUDA
if ! command -v nvcc &> /dev/null; then
    echo "❌ ERROR: nvcc not found"
    exit 1
fi

echo "CUDA: $(nvcc --version | grep release)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo ""

# Create output directory
mkdir -p build/bin build/lib

# Compile kernel + test
echo "[1/2] Compiling WMMA kernel..."
nvcc -arch=sm_90 -O3 --use_fast_math \
    -Xptxas -v,-warn-lmem-usage \
    -I. \
    flashcore/fast/attention_cuda_wmma.cu \
    flashcore/cuda/test_hopper_kernel.cu \
    -o build/bin/test_hopper \
    2>&1 | tee build/compile.log | grep -E "(ptxas|error|warning)" || true

if [ ${PIPESTATUS[0]} -eq 0 ]; then
    echo "✅ Compilation successful"
else
    echo "❌ Compilation failed"
    exit 1
fi

echo ""
echo "[2/2] Running test..."
./build/bin/test_hopper

echo ""
echo "========================================="
echo "BUILD COMPLETE"
echo "========================================="

