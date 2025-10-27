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

# Compile kernel + test (Phase 3B: cuBLASLt Sparse GEMM)
echo "[1/2] Compiling Phase 3B cuBLASLt kernel (sm_90a)..."

# Actual cuBLAS library path on H100 (verified!)
CUBLAS_LIB_PATH="/usr/local/cuda-12.4/targets/x86_64-linux/lib"

echo "   Using cuBLAS libraries from: $CUBLAS_LIB_PATH"

# Set library path for runtime
export LD_LIBRARY_PATH=$CUBLAS_LIB_PATH:${LD_LIBRARY_PATH:-}

nvcc -arch=sm_90a -O3 --use_fast_math \
    -Xptxas -v,-warn-lmem-usage \
    --std=c++17 \
    -DKERNEL_PHASE=4 \
    -I. \
    flashcore/fast/attention_hopper_minimal.cu \
    flashcore/fast/attention_cublaslt.cu \
    flashcore/cuda/test_hopper_kernel.cu \
    -o build/bin/test_hopper \
    ${CUBLAS_LIB_PATH}/libcublas.so.12 \
    ${CUBLAS_LIB_PATH}/libcublasLt.so.12 \
    -Xlinker -rpath=${CUBLAS_LIB_PATH} \
    2>&1 | tee build/compile.log | grep -E "(ptxas|error|warning|libcublas)" || true

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

