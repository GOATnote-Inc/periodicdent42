#!/bin/bash
# ============================================================================
# Build Script for WGMMA Single Operation Test
# ============================================================================
# Target: H100 (sm_90a) ONLY
# ============================================================================

set -e

echo "========================================"
echo "Building WGMMA Single Operation Test"
echo "========================================"

# Configuration
CUDA_HOME=${CUDA_HOME:-/usr/local/cuda-12.4}
NVCC=${CUDA_HOME}/bin/nvcc
TARGET_ARCH=sm_90a
OUTPUT_DIR=build/bin
OUTPUT_BIN=${OUTPUT_DIR}/test_wgmma_single

# Create output directory
mkdir -p ${OUTPUT_DIR}

echo ""
echo "Configuration:"
echo "  CUDA Home: ${CUDA_HOME}"
echo "  NVCC: ${NVCC}"
echo "  Target: ${TARGET_ARCH} (H100 ONLY)"
echo "  Output: ${OUTPUT_BIN}"
echo ""

# Check if nvcc exists
if [ ! -f "${NVCC}" ]; then
    echo "❌ ERROR: nvcc not found at ${NVCC}"
    echo "Set CUDA_HOME environment variable or install CUDA 12.4+"
    exit 1
fi

# Check CUDA version
echo "CUDA Version:"
${NVCC} --version | grep "release"
echo ""

# Build flags
echo "Compiling..."
${NVCC} \
    -arch=${TARGET_ARCH} \
    -O3 \
    --use_fast_math \
    -Xptxas -v,-warn-lmem-usage \
    --std=c++17 \
    -I. \
    test_wgmma_single.cu \
    -o ${OUTPUT_BIN} \
    -L${CUDA_HOME}/lib64 \
    -Xlinker -rpath -Xlinker ${CUDA_HOME}/lib64 \
    -lineinfo

if [ $? -eq 0 ]; then
    echo ""
    echo "✅ Build successful!"
    echo "   Binary: ${OUTPUT_BIN}"
    echo ""
    echo "Register Usage:"
    echo "  (check ptxas output above for register spills)"
    echo ""
    echo "To run:"
    echo "  ./${OUTPUT_BIN}"
    echo ""
else
    echo ""
    echo "❌ Build failed!"
    exit 1
fi

