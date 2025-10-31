#!/bin/bash
# ============================================================================
# Build Script for Corrected WGMMA Implementation
# ============================================================================
# Target: H100 (sm_90a) ONLY
# Expected Performance: 2.8-3.5 TFLOPS on single 64×64×16 WGMMA
# ============================================================================

set -e  # Exit on error

echo "========================================"
echo "  Building WGMMA Corrected Test"
echo "========================================"
echo ""

# Configuration
CUDA_ARCH="90a"
BUILD_DIR="build"
BIN_DIR="${BUILD_DIR}/bin"
OBJ_DIR="${BUILD_DIR}/obj"

# Compiler flags
NVCC_FLAGS=(
    -arch=sm_${CUDA_ARCH}           # H100 architecture
    -O3                              # Maximum optimization
    --use_fast_math                  # Fast math operations
    --expt-relaxed-constexpr        # Relaxed constexpr for device functions
    -lineinfo                        # Line info for profiling
    --ptxas-options=-v              # Verbose PTX assembly output
    --ptxas-options=-warn-lmem-usage # Warn about local memory usage (register spills)
    --ptxas-options=-warn-spills    # Warn about register spills
    -Xcompiler=-Wall                # Enable all warnings
    -Xcompiler=-Wextra              # Extra warnings
    -std=c++17                      # C++17 standard
)

# Debug build option
if [[ "${1}" == "--debug" ]]; then
    echo "🔍 Building in DEBUG mode"
    NVCC_FLAGS+=(
        -g                          # Debug symbols
        -G                          # Device debug symbols
        -O0                         # No optimization
    )
else
    echo "🚀 Building in RELEASE mode"
fi

# Create directories
mkdir -p "${BIN_DIR}"
mkdir -p "${OBJ_DIR}"

echo ""
echo "Compilation Settings:"
echo "  Architecture: sm_${CUDA_ARCH}"
echo "  Optimization: ${NVCC_FLAGS[*]}"
echo ""

# Compile test program
echo "📦 Compiling corrected WGMMA test..."

nvcc "${NVCC_FLAGS[@]}" \
    -I. \
    test_wgmma_single_corrected.cu \
    -o "${BIN_DIR}/test_wgmma_corrected" \
    2>&1 | tee "${BUILD_DIR}/compile.log"

COMPILE_STATUS=${PIPESTATUS[0]}

echo ""
if [ ${COMPILE_STATUS} -eq 0 ]; then
    echo "✅ Build successful!"
    echo ""
    echo "Binary: ${BIN_DIR}/test_wgmma_corrected"
    
    # Extract register usage from compile log
    echo ""
    echo "📊 Resource Usage (from PTX compilation):"
    echo "=========================================="
    grep -E "registers|bytes stack frame|bytes spill" "${BUILD_DIR}/compile.log" || true
    echo "=========================================="
    
    # Check for warnings
    echo ""
    if grep -qi "warning" "${BUILD_DIR}/compile.log"; then
        echo "⚠️  Warnings detected (see ${BUILD_DIR}/compile.log)"
        grep -i "warning" "${BUILD_DIR}/compile.log" | head -10
    else
        echo "✅ No warnings"
    fi
    
    # Check for register spills
    echo ""
    if grep -qi "spill" "${BUILD_DIR}/compile.log"; then
        echo "🔴 WARNING: Register spills detected!"
        echo "   This will significantly impact performance."
        echo "   Consider reducing register usage or increasing occupancy limits."
        grep -i "spill" "${BUILD_DIR}/compile.log"
    else
        echo "✅ No register spills (optimal)"
    fi
    
    echo ""
    echo "🚀 Ready to run:"
    echo "   ./${BIN_DIR}/test_wgmma_corrected"
    echo ""
    echo "📊 Profile with Nsight Compute:"
    echo "   ncu --set full ./${BIN_DIR}/test_wgmma_corrected"
    echo ""
    
else
    echo "❌ Build failed! See ${BUILD_DIR}/compile.log for details"
    echo ""
    tail -20 "${BUILD_DIR}/compile.log"
    exit 1
fi

# ============================================================================
# Expected Output:
# ============================================================================
# ✅ Build successful!
#
# Binary: build/bin/test_wgmma_corrected
#
# 📊 Resource Usage:
# ==========================================
#   test_wgmma_single_corrected: 48 registers, 0 bytes stack, 0 bytes spill
# ==========================================
#
# ✅ No warnings
# ✅ No register spills (optimal)
#
# Target Register Usage: 45-55 registers per thread (no spills)
# Target Performance: 2.8-3.5 TFLOPS
# ============================================================================

