#!/bin/bash
# ============================================================================
# Gate 7 Build Script - TMA + WGMMA Kernel
# ============================================================================
# Quick build for RunPod H100 with CUDA 13.0
# ============================================================================

set -e

echo "========================================"
echo "GATE 7 - BUILD TMA KERNEL"
echo "========================================"
echo ""

# Configuration
KERNEL_SRC="src/attention_bleeding_edge_tma.cu"
OUTPUT_BIN="build/bin/attention_gate7"
BUILD_DIR="build"
ARCH="sm_90a"

# Create directories
mkdir -p build/bin build/lib build/results reports/gate7_bundle logs

# Check CUDA
if ! command -v nvcc &> /dev/null; then
    if [ -d "/usr/local/cuda/bin" ]; then
        export PATH="/usr/local/cuda/bin:$PATH"
        export LD_LIBRARY_PATH="/usr/local/cuda/lib64:$LD_LIBRARY_PATH"
    else
        echo "❌ ERROR: nvcc not found"
        exit 1
    fi
fi

CUDA_VERSION=$(nvcc --version | grep "release" | awk '{print $5}' | cut -d',' -f1)
echo "CUDA Version:  $CUDA_VERSION"

# Check GPU
if command -v nvidia-smi &> /dev/null; then
    GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
    GPU_COMPUTE=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1)
    echo "GPU:           $GPU_NAME"
    echo "Compute Cap:   $GPU_COMPUTE"
    
    if [[ ! $GPU_COMPUTE =~ ^9\. ]]; then
        echo "⚠️  WARNING: Kernel optimized for Hopper (sm_90), you have sm_$GPU_COMPUTE"
    fi
else
    echo "⚠️  WARNING: nvidia-smi not found"
fi

echo ""
echo "Building kernel..."
echo "  Source:      $KERNEL_SRC"
echo "  Output:      $OUTPUT_BIN"
echo "  Arch:        $ARCH"
echo ""

# Compile with full optimization
nvcc -arch=$ARCH \
    -O3 \
    --use_fast_math \
    -lineinfo \
    -Xptxas=-v,-warn-lmem-usage,-O3 \
    --maxrregcount=128 \
    -I. \
    -I/workspace/cutlass/include \
    -DGATE7_TMA_ENABLED \
    -DNDEBUG \
    --std=c++17 \
    -o $OUTPUT_BIN \
    $KERNEL_SRC \
    2>&1 | tee build/compile_gate7.log

# Check compilation
if [ ${PIPESTATUS[0]} -eq 0 ]; then
    echo ""
    echo "✅ Build successful"
    echo ""
    
    # Extract PTX info
    echo "Register & Memory Usage:"
    grep -E "ptxas info.*registers|ptxas info.*bytes smem" build/compile_gate7.log || echo "(No PTX info)"
    
    # Parse metrics
    REGISTERS=$(grep "registers" build/compile_gate7.log | head -1 | grep -oP '\d+(?= registers)' || echo "unknown")
    SMEM=$(grep "bytes smem" build/compile_gate7.log | head -1 | grep -oP '\d+(?= bytes)' || echo "unknown")
    
    echo ""
    echo "Registers/thread:  $REGISTERS"
    echo "Shared memory:     ${SMEM} bytes"
    
    # Occupancy estimate (H100: 65536 regs/SM, 227KB smem/SM)
    if [[ $REGISTERS =~ ^[0-9]+$ ]] && [ $REGISTERS -gt 0 ]; then
        MAX_THREADS=$((65536 / REGISTERS))
        MAX_BLOCKS=$((MAX_THREADS / 256))
        echo "Occupancy (regs):  ~$MAX_BLOCKS blocks/SM"
    fi
    
    if [[ $SMEM =~ ^[0-9]+$ ]] && [ $SMEM -gt 0 ]; then
        MAX_BLOCKS_SMEM=$((227 * 1024 / SMEM))
        echo "Occupancy (smem):  ~$MAX_BLOCKS_SMEM blocks/SM"
    fi
    
    echo ""
    echo "Next steps:"
    echo "  1. Test:    ./test_gate7_correctness.py"
    echo "  2. Bench:   ./benchmark_gate7.sh"
    echo "  3. Profile: ./profile_gate7.sh"
    echo ""
else
    echo ""
    echo "❌ Build failed"
    echo "   See: build/compile_gate7.log"
    exit 1
fi
