#!/bin/bash
# Build FlashCore Hopper kernel on H100

set -e

echo "========================================"
echo "BUILDING FLASHCORE HOPPER KERNEL"
echo "========================================"
echo ""

# Check CUDA version
if ! command -v nvcc &> /dev/null; then
    echo "❌ ERROR: nvcc not found. Install CUDA Toolkit 12.0+"
    exit 1
fi

CUDA_VERSION=$(nvcc --version | grep "release" | awk '{print $5}' | cut -d',' -f1)
echo "CUDA Version: $CUDA_VERSION"

# Check GPU
if ! command -v nvidia-smi &> /dev/null; then
    echo "⚠️  WARNING: nvidia-smi not found. Cannot detect GPU."
else
    GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
    GPU_SM=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1 2>/dev/null || echo "unknown")
    echo "GPU: $GPU_NAME"
    echo "Compute Capability: $GPU_SM"
    
    # Check if Hopper
    if [[ $GPU_SM != 9.* ]]; then
        echo ""
        echo "⚠️  WARNING: This kernel is optimized for Hopper (sm_90)"
        echo "            Your GPU is sm_$GPU_SM"
        echo "            It may not run or will run slowly."
        echo ""
    fi
fi

echo ""
echo "[1/3] Creating build directory..."
mkdir -p build
cd build

echo "[2/3] Running CMake..."
cmake ../flashcore/cuda \
    -DCMAKE_CUDA_ARCHITECTURES=90 \
    -DCMAKE_BUILD_TYPE=Release

echo "[3/3] Compiling kernel..."
make -j$(nproc)

echo ""
echo "========================================"
echo "BUILD COMPLETE"
echo "========================================"
echo ""
echo "Artifacts:"
echo "  Library: build/lib/libflashcore_hopper.so"
echo "  Test:    build/bin/test_hopper"
echo ""
echo "Run test with:"
echo "  ./build/bin/test_hopper"
echo ""

