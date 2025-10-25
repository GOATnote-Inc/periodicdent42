#!/bin/bash
# Debug build for FlashCore cp.async kernel
# Use with compute-sanitizer for precise error location

set -e

KERNEL_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/kernels" && pwd)"
BUILD_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/build"

mkdir -p "$BUILD_DIR"

echo "Building FlashCore cp.async (DEBUG MODE)..."
echo "Kernel: $KERNEL_DIR/flashcore_fused_wmma_cpasync.cu"

nvcc -O2 -g -G -lineinfo \
     -arch=sm_89 \
     --use_fast_math \
     -Xptxas -v \
     -Xptxas -warn-spills \
     -std=c++17 \
     --expt-relaxed-constexpr \
     --expt-extended-lambda \
     -I"$KERNEL_DIR" \
     "$KERNEL_DIR/flashcore_fused_wmma_cpasync.cu" \
     -o "$BUILD_DIR/flashcore_cpasync_dbg"

echo "✅ Debug build complete: $BUILD_DIR/flashcore_cpasync_dbg"
echo ""
echo "Run with compute-sanitizer:"
echo "  compute-sanitizer --tool memcheck $BUILD_DIR/flashcore_cpasync_dbg"
echo "  compute-sanitizer --tool racecheck $BUILD_DIR/flashcore_cpasync_dbg"
echo "  compute-sanitizer --tool synccheck $BUILD_DIR/flashcore_cpasync_dbg"

