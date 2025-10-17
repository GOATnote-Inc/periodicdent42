#!/bin/bash
# Build script for Phase B.1 Test 2: FlashAttention Q@K^T Tile

set -e

# Add CUDA to PATH
export PATH="/usr/local/cuda/bin:$PATH"
export LD_LIBRARY_PATH="/usr/local/cuda/lib64:$LD_LIBRARY_PATH"

echo "Building Phase B.1 Test 2: FlashAttention Q@K^T Tile..."

nvcc -O3 \
  -std=c++17 \
  -gencode=arch=compute_89,code=sm_89 \
  -lcublas \
  bench/test_cublas_qkt_tile.cu \
  -o bench/test_cublas_qkt_tile

echo "✅ Build complete: bench/test_cublas_qkt_tile"
echo ""
echo "Run with: ./bench/test_cublas_qkt_tile"

