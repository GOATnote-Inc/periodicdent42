#!/bin/bash
# Build script for Phase B.1 Test 1: Minimal cuBLAS GEMM

set -e

echo "Building Phase B.1 Test 1: Minimal cuBLAS GEMM..."

nvcc -O3 \
  -std=c++17 \
  -gencode=arch=compute_89,code=sm_89 \
  -lcublas \
  bench/test_cublas_minimal.cu \
  -o bench/test_cublas_minimal

echo "✅ Build complete: bench/test_cublas_minimal"
echo ""
echo "Run with: ./bench/test_cublas_minimal"

