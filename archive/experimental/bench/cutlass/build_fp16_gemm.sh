#!/bin/bash
# Build CUTLASS FP16 GEMM for L4/Ada

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
CUTLASS_DIR="$REPO_ROOT/ext/cutlass"

echo "🔨 Building CUTLASS FP16 GEMM..."
echo "   Source: $SCRIPT_DIR/cutlass_fp16_gemm.cu"
echo "   CUTLASS: $CUTLASS_DIR"
echo ""

# Compile for sm_89 (Ada/L4)
/usr/local/cuda/bin/nvcc \
  -std=c++17 \
  -O3 \
  -gencode=arch=compute_89,code=sm_89 \
  -I "$CUTLASS_DIR/include" \
  -I "$CUTLASS_DIR/tools/util/include" \
  -o "$SCRIPT_DIR/cutlass_fp16_gemm" \
  "$SCRIPT_DIR/cutlass_fp16_gemm.cu" \
  -lcuda

echo "✅ Build complete: $SCRIPT_DIR/cutlass_fp16_gemm"
echo ""
echo "Run with: $SCRIPT_DIR/cutlass_fp16_gemm"

