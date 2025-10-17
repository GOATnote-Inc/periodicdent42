#!/bin/bash
# bench/cutlass/build_cutlass.sh
set -e

export PATH="/usr/local/cuda/bin:$PATH"

REPO_ROOT=$(cd "$(dirname "$0")/../.." && pwd)
CUTLASS_INC="${REPO_ROOT}/ext/cutlass/include"

if [ ! -d "$CUTLASS_INC" ]; then
    echo "❌ CUTLASS not found at $CUTLASS_INC"
    echo "Run: git submodule update --init --recursive"
    exit 1
fi

nvcc -O3 -std=c++17 \
  -I "$CUTLASS_INC" \
  -gencode=arch=compute_89,code=sm_89 \
  bench/cutlass/cutlass_attn_qkt.cu -o bench/cutlass/cutlass_attn_qkt \
  -lcuda

echo "✅ CUTLASS Q@K^T compiled"

