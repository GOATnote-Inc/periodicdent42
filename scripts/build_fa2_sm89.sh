#!/bin/bash
#
# Build FlashAttention-2 for sm_89 (Ada/L4)
#
set -euo pipefail

echo "════════════════════════════════════════════════════════════════════════════════"
echo "BUILD: FlashAttention-2 for sm_89 (Ada/L4)"
echo "════════════════════════════════════════════════════════════════════════════════"
echo ""

cd ~/periodicdent42

# Activate venv
source ~/venv/bin/activate

# Export CUDA arch
export TORCH_CUDA_ARCH_LIST="8.9"
export CUDA_HOME="/usr/local/cuda"
export PATH="$CUDA_HOME/bin:$PATH"

echo "TORCH_CUDA_ARCH_LIST: $TORCH_CUDA_ARCH_LIST"
echo "CUDA_HOME: $CUDA_HOME"
echo ""

# Install FA-2
echo "Building FlashAttention-2..."
echo "(This may take 10-15 minutes)"
echo ""

cd third_party/flash-attention

# Install dependencies
pip install -q packaging ninja

# Build
MAX_JOBS=8 python setup.py install 2>&1 | tee ~/fa2_build.log

echo ""
echo "✅ FlashAttention-2 build complete"
echo ""

# Test import
python -c "import flash_attn; print(f'FlashAttention version: {flash_attn.__version__}')" && echo "✅ Import successful" || echo "❌ Import failed"

echo ""
echo "Build log: ~/fa2_build.log"

