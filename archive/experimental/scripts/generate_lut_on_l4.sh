#!/bin/bash
set -euo pipefail

cd ~/periodicdent42
source venv/bin/activate
export PATH=/usr/local/cuda-12.2/bin:$PATH
export TORCH_CUDA_ARCH_LIST=8.9

echo "Generating WMMA 16x16 accumulator LUT..."
python scripts/generate_wmma_lut.py

echo ""
echo "Verifying generated header..."
if [ -f cudadent42/bench/kernels/wmma16x16_accum_lut.h ]; then
    echo "✅ Header generated successfully"
    wc -l cudadent42/bench/kernels/wmma16x16_accum_lut.h
    head -10 cudadent42/bench/kernels/wmma16x16_accum_lut.h
else
    echo "❌ Header generation failed"
    exit 1
fi

