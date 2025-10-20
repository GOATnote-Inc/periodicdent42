#!/bin/bash
set -euo pipefail

cd ~/periodicdent42
source venv/bin/activate
export PATH=/usr/local/cuda-12.2/bin:$PATH
export TORCH_CUDA_ARCH_LIST=8.9

echo "===================================================================="
echo "Step 2: Build & PTXAS Validation (Stage-2 baseline with XOR swizzle)"
echo "===================================================================="

# Build with Stage-2 behavior (fused softmax OFF)
USE_CP_ASYNC=1 USE_WMMA_PV=1 USE_FUSED_SOFTMAX=0 \
python -m tasks.fp8_sdpa_stage_c_wmma.build 2>&1 | tee .build_step2.log

echo ""
echo "===================================================================="
echo "PTXAS Summary:"
echo "===================================================================="
grep -E "Function properties|Used [0-9]+ registers|spill|smem" .build_step2.log || echo "No PTXAS output found"

echo ""
echo "===================================================================="
echo "Build complete. Log saved to .build_step2.log"
echo "===================================================================="

