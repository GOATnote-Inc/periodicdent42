#!/bin/bash
set -euo pipefail

cd ~/periodicdent42
source venv/bin/activate
export PATH=/usr/local/cuda-12.2/bin:$PATH
export TORCH_CUDA_ARCH_LIST=8.9

echo "===================================================================="
echo "Step 2: Correctness Validation (6 test combinations)"
echo "===================================================================="

# Run correctness tests with Stage-2 behavior (XOR swizzle enabled)
USE_CP_ASYNC=1 USE_WMMA_PV=1 USE_FUSED_SOFTMAX=0 \
python -m tasks.fp8_sdpa_stage_c_wmma.runner --shapes small,mission --seeds 0,1,2 \
2>&1 | tee .corr_s2_step2.log

echo ""
echo "===================================================================="
echo "Summary:"
echo "===================================================================="
grep -E "PASS|FAIL|max_abs_err|avg_abs_err" .corr_s2_step2.log | tail -20 || echo "Tests output"

echo ""
echo "===================================================================="
echo "Correctness validation complete. Log saved to .corr_s2_step2.log"
echo "===================================================================="

