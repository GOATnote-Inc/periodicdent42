#!/usr/bin/env bash
#
# Quick Test Script for Child-V2c Iteration 3
#
# Run this on the GPU instance to validate V2c-v3 (scalar Q@K^T)
#
# Expected result: 5/5 tests pass, ~2400-2500 μs latency

set -euo pipefail

cd "$(dirname "$0")"

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Testing Child-V2c Iteration 3 (Scalar Q@K^T Validation)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

# Check GPU
if ! command -v nvidia-smi &> /dev/null; then
    echo "❌ nvidia-smi not found. Are you on the GPU instance?"
    exit 1
fi

echo "🔍 GPU Info:"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
echo ""

# Check PyTorch
if ! python3 -c "import torch" 2>/dev/null; then
    echo "❌ PyTorch not found. Install: pip3 install torch"
    exit 1
fi

PYTORCH_VERSION=$(python3 -c "import torch; print(torch.__version__)")
CUDA_AVAILABLE=$(python3 -c "import torch; print(torch.cuda.is_available())")
echo "✅ PyTorch: $PYTORCH_VERSION (CUDA: $CUDA_AVAILABLE)"
echo ""

# Run tests
echo "🧪 Running Acceptance Tests..."
echo ""

python3 bench/test_v2c.py

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Testing Complete!"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "Expected Results:"
echo "  ✅ 5/5 tests pass (max_diff < 0.001)"
echo "  ✅ Latency: 2400-2500 μs (scalar baseline)"
echo "  ✅ No CUDA errors"
echo ""
echo "If passing → Proceed to Iteration 4 (WMMA + K^T)"
echo "If failing → Check V2C_ITER3_STATUS.md debug plan"
echo ""

