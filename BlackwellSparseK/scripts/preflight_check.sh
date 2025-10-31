#!/usr/bin/env bash
# Preflight environment integrity check
# Usage: bash scripts/preflight_check.sh

set -e

echo "🔍 Verifying CUDA + CUTLASS integrity..."

# Check CUDA 13.0
if ! nvcc --version | grep -q "V13.0"; then
    echo "❌ Wrong CUDA version (expected 13.0)"
    nvcc --version
    exit 1
fi
echo "✅ CUDA 13.0 verified"

# Check CUTLASS 4.3
if [ -d /opt/cutlass ]; then
    cd /opt/cutlass
    CUTLASS_VERSION=$(git describe --tags 2>/dev/null || git rev-parse --short HEAD)
    if ! echo "$CUTLASS_VERSION" | grep -qE "v4\.(3|1)\.0"; then
        echo "⚠️  CUTLASS version: $CUTLASS_VERSION (expected v4.3.0 or v4.1.0+)"
    else
        echo "✅ CUTLASS $CUTLASS_VERSION verified"
    fi
    cd - > /dev/null
else
    echo "❌ CUTLASS not found at /opt/cutlass"
    exit 1
fi

# Check GPU
if command -v nvidia-smi &> /dev/null; then
    echo "✅ GPU:"
    nvidia-smi --query-gpu=name,compute_cap,memory.total --format=csv,noheader
else
    echo "⚠️  nvidia-smi not found (CPU-only environment)"
fi

# Check Python dependencies
echo "✅ Python environment:"
python3 -c "import torch; print('PyTorch:', torch.__version__)"
python3 -c "import xformers; print('xFormers:', xformers.__version__)" 2>/dev/null || echo "xFormers: not installed"
python3 -c "import vllm; print('vLLM:', vllm.__version__)" 2>/dev/null || echo "vLLM: not installed"

echo ""
echo "✅ Preflight check passed - environment ready"

