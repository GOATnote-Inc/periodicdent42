#!/usr/bin/env bash
set -euo pipefail

echo "🔍 Verifying CUDA Development Environment"
echo "=========================================="
echo ""

# Check environment variables
echo "📦 Environment Variables:"
echo "  TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST:-NOT SET}"
echo "  MAX_JOBS=${MAX_JOBS:-NOT SET}"
echo "  CUDAFLAGS=${CUDAFLAGS:-NOT SET}"
echo "  CCACHE_DIR=${CCACHE_DIR:-NOT SET}"
echo ""

# Check tools
echo "🔧 Tools:"
which ninja && ninja --version | head -1 || echo "  ❌ ninja not found"
which ccache && ccache --version | head -1 || echo "  ⚠️  ccache not found (optional)"
which ncu && ncu --version | head -1 || echo "  ⚠️  ncu not found (optional, on GPU only)"
echo ""

# Check PyTorch
echo "🐍 PyTorch:"
python3 -c "import torch; print(f'  PyTorch {torch.__version__}'); print(f'  CUDA {torch.version.cuda}'); print(f'  Ninja: {torch.utils.cpp_extension.is_ninja_available()}')" 2>/dev/null || echo "  ❌ PyTorch/CUDA not available"
echo ""

# Check GPU (if available)
echo "🎮 GPU:"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader 2>/dev/null || echo "  ⚠️  nvidia-smi not available (run on GPU)"
echo ""

echo "✅ Verification complete"

