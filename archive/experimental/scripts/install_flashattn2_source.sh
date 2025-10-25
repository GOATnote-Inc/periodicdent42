#!/bin/bash
# Install FlashAttention-2 from source for L4/Ada (sm_89)

set -e

echo "=================================================="
echo "Installing FlashAttention-2 from Source (sm_89)"
echo "=================================================="
echo ""

# Load CUDA
export PATH="/usr/local/cuda/bin:$PATH"
export LD_LIBRARY_PATH="/usr/local/cuda/lib64:$LD_LIBRARY_PATH"

# Build settings
export TORCH_CUDA_ARCH_LIST="8.9"
export MAX_JOBS=8

echo "📦 Environment:"
echo "   CUDA: $(nvcc --version | grep release | awk '{print $6}')"
echo "   TORCH_CUDA_ARCH_LIST: $TORCH_CUDA_ARCH_LIST"
echo "   MAX_JOBS: $MAX_JOBS"
echo ""

# Dependencies
echo "📚 Installing dependencies..."
pip install -q packaging ninja einops
echo "✅ Dependencies OK"
echo ""

# Check if already installed
if python3 -c "import flash_attn; print(f'✅ FA2 {flash_attn.__version__} already installed')" 2>/dev/null; then
    echo "Skipping installation (already present)"
    exit 0
fi

# Clone and build
echo "🔨 Cloning FlashAttention-2..."
cd /tmp
rm -rf flash-attention
git clone https://github.com/Dao-AILab/flash-attention.git
cd flash-attention

echo ""
echo "🔧 Building (this takes 5-7 minutes)..."
pip install -v . --no-build-isolation 2>&1 | grep -E "(Building|Installing|Successfully|ERROR)" || true

echo ""
echo "=================================================="
echo "✅ Installation Complete"
echo "=================================================="

# Verify
python3 -c "import flash_attn; print(f'FlashAttention-2 version: {flash_attn.__version__}')"

cd ~
rm -rf /tmp/flash-attention

echo "✅ Verified and cleaned up"

