#!/bin/bash
# Install FlashAttention-2 for L4/Ada (sm_89)
# Optimized for production performance comparison

set -e

echo "=================================================="
echo "Installing FlashAttention-2 for L4/Ada (sm_89)"
echo "=================================================="
echo ""

# Load CUDA into PATH
if [ -d "/usr/local/cuda/bin" ]; then
    export PATH="/usr/local/cuda/bin:$PATH"
    export LD_LIBRARY_PATH="/usr/local/cuda/lib64:$LD_LIBRARY_PATH"
fi

# Environment setup
export TORCH_CUDA_ARCH_LIST="8.9"  # Target Ada/L4 specifically
export MAX_JOBS=8                  # Parallel compilation
export FLASH_ATTENTION_FORCE_BUILD=TRUE

echo "📦 Environment:"
echo "   TORCH_CUDA_ARCH_LIST: $TORCH_CUDA_ARCH_LIST"
echo "   MAX_JOBS: $MAX_JOBS"
echo ""

# Check CUDA
echo "🔍 Checking CUDA installation..."
nvcc --version || { echo "❌ CUDA not found"; exit 1; }
echo ""

# Install dependencies
echo "📚 Installing dependencies..."
pip install -q packaging ninja einops
echo "✅ Dependencies installed"
echo ""

# Install FlashAttention-2
echo "🔨 Building FlashAttention-2 (this will take 3-5 minutes)..."
echo "   Target: sm_89 (Ada/L4)"
echo "   Version: Latest (2.5.x)"
echo ""

pip install flash-attn --no-build-isolation 2>&1 | \
    grep -E "(Processing|Collecting|Building|Installing|Successfully)" || true

echo ""
echo "=================================================="
echo "✅ FlashAttention-2 Installation Complete"
echo "=================================================="

# Verify installation
python3 -c "import flash_attn; print(f'FlashAttention-2 version: {flash_attn.__version__}')" || \
    { echo "❌ Import failed"; exit 1; }

echo "✅ Import successful"
echo ""
echo "Next: Run benchmark comparison"

