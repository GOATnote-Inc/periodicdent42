#!/usr/bin/env bash
#
# Development Environment Setup Script
#
# Sets up optimized CUDA build environment for L4 (SM_89)
# with Ninja, ccache, and environment variables.
#
# Usage:
#   bash scripts/setup_dev_env.sh
#
# Author: GOATnote Autonomous Research Lab Initiative
# Date: 2025-10-14

set -euo pipefail

echo "=============================================================="
echo "Setting up CUDA Development Environment"
echo "=============================================================="
echo ""

# Detect OS
if [[ "$OSTYPE" == "linux-gnu"* ]]; then
    OS="linux"
elif [[ "$OSTYPE" == "darwin"* ]]; then
    OS="macos"
else
    echo "❌ Unsupported OS: $OSTYPE"
    exit 1
fi

echo "Detected OS: $OS"
echo ""

# Install Python dependencies
echo "📦 Installing Python dependencies..."
pip install --upgrade pip
pip install ninja ccache

echo "   ✅ Python dependencies installed"
echo ""

# Detect GPU
if command -v nvidia-smi &> /dev/null; then
    echo "🔍 Detecting GPU..."
    GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
    echo "   GPU: $GPU_NAME"
    
    # Detect CUDA architecture
    if [[ "$GPU_NAME" == *"L4"* ]]; then
        CUDA_ARCH="8.9"
        echo "   Architecture: SM_89 (Ada Lovelace)"
    elif [[ "$GPU_NAME" == *"A100"* ]]; then
        CUDA_ARCH="8.0"
        echo "   Architecture: SM_80 (Ampere)"
    elif [[ "$GPU_NAME" == *"H100"* ]]; then
        CUDA_ARCH="9.0"
        echo "   Architecture: SM_90 (Hopper)"
    else
        CUDA_ARCH="8.0"
        echo "   Architecture: SM_80 (default)"
    fi
else
    echo "⚠️  nvidia-smi not found - GPU detection skipped"
    CUDA_ARCH="8.9"
    echo "   Defaulting to: SM_89 (L4)"
fi
echo ""

# Setup environment variables
echo "🔧 Setting environment variables..."

# Detect shell config file
if [[ "$OS" == "macos" ]]; then
    if [[ -f "$HOME/.zshrc" ]]; then
        SHELL_RC="$HOME/.zshrc"
    elif [[ -f "$HOME/.bashrc" ]]; then
        SHELL_RC="$HOME/.bashrc"
    else
        SHELL_RC="$HOME/.zshrc"
        touch "$SHELL_RC"
    fi
else
    SHELL_RC="$HOME/.bashrc"
fi

echo "   Config file: $SHELL_RC"

# Check if already configured
if grep -q "TORCH_CUDA_ARCH_LIST" "$SHELL_RC" 2>/dev/null; then
    echo "   ⚠️  Environment variables already configured in $SHELL_RC"
    echo "   Skipping to avoid duplicates"
else
    # Append environment variables
    cat >> "$SHELL_RC" << EOF

# CUDA Development Environment (added by setup_dev_env.sh)
export TORCH_CUDA_ARCH_LIST="$CUDA_ARCH"
export MAX_JOBS=\$(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 4)
export CUDAFLAGS="--use_fast_math -O3"
export CCACHE_DIR=\$HOME/.ccache
export CCACHE_MAXSIZE=5G
export TORCH_CUDA_BUILD_CACHE=\$HOME/.torch_cuda_cache
export NVCC_THREADS=\$(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 4)
export PATH=\$HOME/.local/bin:\$PATH
EOF
    
    echo "   ✅ Environment variables added to $SHELL_RC"
fi

# Export for current session
export TORCH_CUDA_ARCH_LIST="$CUDA_ARCH"
export MAX_JOBS=$(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 4)
export CUDAFLAGS="--use_fast_math -O3"
export CCACHE_DIR=$HOME/.ccache
export CCACHE_MAXSIZE=5G
export TORCH_CUDA_BUILD_CACHE=$HOME/.torch_cuda_cache
export NVCC_THREADS=$(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 4)
export PATH=$HOME/.local/bin:$PATH

echo "   TORCH_CUDA_ARCH_LIST=$CUDA_ARCH"
echo "   MAX_JOBS=$MAX_JOBS"
echo ""

# Setup ccache
echo "📦 Configuring ccache..."
mkdir -p "$HOME/.ccache"
ccache --max-size=5G > /dev/null 2>&1 || true
echo "   ✅ ccache configured (max size: 5GB)"
echo ""

# Setup build cache
echo "📁 Setting up build cache..."
mkdir -p "$HOME/.torch_cuda_cache"
echo "   ✅ Build cache directory created"
echo ""

# Verify installation
echo "🔍 Verifying installation..."

# Check Ninja
if command -v ninja &> /dev/null; then
    NINJA_VERSION=$(ninja --version)
    echo "   ✅ Ninja: $NINJA_VERSION"
else
    echo "   ❌ Ninja not found"
fi

# Check ccache
if command -v ccache &> /dev/null; then
    echo "   ✅ ccache: installed"
else
    echo "   ❌ ccache not found"
fi

# Check CUDA
if command -v nvcc &> /dev/null; then
    CUDA_VERSION=$(nvcc --version | grep "release" | awk '{print $5}' | tr -d ',')
    echo "   ✅ CUDA: $CUDA_VERSION"
else
    echo "   ⚠️  nvcc not found (CUDA toolkit may not be installed)"
fi

# Check PyTorch
if python3 -c "import torch" 2>/dev/null; then
    PYTORCH_VERSION=$(python3 -c "import torch; print(torch.__version__)")
    CUDA_AVAILABLE=$(python3 -c "import torch; print(torch.cuda.is_available())")
    echo "   ✅ PyTorch: $PYTORCH_VERSION (CUDA available: $CUDA_AVAILABLE)"
else
    echo "   ⚠️  PyTorch not installed"
fi

echo ""
echo "=============================================================="
echo "✅ Setup Complete!"
echo "=============================================================="
echo ""
echo "Next steps:"
echo "  1. Reload shell: source $SHELL_RC"
echo "  2. Verify: python3 -c 'import torch; print(torch.cuda.is_available())'"
echo "  3. Build kernel: python3 bench/_build.py --kernel fa_s512"
echo ""
echo "Documentation:"
echo "  - Dev environment: docs/dev_env.md"
echo "  - Performance guardrails: docs/perf_guardrails.md"
echo ""

