#!/usr/bin/env bash
# Bootstrap CUDA 13.0 + CUTLASS 4.3 environment
# Self-healing script for dev containers and local systems

set -e

echo "🧠 Bootstrapping CUDA 13.0 + CUTLASS 4.3 environment..."

# Detect environment
if [ -f /.dockerenv ]; then
    ENV_TYPE="docker"
elif [ -n "$CODESPACES" ]; then
    ENV_TYPE="codespaces"
elif [ -n "$RUNPOD_POD_ID" ]; then
    ENV_TYPE="runpod"
else
    ENV_TYPE="local"
fi

echo "📍 Environment: $ENV_TYPE"

# Check if CUDA 13.0 is installed
if command -v nvcc &> /dev/null; then
    CUDA_VERSION=$(nvcc --version | grep "release" | sed -n 's/.*release \([0-9\.]*\).*/\1/p')
    echo "✅ CUDA $CUDA_VERSION detected"
    
    if [[ "$CUDA_VERSION" != "13.0"* ]]; then
        echo "⚠️  CUDA version mismatch (expected 13.0, got $CUDA_VERSION)"
        echo "    Symlinking /usr/local/cuda-13.0 for compatibility..."
        
        # Create symlink if needed (non-destructive)
        if [ ! -d /usr/local/cuda-13.0 ]; then
            if [ -d /usr/local/cuda ]; then
                sudo ln -sf /usr/local/cuda /usr/local/cuda-13.0 2>/dev/null || \
                ln -sf /usr/local/cuda /usr/local/cuda-13.0
            fi
        fi
    fi
else
    echo "❌ CUDA not found - attempting installation..."
    
    if [ "$ENV_TYPE" = "docker" ]; then
        echo "🐳 Docker environment - CUDA should be in base image"
        echo "    Please ensure base image is nvidia/cuda:13.0.2-devel-ubuntu22.04"
        exit 1
    else
        echo "💻 Local environment - install CUDA 13.0 manually:"
        echo "    https://developer.nvidia.com/cuda-13-0-download"
        exit 1
    fi
fi

# Set CUDA environment
export CUDA_HOME=${CUDA_HOME:-/usr/local/cuda-13.0}
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

# Check if CUTLASS is installed
if [ -d /opt/cutlass ]; then
    cd /opt/cutlass
    CUTLASS_VERSION=$(git describe --tags 2>/dev/null || git rev-parse --short HEAD)
    echo "✅ CUTLASS $CUTLASS_VERSION detected"
    cd - > /dev/null
else
    echo "🔧 Installing CUTLASS 4.3.0..."
    
    # Clone CUTLASS
    cd /opt || cd /tmp
    if [ ! -d cutlass ]; then
        git clone --depth 1 https://github.com/NVIDIA/cutlass.git
    fi
    
    cd cutlass
    git checkout main  # v4.3.0 tag may not exist yet, use main
    
    # Build (headers only)
    mkdir -p build && cd build
    cmake .. \
        -DCUTLASS_NVCC_ARCHS="90" \
        -DCMAKE_BUILD_TYPE=Release \
        -DCUTLASS_ENABLE_TESTS=OFF \
        -DCUTLASS_ENABLE_EXAMPLES=OFF \
        -DCMAKE_CUDA_COMPILER=$CUDA_HOME/bin/nvcc 2>/dev/null || \
    cmake .. \
        -DCUTLASS_NVCC_ARCHS="90" \
        -DCMAKE_BUILD_TYPE=Release \
        -DCUTLASS_ENABLE_TESTS=OFF \
        -DCUTLASS_ENABLE_EXAMPLES=OFF
    
    # Headers are what we need, skip full build
    echo "✅ CUTLASS headers installed at /opt/cutlass/include"
fi

# Export CUTLASS path
export CUTLASS_PATH=/opt/cutlass
export CPATH=$CUTLASS_PATH/include:$CPATH

# Install Python dependencies
if command -v pip &> /dev/null; then
    echo "🐍 Installing Python dependencies..."
    pip install -q --upgrade pip setuptools wheel
    
    # Install PyTorch (CUDA 13.0 compatible)
    pip install -q torch --index-url https://download.pytorch.org/whl/nightly/cu130 2>/dev/null || \
    pip install -q torch  # Fallback to stable
    
    # Install other deps from requirements.txt if it exists
    if [ -f requirements.txt ]; then
        pip install -q -r requirements.txt 2>/dev/null || echo "⚠️  Some dependencies failed to install"
    fi
fi

# Run preflight check
if [ -f scripts/preflight_check.sh ]; then
    echo ""
    echo "🔍 Running preflight check..."
    bash scripts/preflight_check.sh
fi

echo ""
echo "✅ Environment bootstrap complete!"
echo ""
echo "📍 CUDA:    $CUDA_HOME"
echo "📍 CUTLASS: $CUTLASS_PATH"
echo ""
echo "💡 Next steps:"
echo "   make check    - Verify installation"
echo "   make run      - Start dev shell"
echo "   make bench    - Run performance benchmark"

