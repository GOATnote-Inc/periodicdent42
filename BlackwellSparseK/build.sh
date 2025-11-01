#!/bin/bash
# Build BlackwellSparseK PyTorch Extension

set -e  # Exit on error

echo "=================================="
echo "BlackwellSparseK Build Script"
echo "=================================="
echo ""

# Check CUDA
if ! command -v nvcc &> /dev/null; then
    echo "❌ Error: nvcc not found"
    echo "   Please install CUDA Toolkit 13.0.2+"
    echo "   Or set CUDA_HOME: export CUDA_HOME=/usr/local/cuda-13.0"
    exit 1
fi

echo "✅ nvcc found: $(nvcc --version | grep release)"
echo ""

# Check PyTorch
if ! python3 -c "import torch" &> /dev/null; then
    echo "❌ Error: PyTorch not found"
    echo "   Install: pip install torch"
    exit 1
fi

TORCH_VERSION=$(python3 -c "import torch; print(torch.__version__)")
CUDA_AVAILABLE=$(python3 -c "import torch; print(torch.cuda.is_available())")

echo "✅ PyTorch found: $TORCH_VERSION"
echo "   CUDA available: $CUDA_AVAILABLE"
echo ""

if [ "$CUDA_AVAILABLE" != "True" ]; then
    echo "⚠️  Warning: PyTorch CUDA not available"
    echo "   Install CUDA-enabled PyTorch from https://pytorch.org"
fi

# Set CUDA_HOME if not set
if [ -z "$CUDA_HOME" ]; then
    export CUDA_HOME=/usr/local/cuda-13.0
    echo "Setting CUDA_HOME=$CUDA_HOME"
fi

# Detect GPU architecture
if command -v nvidia-smi &> /dev/null; then
    GPU=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
    echo "🎯 GPU detected: $GPU"
    
    # Set architecture based on GPU
    if [[ $GPU == *"L4"* ]]; then
        export TORCH_CUDA_ARCH_LIST="8.9"
        echo "   Using sm_89 (Ada)"
    elif [[ $GPU == *"H100"* ]]; then
        export TORCH_CUDA_ARCH_LIST="9.0"
        echo "   Using sm_90a (Hopper)"
    elif [[ $GPU == *"A100"* ]]; then
        export TORCH_CUDA_ARCH_LIST="8.0"
        echo "   Using sm_80 (Ampere)"
    else
        echo "   Using default architecture"
    fi
fi

echo ""

# Clean previous build
echo "🧹 Cleaning previous build..."
rm -rf build/ dist/ *.egg-info
rm -f python/*.so

# Build
echo "🔨 Building C++ extension..."
python3 setup.py build_ext --inplace

if [ $? -eq 0 ]; then
    echo ""
    echo "=================================="
    echo "✅ Build successful!"
    echo "=================================="
    echo ""
    echo "Test installation:"
    echo "  python3 -c 'import blackwellsparsek; print(blackwellsparsek.__version__)'"
    echo ""
    echo "Run quickstart:"
    echo "  python3 examples/quickstart.py"
    echo ""
else
    echo ""
    echo "=================================="
    echo "❌ Build failed"
    echo "=================================="
    echo ""
    echo "Troubleshooting:"
    echo "  1. Check CUDA_HOME is set correctly"
    echo "  2. Ensure nvcc version matches PyTorch CUDA version"
    echo "  3. Try: VERBOSE=1 ./build.sh"
    exit 1
fi

