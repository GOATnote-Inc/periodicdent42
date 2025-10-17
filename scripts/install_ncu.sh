#!/bin/bash
#
# Install Nsight Compute (ncu) on L4 instance
#
set -e

echo "════════════════════════════════════════════════════════════════════════════════"
echo "INSTALL: Nsight Compute (ncu)"
echo "════════════════════════════════════════════════════════════════════════════════"
echo ""

# NCU comes with CUDA toolkit
CUDA_PATH="/usr/local/cuda"

if [ -f "$CUDA_PATH/bin/ncu" ]; then
    echo "✅ NCU already installed at $CUDA_PATH/bin/ncu"
    $CUDA_PATH/bin/ncu --version
    
    # Add to PATH if not already there
    if ! echo "$PATH" | grep -q "$CUDA_PATH/bin"; then
        echo "export PATH=$CUDA_PATH/bin:\$PATH" >> ~/.bashrc
        echo "✅ Added $CUDA_PATH/bin to PATH in ~/.bashrc"
    fi
else
    echo "⚠️  NCU not found in $CUDA_PATH/bin"
    echo ""
    echo "Checking alternative locations..."
    
    # Check common CUDA locations
    for cuda_dir in /usr/local/cuda-12.1 /usr/local/cuda-12 /opt/cuda; do
        if [ -f "$cuda_dir/bin/ncu" ]; then
            echo "✅ Found NCU at $cuda_dir/bin/ncu"
            export PATH="$cuda_dir/bin:$PATH"
            echo "export PATH=$cuda_dir/bin:\$PATH" >> ~/.bashrc
            $cuda_dir/bin/ncu --version
            exit 0
        fi
    done
    
    echo "❌ NCU not found. Install CUDA toolkit with Nsight Compute:"
    echo "   https://developer.nvidia.com/nsight-compute"
    exit 1
fi

echo ""
echo "Current PATH:"
echo "$PATH" | tr ':' '\n' | grep -i cuda || echo "  (no CUDA paths found)"
echo ""

# Test NCU
echo "Testing NCU..."
ncu --version || {
    echo "⚠️  NCU not in PATH. Reload shell:"
    echo "   source ~/.bashrc"
}

echo ""
echo "✅ NCU installation check complete"

