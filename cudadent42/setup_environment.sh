#!/bin/bash
# Pattern 9: Environment Validation Script
# Created: October 12, 2025
# Purpose: Validate GPU environment before building CUDA extensions
# Time: ~5 minutes

set -e  # Exit on any error

echo "🔍 Pattern 9: Environment Validation (5 minutes)"
echo "=================================================="
echo ""

# 1. Check PyTorch (30 sec)
echo "⏱️  Step 1/5: Checking PyTorch..."
python3 -c "
import torch
expected = '2.2.1+cu121'
actual = torch.__version__
if actual != expected:
    print(f'⚠️  PyTorch version mismatch: {actual} != {expected}')
    print('Installing PyTorch 2.2.1+cu121...')
    exit(1)
else:
    print(f'✅ PyTorch {actual}')
" || {
    pip3 install --user torch==2.2.1 --index-url https://download.pytorch.org/whl/cu121
    echo "✅ PyTorch installed"
}
echo ""

# 2. Check NumPy (30 sec)
echo "⏱️  Step 2/5: Checking NumPy..."
python3 -c "
import numpy
major = int(numpy.__version__.split('.')[0])
if major >= 2:
    print(f'⚠️  NumPy {numpy.__version__} is incompatible with PyTorch 2.2.1')
    print('Installing NumPy 1.x...')
    exit(1)
else:
    print(f'✅ NumPy {numpy.__version__}')
" || {
    pip3 install --user 'numpy<2'
    echo "✅ NumPy 1.x installed"
}
echo ""

# 3. Check CUDA (30 sec)
echo "⏱️  Step 3/5: Checking CUDA..."
python3 -c "
import torch
if not torch.cuda.is_available():
    print('❌ CUDA not available')
    exit(1)
device_name = torch.cuda.get_device_name(0)
cuda_version = torch.version.cuda
print(f'✅ CUDA {cuda_version}: {device_name}')
"
echo ""

# 4. Get PyTorch library path
echo "⏱️  Step 4/5: Setting up library paths..."
TORCH_LIB=$(python3 -c "import torch; print(torch.__path__[0])")/lib
echo "✅ PyTorch lib: $TORCH_LIB"
export LD_LIBRARY_PATH=$TORCH_LIB:$LD_LIBRARY_PATH
echo ""

# 5. Check if CUDA extension exists and loads
echo "⏱️  Step 5/5: Checking CUDA extension..."
if [ -f "flashmoe_science/_C.cpython-310-x86_64-linux-gnu.so" ]; then
    echo "📦 CUDA extension found, testing load..."
    python3 -c "
import flashmoe_science._C as fa
print('✅ Extension loaded')
print('📋 Exported functions:', dir(fa))
" && {
        echo ""
        echo "🎉 Environment validation COMPLETE!"
        echo "   All systems operational."
        echo ""
        echo "Next steps:"
        echo "  1. Run benchmark: python3 benches/bench_correctness_and_speed.py"
        echo "  2. Or rebuild: python3 setup_native_fixed.py build_ext --inplace"
    } || {
        echo "⚠️  Extension found but failed to load"
        echo "    Rebuilding recommended..."
        echo ""
        echo "Next steps:"
        echo "  1. Clean: python3 setup_native_fixed.py clean"
        echo "  2. Build: python3 setup_native_fixed.py build_ext --inplace"
        echo "  3. Test: python3 benches/bench_correctness_and_speed.py"
    }
else
    echo "📦 No CUDA extension found"
    echo "   First-time setup required."
    echo ""
    echo "Next steps:"
    echo "  1. Build: python3 setup_native_fixed.py build_ext --inplace"
    echo "  2. Test: python3 benches/bench_correctness_and_speed.py"
fi

echo ""
echo "💡 Tip: Source this script to preserve LD_LIBRARY_PATH:"
echo "   source ./setup_environment.sh"
echo ""
