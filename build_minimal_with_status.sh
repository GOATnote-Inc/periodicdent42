#!/bin/bash
# Minimal build script with status reporting
# Created: Session N+1 to fix frozen SSH issue

set -e
cd ~/periodicdent42/cudadent42

echo "🧹 [1/5] Cleaning previous build..."
python3 -c "import sys; sys.path.insert(0, 'python'); from setuptools import setup" 2>/dev/null || true
rm -rf build/ dist/ *.egg-info flashmoe_science.*.so flashmoe_science/_C.*.so
echo "✅ Clean complete"

echo ""
echo "📝 [2/5] Creating minimal setup.py..."
cat > setup_minimal.py << 'SETUPEOF'
from setuptools import setup
from torch.utils import cpp_extension

sources = [
    'python/flashmoe_science/csrc/bindings.cpp',
    'python/flashmoe_science/csrc/flash_attention_wrapper.cpp',
    'python/flashmoe_science/csrc/flash_attention_science.cu',
]

setup(
    name='flashmoe-science',
    ext_modules=[
        cpp_extension.CUDAExtension(
            'flashmoe_science._C',
            sources,
            extra_compile_args={
                'cxx': ['-O3'],
                'nvcc': [
                    '-O3',
                    '-gencode=arch=compute_89,code=sm_89',
                    '--use_fast_math',
                    '-Xptxas=-v',  # Show register/smem usage
                ]
            }
        )
    ],
    cmdclass={'build_ext': cpp_extension.BuildExtension}
)
SETUPEOF
echo "✅ Setup script created"

echo ""
echo "🔨 [3/5] Building extension (this may take 1-2 minutes)..."
echo "    (Compiling 3 source files for SM89/L4)"
python3 setup_minimal.py build_ext --inplace 2>&1 | tee build.log | tail -50

echo ""
echo "🔍 [4/5] Checking for .so file..."
if [ -f "flashmoe_science/_C.cpython-310-x86_64-linux-gnu.so" ]; then
    echo "✅ Extension built successfully"
    ls -lh flashmoe_science/_C.*.so
else
    echo "❌ Extension file not found"
    exit 1
fi

echo ""
echo "🧪 [5/5] Testing import..."
python3 -c "import sys; sys.path.insert(0, 'python'); import flashmoe_science._C; print('✅ SUCCESS: Extension imported')"

echo ""
echo "🎉 Build complete!"

