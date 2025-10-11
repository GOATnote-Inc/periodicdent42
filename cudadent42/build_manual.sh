#!/usr/bin/env bash
set -euo pipefail

# Manual build script for FP16 + BF16 CUDA extensions
# This proves both extensions can be compiled and linked together

echo "╔═══════════════════════════════════════════════════════════════════════╗"
echo "║  Manual Build: FP16 + BF16 Extensions (Production Grade)             ║"
echo "╚═══════════════════════════════════════════════════════════════════════╝"
echo ""

# Detect architecture
if command -v nvidia-smi &> /dev/null; then
    GPU_SM=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -n1 | tr -d '.' || echo "89")
    echo "✅ Detected GPU: SM_${GPU_SM}"
else
    GPU_SM="89"
    echo "⚠️  No GPU detected, defaulting to SM_${GPU_SM} (L4)"
fi

# Determine BF16 support
if [[ "$GPU_SM" -ge 80 ]]; then
    HAS_BF16=1
    echo "✅ BF16 support: ENABLED (SM${GPU_SM} >= SM80)"
else
    HAS_BF16=0
    echo "❌ BF16 support: DISABLED (SM${GPU_SM} < SM80)"
fi

# Get PyTorch paths
TORCH_DIR=$(python3 -c "import torch; import os; print(os.path.dirname(torch.__file__))" 2>/dev/null || echo "/usr/local/lib/python3.10/dist-packages/torch")
TORCH_INCLUDE="${TORCH_DIR}/include"
TORCH_LIB="${TORCH_DIR}/lib"

# Get Python include
PYTHON_INCLUDE=$(python3 -c "import sysconfig; print(sysconfig.get_path('include'))" 2>/dev/null || echo "/usr/include/python3.10")

# Get ABI flag from PyTorch
ABI_FLAG=$(python3 -c "import torch; print('1' if torch._C._GLIBCXX_USE_CXX11_ABI else '0')" 2>/dev/null || echo "1")
echo "🔗 ABI flag: _GLIBCXX_USE_CXX11_ABI=${ABI_FLAG}"

# Build directories
mkdir -p build/manual
cd build/manual

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "STEP 1: Compile FP16 CUDA kernel"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

/usr/local/cuda/bin/nvcc \
    -c ../../python/flashmoe_science/csrc/flash_attention_fp16_sm75.cu \
    -o flash_attention_fp16_sm75.o \
    -I../../python/flashmoe_science/csrc \
    -I"${TORCH_INCLUDE}" \
    -I"${TORCH_INCLUDE}/torch/csrc/api/include" \
    -I/usr/local/cuda/include \
    -O3 \
    --use_fast_math \
    -lineinfo \
    --expt-relaxed-constexpr \
    --expt-extended-lambda \
    -gencode=arch=compute_${GPU_SM},code=sm_${GPU_SM} \
    -gencode=arch=compute_${GPU_SM},code=compute_${GPU_SM} \
    -Xcompiler=-fno-strict-aliasing \
    -Xcompiler=-fPIC \
    -Xcompiler=-fno-omit-frame-pointer \
    -DCUDA_NO_BFLOAT16 \
    -D__CUDA_NO_BFLOAT16_OPERATORS__ \
    -DFLASHMOE_HAS_BF16=${HAS_BF16} \
    -D_GLIBCXX_USE_CXX11_ABI=${ABI_FLAG} \
    -std=c++17

echo "✅ FP16 kernel compiled: flash_attention_fp16_sm75.o"

if [[ "$HAS_BF16" -eq 1 ]]; then
    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "STEP 2: Compile BF16 CUDA kernel (SM80+)"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    
    /usr/local/cuda/bin/nvcc \
        -c ../../python/flashmoe_science/csrc/flash_attention_bf16_sm80.cu \
        -o flash_attention_bf16_sm80.o \
        -I../../python/flashmoe_science/csrc \
        -I"${TORCH_INCLUDE}" \
        -I"${TORCH_INCLUDE}/torch/csrc/api/include" \
        -I/usr/local/cuda/include \
        -O3 \
        --use_fast_math \
        -lineinfo \
        --expt-relaxed-constexpr \
        --expt-extended-lambda \
        -gencode=arch=compute_${GPU_SM},code=sm_${GPU_SM} \
        -gencode=arch=compute_${GPU_SM},code=compute_${GPU_SM} \
        -Xcompiler=-fno-strict-aliasing \
        -Xcompiler=-fPIC \
        -Xcompiler=-fno-omit-frame-pointer \
        -DFLASHMOE_HAS_BF16=${HAS_BF16} \
        -D_GLIBCXX_USE_CXX11_ABI=${ABI_FLAG} \
        -std=c++17
    
    echo "✅ BF16 kernel compiled: flash_attention_bf16_sm80.o"
    BF16_OBJ="flash_attention_bf16_sm80.o"
else
    echo ""
    echo "⏭️  Skipping BF16 compilation (SM${GPU_SM} < SM80)"
    BF16_OBJ=""
fi

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "STEP 3: Compile Python bindings (C++)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

g++ \
    -c ../../python/flashmoe_science/csrc/bindings_new.cpp \
    -o bindings_new.o \
    -I../../python/flashmoe_science/csrc \
    -I"${TORCH_INCLUDE}" \
    -I"${TORCH_INCLUDE}/torch/csrc/api/include" \
    -I"${PYTHON_INCLUDE}" \
    -I/usr/local/cuda/include \
    -O3 \
    -fPIC \
    -std=c++17 \
    -DCUDA_NO_BFLOAT16 \
    -D__CUDA_NO_BFLOAT16_OPERATORS__ \
    -DFLASHMOE_HAS_BF16=${HAS_BF16} \
    -D_GLIBCXX_USE_CXX11_ABI=${ABI_FLAG} \
    -DTORCH_API_INCLUDE_EXTENSION_H \
    -DTORCH_EXTENSION_NAME=_C

echo "✅ Bindings compiled: bindings_new.o"

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "STEP 4: Link into shared library"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

g++ -shared \
    flash_attention_fp16_sm75.o \
    ${BF16_OBJ} \
    bindings_new.o \
    -o _C.so \
    -L"${TORCH_LIB}" \
    -L/usr/local/cuda/lib64 \
    -lc10 \
    -lc10_cuda \
    -ltorch \
    -ltorch_cpu \
    -ltorch_python \
    -ltorch_cuda \
    -lcudart \
    -Wl,-rpath,"${TORCH_LIB}" \
    -Wl,-rpath,/usr/local/cuda/lib64

echo "✅ Shared library linked: _C.so"

# Copy to final location
mkdir -p ../../flashmoe_science
cp _C.so ../../flashmoe_science/_C.so

cd ../..

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "STEP 5: Verify symbols"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

echo ""
echo "Checking for C-linkage wrappers:"
nm flashmoe_science/_C.so | grep flash_attention_forward || echo "⚠️  No symbols found (check nm)"

echo ""
echo "Checking for Python binding:"
nm flashmoe_science/_C.so | grep PyInit || echo "⚠️  No PyInit symbol"

echo ""
echo "╔═══════════════════════════════════════════════════════════════════════╗"
echo "║  ✅ BUILD COMPLETE!                                                   ║"
echo "╚═══════════════════════════════════════════════════════════════════════╝"
echo ""
echo "Output: flashmoe_science/_C.so"
echo "Size: $(du -h flashmoe_science/_C.so | cut -f1)"
echo ""
echo "Next: python3 -c 'import sys; sys.path.insert(0, \".\"); import flashmoe_science._C'"

