#!/bin/bash
# Deploy and test on current H100 (CUDA 12.8)

echo "📦 Packaging source..."
tar czf kernel_src.tar.gz src/ benchmarks/

echo "🚀 Deploying to H100..."
scp -P 15608 kernel_src.tar.gz root@157.66.254.40:/tmp/

echo "🔨 Building on H100..."
ssh -p 15608 root@157.66.254.40 'bash -s' << 'REMOTE'
cd /tmp
tar xzf kernel_src.tar.gz
cd src

# Check CUDA
python3 -c "import torch; print(f'CUDA: {torch.version.cuda}')"

# Try PyTorch extension build (uses CUDA 12.8)
cat > test_sparse_torch.py << 'PY'
import torch
import time
from torch.utils.cpp_extension import load_inline

# Your sparse kernel source
cuda_src = open('sparse_h100_async.cu').read()

# Build as PyTorch extension
module = load_inline(
    name='sparse_h100',
    cpp_sources=[''],
    cuda_sources=[cuda_src],
    extra_cuda_cflags=['-O3', '--use_fast_math', '-arch=sm_90a'],
    verbose=True
)

print("✅ Kernel compiled!")
print("🧪 Running benchmark...")

# TODO: Call your kernel here
PY

python3 test_sparse_torch.py
REMOTE

echo "✅ Done"
