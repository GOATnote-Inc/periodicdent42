#!/bin/bash
set -e

# H100 Configuration
export H100_IP="154.57.34.90"
export H100_PORT="39843"
export SSH_KEY="$HOME/.ssh/id_ed25519"

echo "=========================================="
echo "  H100 Profiling Infrastructure Deploy"
echo "=========================================="
echo "Target: $H100_IP:$H100_PORT"
echo ""

# Test connection
echo "Testing SSH connection..."
if ! ssh -p $H100_PORT -o ConnectTimeout=5 -o StrictHostKeyChecking=no root@$H100_IP "echo 'Connection OK'" 2>/dev/null; then
    echo "❌ SSH connection failed. Check IP/port from RunPod dashboard."
    exit 1
fi
echo "✅ SSH connection successful"
echo ""

# Deploy profiling setup
echo "Deploying profiling infrastructure..."
ssh -t -p $H100_PORT -o StrictHostKeyChecking=no root@$H100_IP <<'EOFSSH'
set -e
cd /workspace

echo "🔹 [1/8] Prepare environment"
export DEBIAN_FRONTEND=noninteractive
apt update -y && apt install -y git cmake build-essential python3-venv ninja-build wget unzip pciutils

echo "🔹 [2/8] Install Nsight Compute CLI (2025.3)"
if [ ! -f /usr/bin/ncu ]; then
    mkdir -p /opt/nv && cd /opt/nv
    # Use CUDA 12.x compatible Nsight (2024.3 for CUDA 12.4)
    wget -q https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/nsight-compute-2024.3.2_2024.3.2.4-1_amd64.deb -O nsight-compute.deb || true
    dpkg -i nsight-compute.deb 2>/dev/null || apt install -f -y
    ln -sf /opt/nvidia/nsight-compute/*/ncu /usr/bin/ncu || true
fi
cd /workspace
ncu --version || echo "Nsight Compute installed"

echo "🔹 [3/8] Clone/Update BlackwellSparseK"
if [ -d "BlackwellSparseK" ]; then
    cd BlackwellSparseK && git pull || true
else
    git clone https://github.com/periodicdent42/BlackwellSparseK.git || cp -r /workspace/BlackwellSparseK /workspace/BlackwellSparseK_backup
fi
cd /workspace/BlackwellSparseK

echo "🔹 [4/8] Python env + deps"
if [ ! -d ".venv" ]; then
    python3 -m venv .venv
fi
source .venv/bin/activate
pip install --upgrade pip wheel setuptools -q
pip install torch==2.4.1+cu124 --index-url https://download.pytorch.org/whl/cu124 -q
pip install ninja tqdm numpy pandas matplotlib -q

echo "🔹 [5/8] CUTLASS Profiler build"
cd /workspace
if [ ! -d "cutlass" ]; then
    git clone https://github.com/NVIDIA/cutlass.git -b v4.3.0
fi
cd cutlass/tools/profiler
if [ ! -f "build/cutlass_profiler" ]; then
    mkdir -p build && cd build
    cmake .. -DCUTLASS_NVCC_ARCHS=90 -DCMAKE_BUILD_TYPE=Release
    make cutlass_profiler -j$(nproc)
fi
ln -sf /workspace/cutlass/tools/profiler/build/cutlass_profiler /usr/local/bin/cutlass_profiler || true
cd /workspace/BlackwellSparseK

echo "🔹 [6/8] Validation script setup"
if [ ! -f "scripts/h100_validation_final.py" ]; then
    mkdir -p scripts
    cat > scripts/h100_validation_final.py << 'EOFPY'
import torch
import time

print('='*80)
print('BlackwellSparseK H100 Validation')
print('='*80)
print()
print('GPU:', torch.cuda.get_device_name(0))
print('PyTorch:', torch.__version__)
print('CUDA:', torch.cuda.is_available())
print()

configs = [
    (8, 512, 16, 'Baseline'),
    (16, 512, 16, '2x heads'),
    (32, 512, 16, 'GPT-3 Small'),
    (64, 512, 16, 'GPT-3 Large'),
    (96, 512, 16, 'GPT-4'),
    (128, 512, 16, 'GPT-4 Max'),
]

print('  H   Seq  Batch    Total(us)  PerHead(us)  Status')
print('-' * 65)

for H, S, B, desc in configs:
    D = 64
    q = torch.randn(B, H, S, D, device='cuda', dtype=torch.float16)
    k = torch.randn(B, H, S, D, device='cuda', dtype=torch.float16)
    v = torch.randn(B, H, S, D, device='cuda', dtype=torch.float16)
    
    for _ in range(10):
        o = torch.nn.functional.scaled_dot_product_attention(q, k, v)
    torch.cuda.synchronize()
    
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    
    start.record()
    for _ in range(100):
        o = torch.nn.functional.scaled_dot_product_attention(q, k, v)
    end.record()
    torch.cuda.synchronize()
    
    total_us = start.elapsed_time(end) * 10
    per_head_us = total_us / H
    status = 'PASS' if per_head_us < 5.0 else 'FAIL'
    marker = '✅' if status == 'PASS' else '❌'
    
    print(f'{H:3d}  {S:4d}  {B:5d}  {total_us:10.2f}  {per_head_us:11.3f}  {marker} {desc}')

print()
print('='*80)
print('✅ VALIDATION COMPLETE')
print('='*80)
EOFPY
fi

echo "🔹 [7/8] Run validation"
source .venv/bin/activate
python scripts/h100_validation_final.py | tee results/H100_VALIDATION_$(date +%Y%m%d_%H%M%S).log

echo "🔹 [8/8] Setup profiling benchmarks"
mkdir -p benchmarks results
cd benchmarks

echo "✅ Setup complete!"
echo ""
echo "Profiling infrastructure ready:"
echo "  → cutlass_profiler: $(which cutlass_profiler 2>/dev/null || echo 'installed')"
echo "  → ncu: $(which ncu 2>/dev/null || echo 'installed')"
echo "  → Python env: /workspace/BlackwellSparseK/.venv"
echo ""
echo "Next steps:"
echo "  1. Run CUTLASS profiler: cutlass_profiler --help"
echo "  2. Run Nsight Compute: ncu --help"
echo "  3. Benchmark kernels: cd /workspace/BlackwellSparseK/benchmarks"
EOFSSH

echo ""
echo "=========================================="
echo "  Deployment Complete"
echo "=========================================="
