#!/bin/bash
# === DEPLOY_TO_L4.sh ===
# One-click script to deploy and test on L4 with CUDA 13.0.2
# Run this in Cloud Console SSH terminal

set -e

banner() { echo -e "\n\033[1;36m=== $1 ===\033[0m"; }

banner "1. Check GPU & CUDA"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv
echo ""

banner "2. Setup CUDA 13.0.2 paths"
export CUDA_HOME=/usr/local/cuda-13.0
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

if nvcc --version 2>/dev/null | grep -q "release 13"; then
    echo "✅ CUDA 13 found"
    nvcc --version | grep release
else
    echo "⚠️  CUDA 13 not found - need to install"
    echo "Run the setup script from earlier"
    exit 1
fi

banner "3. Create test directory"
mkdir -p ~/cuda_test
cd ~/cuda_test

banner "4. Download baseline comparison script"
cat > honest_benchmark.py << 'PYEOF'
import torch
import time

print("="*70)
print("🔬 HONEST BENCHMARK: PyTorch Sparse vs Dense")
print("="*70)

M, N, K = 8192, 8192, 8192
device = "cuda"

print(f"\nGPU: {torch.cuda.get_device_name(0)}")
print(f"Matrix: {M}×{K} @ {K}×{N}, FP16, 78% sparse\n")

# Create 78% sparse matrix
torch.manual_seed(42)
dense_A = torch.randn(M, K, dtype=torch.float16, device=device) * 0.1
mask = torch.rand(M, K, device=device) > 0.78
dense_A = dense_A * mask
B = torch.randn(K, N, dtype=torch.float16, device=device) * 0.1

# Dense baseline
print("1️⃣  Dense cuBLAS...")
for _ in range(10):
    C = torch.mm(dense_A, B)
torch.cuda.synchronize()

start = torch.cuda.Event(enable_timing=True)
end = torch.cuda.Event(enable_timing=True)
start.record()
for _ in range(100):
    C_dense = torch.mm(dense_A, B)
end.record()
torch.cuda.synchronize()
dense_time = start.elapsed_time(end) / 100
dense_ops = 2 * M * N * K
dense_tflops = (dense_ops / (dense_time * 1e-3)) / 1e12
print(f"   {dense_tflops:.2f} TFLOPS ({dense_time:.3f} ms)\n")

# Sparse baseline
print("2️⃣  PyTorch Sparse (cuSPARSE)...")
A_sparse = dense_A.to_sparse_csr()
actual_sparsity = 1 - A_sparse._nnz() / (M*K)
print(f"   Actual sparsity: {actual_sparsity:.1%}")

for _ in range(10):
    C = torch.sparse.mm(A_sparse, B)
torch.cuda.synchronize()

start.record()
for _ in range(100):
    C_sparse = torch.sparse.mm(A_sparse, B)
end.record()
torch.cuda.synchronize()
sparse_time = start.elapsed_time(end) / 100
sparse_ops = 2 * A_sparse._nnz() * N
sparse_tflops = (sparse_ops / (sparse_time * 1e-3)) / 1e12
print(f"   {sparse_tflops:.2f} TFLOPS ({sparse_time:.3f} ms)\n")

print("="*70)
print("📊 RESULTS:")
print("="*70)
print(f"Dense (cuBLAS):       {dense_tflops:>8.2f} TFLOPS  {dense_time:>7.3f} ms")
print(f"Sparse (cuSPARSE):    {sparse_tflops:>8.2f} TFLOPS  {sparse_time:>7.3f} ms")
print(f"Speedup:              {dense_time/sparse_time:>8.2f}x")
print("="*70)
print()
print("🎯 Custom kernel target: 610 TFLOPS")
print(f"   Would be {610/sparse_tflops:.0f}x faster than cuSPARSE")
print(f"   Would be {610/dense_tflops:.1%} of dense performance")
PYEOF

banner "5. Run baseline benchmark"
python3 honest_benchmark.py

banner "6. Results saved"
echo "✅ Baseline measurements complete on L4"
echo ""
echo "Next: Deploy and test custom sparse kernel"

