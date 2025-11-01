#!/bin/bash
# ONE-COMMAND benchmark - just provide H100 IP and port
# Usage: ./RUN_NOW.sh root@YOUR_IP YOUR_PORT

set -e

H100_HOST=${1:-root@154.57.34.90}
H100_PORT=${2:-23673}

echo "🚀 RUNNING BENCHMARK NOW (no more docs)"
echo "Target: $H100_HOST:$H100_PORT"
echo ""

# Deploy and run in one shot
ssh -p $H100_PORT $H100_HOST 'bash -s' << 'ENDSSH'
cd /tmp
rm -rf benchmark_test
mkdir -p benchmark_test
cd benchmark_test

# Minimal PyTorch sparse vs dense comparison (no custom kernel needed)
cat > test.py << 'EOF'
import torch
import time

print("🔬 QUICK HONEST TEST: PyTorch Sparse vs Dense")
print("="*60)

M, N, K = 8192, 8192, 8192
device = "cuda"

# Create 78% sparse matrix (similar to attention)
torch.manual_seed(42)
dense_A = torch.randn(M, K, dtype=torch.float16, device=device) * 0.1
mask = torch.rand(M, K, device=device) > 0.78  # 78% sparse
dense_A = dense_A * mask

# Dense baseline
print("\n1️⃣  Dense matmul (cuBLAS)...")
B = torch.randn(K, N, dtype=torch.float16, device=device) * 0.1

# Warmup
for _ in range(10):
    C = torch.mm(dense_A, B)
torch.cuda.synchronize()

# Time dense
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
print(f"   Time: {dense_time:.3f} ms")
print(f"   TFLOPS: {dense_tflops:.2f}")

# Sparse (CSR)
print("\n2️⃣  Sparse CSR matmul (cuSPARSE backend)...")
A_sparse = dense_A.to_sparse_csr()
print(f"   Sparsity: {1 - A_sparse._nnz() / (M*K):.1%}")

# Warmup
for _ in range(10):
    C = torch.sparse.mm(A_sparse, B)
torch.cuda.synchronize()

# Time sparse
start.record()
for _ in range(100):
    C_sparse = torch.sparse.mm(A_sparse, B)
end.record()
torch.cuda.synchronize()
sparse_time = start.elapsed_time(end) / 100

sparse_ops = 2 * A_sparse._nnz() * N
sparse_tflops = (sparse_ops / (sparse_time * 1e-3)) / 1e12
print(f"   Time: {sparse_time:.3f} ms")
print(f"   TFLOPS: {sparse_tflops:.2f}")

# Compare
print("\n📊 RESULTS:")
print("="*60)
print(f"Dense (cuBLAS):     {dense_tflops:>7.2f} TFLOPS  {dense_time:>6.2f} ms")
print(f"Sparse (cuSPARSE):  {sparse_tflops:>7.2f} TFLOPS  {sparse_time:>6.2f} ms")
print(f"Speedup:            {dense_time/sparse_time:>7.2f}x")
print("="*60)

# Correctness
diff = torch.abs(C_dense - C_sparse).max()
print(f"\n✅ Max diff: {diff:.6f} (should be near 0)")

print("\n💡 INTERPRETATION:")
if sparse_tflops > dense_tflops * 1.5:
    print("✅ Sparse wins significantly - sparsity helps")
elif sparse_tflops > dense_tflops:
    print("⚠️  Sparse slightly better - marginal gain")
else:
    print("❌ Dense is faster - sparse overhead not worth it")

print("\nCritical question: How does OUR kernel compare?")
print("Need to test: Custom kernel vs this sparse baseline")
EOF

python3 test.py

ENDSSH

echo ""
echo "✅ BENCHMARK COMPLETE"
echo ""
echo "This shows PyTorch sparse (cuSPARSE) performance."
echo "Next: Compare custom kernel to these numbers."

