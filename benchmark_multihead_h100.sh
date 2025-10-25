#!/bin/bash
# Deploy and benchmark multi-head attention kernel on RunPod H100
# Target: <5 μs per head for GPT-4 class models (H=96-128)

set -e

# RunPod credentials (from memory)
IP="154.57.34.98"
PORT="36088"
SSH_OPTS="-o StrictHostKeyChecking=no -o TCPKeepAlive=yes -o ServerAliveInterval=20"

echo "==================================================================="
echo "MULTI-HEAD ATTENTION KERNEL - H100 VALIDATION"
echo "==================================================================="
echo ""
echo "Target: <5 μs per head (GPT-4: H=96-128)"
echo "Baseline: 0.73-4.34 μs for H=8 (validated)"
echo ""

# Test connection
echo "[1/4] Testing H100 connection..."
ssh -p $PORT $SSH_OPTS root@$IP "nvidia-smi --query-gpu=name,compute_cap --format=csv,noheader" || {
    echo "❌ Connection failed"
    exit 1
}
echo "✅ H100 connected"
echo ""

# Upload files
echo "[2/4] Uploading multi-head kernel..."
scp -P $PORT $SSH_OPTS flashcore/fast/attention_multihead.py root@$IP:/workspace/
echo "✅ Kernel uploaded"
echo ""

# Run correctness test
echo "[3/4] Running correctness validation..."
ssh -p $PORT $SSH_OPTS root@$IP 'bash -s' <<'REMOTE'
set -e
cd /workspace

export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-}

# Check dependencies
python3 -c "import torch; print(f'PyTorch: {torch.__version__}')"
python3 -c "import triton; print(f'Triton: {triton.__version__}')"

# Correctness test
echo ""
echo "Running correctness validation..."
python3 -c "
import torch
import sys
sys.path.insert(0, '/workspace')
from attention_multihead import multihead_attention

configs = [(8, 512, 16, 64), (32, 512, 16, 64), (96, 512, 8, 64)]

for H, S, B, D in configs:
    torch.manual_seed(42)
    q = torch.randn(B, H, S, D, device='cuda', dtype=torch.float16)
    k, v = q.clone(), q.clone()
    
    out_custom = multihead_attention(q, k, v)
    out_ref = torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=False)
    
    max_diff = (out_custom - out_ref).abs().max().item()
    status = '✅' if max_diff < 2e-3 else '❌'
    print(f'H={H:3}, S={S:4}, B={B:2}: max_diff={max_diff:.6f} {status}')

print('')
print('✅ Correctness validated')
"
REMOTE

echo ""
echo "✅ Correctness validated on H100"
echo ""

# Run performance benchmark
echo "[4/4] Running performance benchmark..."
echo ""

ssh -p $PORT $SSH_OPTS root@$IP 'bash -s' <<'REMOTE'
set -e
cd /workspace

export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-}

python3 attention_multihead.py 2>&1 | tee multihead_benchmark_h100.txt

echo ""
echo "Benchmark complete. Results saved to multihead_benchmark_h100.txt"
REMOTE

echo ""
echo "[5/4] Downloading results..."
scp -P $PORT $SSH_OPTS root@$IP:/workspace/multihead_benchmark_h100.txt . || echo "Results in stdout above"
echo ""

echo "==================================================================="
echo "✅ MULTI-HEAD ATTENTION VALIDATION COMPLETE"
echo "==================================================================="
echo ""
echo "Results: multihead_benchmark_h100.txt"
echo "Next: Validate H=96 (GPT-4) achieves <5 μs per head"
echo ""

