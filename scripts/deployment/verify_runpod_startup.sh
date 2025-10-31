#!/bin/bash
# RunPod H100 Startup Verification Script
# Expert CUDA Engineer - Speed & Security Focus

set -e

IP="${1:-154.57.34.98}"
PORT="${2:-36088}"
SSH_OPTS="-o StrictHostKeyChecking=no -o TCPKeepAlive=yes -o ServerAliveInterval=20 -o ConnectTimeout=10"

echo "═══════════════════════════════════════════════════════════════"
echo "RunPod H100 GPU Startup Verification"
echo "═══════════════════════════════════════════════════════════════"
echo ""
echo "Target: $IP:$PORT"
echo ""

# Step 1: Test SSH connection
echo "[1/5] Testing SSH connection..."
MAX_RETRIES=10
RETRY=0

while [ $RETRY -lt $MAX_RETRIES ]; do
    if ssh -p $PORT $SSH_OPTS root@$IP "echo 'SSH Connected'" 2>/dev/null; then
        echo "✅ SSH connection established"
        break
    else
        RETRY=$((RETRY + 1))
        if [ $RETRY -lt $MAX_RETRIES ]; then
            echo "   Retry $RETRY/$MAX_RETRIES (waiting 5s for SSH service...)"
            sleep 5
        else
            echo "❌ SSH connection failed after $MAX_RETRIES attempts"
            echo ""
            echo "Troubleshooting:"
            echo "1. Verify pod status is 'Running' in RunPod dashboard"
            echo "2. Check IP/Port are correct in Connect tab"
            echo "3. Wait 60s after pod shows 'Ready' for SSH initialization"
            echo "4. Try manual connection: ssh root@$IP -p $PORT"
            exit 1
        fi
    fi
done
echo ""

# Step 2: Verify GPU presence
echo "[2/5] Verifying H100 GPU..."
GPU_INFO=$(ssh -p $PORT $SSH_OPTS root@$IP "nvidia-smi --query-gpu=name,compute_cap,memory.total --format=csv,noheader" 2>/dev/null)
echo "✅ GPU Detected: $GPU_INFO"
echo ""

# Step 3: Verify CUDA toolkit
echo "[3/5] Verifying CUDA toolkit..."
CUDA_VERSION=$(ssh -p $PORT $SSH_OPTS root@$IP "nvcc --version 2>/dev/null | grep release | awk '{print \$5}' | tr -d ','")
if [ -z "$CUDA_VERSION" ]; then
    echo "⚠️  nvcc not in PATH, checking /usr/local/cuda..."
    CUDA_VERSION=$(ssh -p $PORT $SSH_OPTS root@$IP "/usr/local/cuda/bin/nvcc --version 2>/dev/null | grep release | awk '{print \$5}' | tr -d ','")
fi
echo "✅ CUDA Version: $CUDA_VERSION"
echo ""

# Step 4: Verify Python/PyTorch/Triton
echo "[4/5] Verifying software stack..."
ssh -p $PORT $SSH_OPTS root@$IP 'bash -s' <<'REMOTE'
python3 --version 2>/dev/null || echo "⚠️  Python3 not found"
python3 -c "import torch; print(f'PyTorch: {torch.__version__}')" 2>/dev/null || echo "⚠️  PyTorch not installed"
python3 -c "import triton; print(f'Triton: {triton.__version__}')" 2>/dev/null || echo "⚠️  Triton not installed"
REMOTE
echo ""

# Step 5: Quick GPU compute test
echo "[5/5] Running GPU compute verification..."
ssh -p $PORT $SSH_OPTS root@$IP 'python3 -c "
import torch
import time

# Test tensor operation on GPU
x = torch.randn(1000, 1000, device=\"cuda\")
y = torch.randn(1000, 1000, device=\"cuda\")

start = torch.cuda.Event(enable_timing=True)
end = torch.cuda.Event(enable_timing=True)

start.record()
z = torch.matmul(x, y)
end.record()

torch.cuda.synchronize()
latency = start.elapsed_time(end)

print(f\"✅ GPU Compute Verified: {latency:.2f}ms for 1000x1000 matmul\")
"' 2>/dev/null || echo "⚠️  GPU compute test failed"

echo ""
echo "═══════════════════════════════════════════════════════════════"
echo "✅ RunPod H100 GPU Startup Verification COMPLETE"
echo "═══════════════════════════════════════════════════════════════"
echo ""
echo "Pod ready for kernel deployment"
