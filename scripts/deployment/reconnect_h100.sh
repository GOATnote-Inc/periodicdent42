#!/bin/bash
# Persistent H100 Reconnection - GPU Stays Live
# Expert CUDA Architect - Speed & Security

IP="154.57.34.98"
PORT="36088"
SSH_OPTS="-o StrictHostKeyChecking=no -o TCPKeepAlive=yes -o ServerAliveInterval=20 -o ConnectTimeout=5"

echo "═══════════════════════════════════════════════════════════════"
echo "H100 GPU Reconnection (Persistent Mode)"
echo "═══════════════════════════════════════════════════════════════"
echo ""
echo "Target: $IP:$PORT"
echo "Strategy: Keep trying until SSH responds (GPU stays live)"
echo ""

ATTEMPT=1
while true; do
    echo -n "[Attempt $ATTEMPT] Testing connection... "
    
    if ssh -p $PORT $SSH_OPTS root@$IP "echo 'Connected' && nvidia-smi --query-gpu=name --format=csv,noheader" 2>/dev/null; then
        echo ""
        echo "═══════════════════════════════════════════════════════════════"
        echo "✅ H100 ONLINE - SSH ESTABLISHED"
        echo "═══════════════════════════════════════════════════════════════"
        
        # Quick verification
        echo ""
        echo "Running full verification..."
        ssh -p $PORT $SSH_OPTS root@$IP 'bash -s' <<'REMOTE'
set -e

echo "[GPU Info]"
nvidia-smi --query-gpu=name,compute_cap,memory.total,memory.free --format=csv,noheader

echo ""
echo "[CUDA]"
/usr/local/cuda/bin/nvcc --version 2>/dev/null | grep release || echo "nvcc check failed"

echo ""
echo "[Python Stack]"
python3 --version
python3 -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.version.cuda}')"
python3 -c "import triton; print(f'Triton: {triton.__version__}')"

echo ""
echo "[Quick Compute Test]"
python3 -c "
import torch
x = torch.randn(1000, 1000, device='cuda')
y = x @ x
torch.cuda.synchronize()
print(f'✅ GPU compute verified: {y.shape}')
"
REMOTE
        
        echo ""
        echo "═══════════════════════════════════════════════════════════════"
        echo "✅ H100 READY FOR KERNEL DEPLOYMENT"
        echo "═══════════════════════════════════════════════════════════════"
        exit 0
    else
        echo "Failed (SSH not ready)"
        ATTEMPT=$((ATTEMPT + 1))
        
        # Exponential backoff up to 10 seconds
        if [ $ATTEMPT -le 5 ]; then
            WAIT=2
        elif [ $ATTEMPT -le 10 ]; then
            WAIT=5
        else
            WAIT=10
        fi
        
        sleep $WAIT
    fi
done
