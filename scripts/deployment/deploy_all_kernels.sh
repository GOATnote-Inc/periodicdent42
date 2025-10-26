#!/bin/bash
# Deploy all 4 novel kernels to H100 for validation
set -e

IP="154.57.34.90"
PORT="23673"
SSH_OPTS="-o StrictHostKeyChecking=no"

echo "═══════════════════════════════════════════════════════════════"
echo "DEPLOYING 4 NOVEL KERNELS TO H100"
echo "═══════════════════════════════════════════════════════════════"
echo ""

# Create workspace
echo "[1/5] Creating workspace..."
ssh -p $PORT $SSH_OPTS root@$IP "mkdir -p /workspace/flashcore_validation"
echo "✅ Workspace ready"
echo ""

# Upload kernels
echo "[2/5] Uploading kernels..."
scp -P $PORT $SSH_OPTS \
    flashcore/fast/attention_multihead.py \
    flashcore/fast/attention_fp8.py \
    flashcore/fast/attention_longcontext.py \
    flashcore/fast/attention_production.py \
    root@$IP:/workspace/flashcore_validation/
echo "✅ 4 kernels uploaded"
echo ""

# Quick validation test
echo "[3/5] Running quick validation..."
ssh -p $PORT $SSH_OPTS root@$IP 'bash -s' <<'REMOTE'
cd /workspace/flashcore_validation
export PATH=/usr/local/cuda/bin:$PATH

python3 << 'PYEOF'
import torch
print(f"GPU: {torch.cuda.get_device_name()}")
print(f"Compute: sm_{torch.cuda.get_device_capability()[0]}{torch.cuda.get_device_capability()[1]}")
print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
print(f"PyTorch: {torch.__version__}")
import triton
print(f"Triton: {triton.__version__}")
print("")
print("✅ Environment validated")
PYEOF
REMOTE

echo ""
echo "═══════════════════════════════════════════════════════════════"
echo "✅ DEPLOYMENT COMPLETE - READY FOR VALIDATION"
echo "═══════════════════════════════════════════════════════════════"
