#!/bin/bash
# Run block size tuning on H100

set -e

RUNPOD_IP="154.57.34.90"
RUNPOD_PORT="14727"

echo "========================================="
echo "STAGE 2 REDUX: BLOCK SIZE TUNING"
echo "========================================="
echo ""

# Deploy tuning script
echo "[1/2] Deploying block size tuner..."
scp -P ${RUNPOD_PORT} -o StrictHostKeyChecking=no \
    flashcore/fast/tune_block_sizes_h100.py \
    root@${RUNPOD_IP}:/workspace/flashcore_llama/flashcore/fast/
echo "✅ Deployed"
echo ""

# Run tuning
echo "[2/2] Running block size sweep on H100..."
echo "This will test 9 configurations (32x32, 32x64, 32x128, ...)"
echo ""

ssh -p ${RUNPOD_PORT} -o StrictHostKeyChecking=no root@${RUNPOD_IP} << 'ENDSSH'
cd /workspace/flashcore_llama
export PYTHONPATH=/workspace/flashcore_llama:$PYTHONPATH

echo "GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader)"
echo ""

python3 flashcore/fast/tune_block_sizes_h100.py

ENDSSH

echo ""
echo "========================================="
echo "BLOCK TUNING COMPLETE"
echo "========================================="

