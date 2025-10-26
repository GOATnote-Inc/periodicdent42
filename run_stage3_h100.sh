#!/bin/bash
# Test Stage 3 persistent CTAs on H100

set -e

RUNPOD_IP="154.57.34.90"
RUNPOD_PORT="14727"

echo "========================================="
echo "STAGE 3: PERSISTENT CTAS (Grid-Stride)"
echo "========================================="
echo ""

# Deploy Stage 3 kernel
echo "[1/2] Deploying persistent CTA kernel..."
scp -P ${RUNPOD_PORT} -o StrictHostKeyChecking=no \
    flashcore/fast/attention_stage3_persistent.py \
    root@${RUNPOD_IP}:/workspace/flashcore_llama/flashcore/fast/
echo "✅ Deployed"
echo ""

# Run Stage 3 test
echo "[2/2] Running batching efficiency test..."
echo "Will test B=1, 8, 16, 32 to measure batching speedup"
echo ""

ssh -p ${RUNPOD_PORT} -o StrictHostKeyChecking=no root@${RUNPOD_IP} << 'ENDSSH'
cd /workspace/flashcore_llama
export PYTHONPATH=/workspace/flashcore_llama:$PYTHONPATH

echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "SMs: $(nvidia-smi --query-gpu=sm --format=csv,noheader 2>/dev/null || echo '132 (H100)')"
echo ""

python3 flashcore/fast/attention_stage3_persistent.py

ENDSSH

echo ""
echo "========================================="
echo "STAGE 3 TEST COMPLETE"
echo "========================================="

