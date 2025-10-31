#!/bin/bash
# Test Stage 2 optimized kernel on H100

set -e

RUNPOD_IP="154.57.34.90"
RUNPOD_PORT="14727"

echo "========================================="
echo "STAGE 2: OPTIMIZED INSTRUCTION SCHEDULING"
echo "========================================="
echo ""

# Deploy Stage 2 kernel
echo "[1/2] Deploying Stage 2 kernel..."
scp -P ${RUNPOD_PORT} -o StrictHostKeyChecking=no \
    flashcore/fast/attention_stage2_optimized.py \
    root@${RUNPOD_IP}:/workspace/flashcore_llama/flashcore/fast/
echo "✅ Deployed"
echo ""

# Run Stage 2 test
echo "[2/2] Running Stage 2 validation..."
ssh -p ${RUNPOD_PORT} -o StrictHostKeyChecking=no root@${RUNPOD_IP} << 'ENDSSH'
cd /workspace/flashcore_llama
export PYTHONPATH=/workspace/flashcore_llama:$PYTHONPATH

echo "Environment:"
echo "  GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo ""

python3 flashcore/fast/attention_stage2_optimized.py

ENDSSH

echo ""
echo "========================================="
echo "STAGE 2 TEST COMPLETE"
echo "========================================="

