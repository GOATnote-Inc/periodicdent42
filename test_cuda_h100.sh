#!/bin/bash
# Deploy and test CUDA kernel on H100 (simple version)

set -e

RUNPOD_IP="154.57.34.90"
RUNPOD_PORT="14727"

echo "========================================="
echo "CUDA KERNEL: BUILD & TEST ON H100"
echo "========================================="
echo ""

echo "[1/3] Deploying files..."
scp -P ${RUNPOD_PORT} -o StrictHostKeyChecking=no \
    flashcore/fast/attention_hopper_cuda.cu \
    flashcore/cuda/test_hopper_kernel.cu \
    build_cuda_simple.sh \
    root@${RUNPOD_IP}:/workspace/flashcore_llama/
echo "✅ Deployed"
echo ""

echo "[2/3] Building on H100..."
ssh -p ${RUNPOD_PORT} -o StrictHostKeyChecking=no root@${RUNPOD_IP} << 'ENDSSH'
cd /workspace/flashcore_llama
mkdir -p flashcore/cuda flashcore/fast
mv -f attention_hopper_cuda.cu flashcore/fast/ 2>/dev/null || true
mv -f test_hopper_kernel.cu flashcore/cuda/ 2>/dev/null || true
chmod +x build_cuda_simple.sh
./build_cuda_simple.sh
ENDSSH

echo ""
echo "========================================="
echo "BUILD & TEST COMPLETE"
echo "========================================="

