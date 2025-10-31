#!/bin/bash
# Test WMMA kernel on H100 - Iteration 1

set -e

RUNPOD_IP="154.57.34.90"
RUNPOD_PORT="14727"

echo "========================================="
echo "ITERATION 1: WMMA TENSOR CORES"
echo "========================================="
echo ""
echo "Target: 100-200 TFLOPS (62-125× improvement)"
echo "Baseline: 1.6 TFLOPS"
echo ""

echo "[1/2] Deploying..."
scp -P ${RUNPOD_PORT} -o StrictHostKeyChecking=no \
    flashcore/fast/attention_cuda_wmma.cu \
    flashcore/cuda/test_hopper_kernel.cu \
    build_cuda_simple.sh \
    root@${RUNPOD_IP}:/workspace/flashcore_llama/
echo "✅ Deployed"
echo ""

echo "[2/2] Building and testing..."
ssh -p ${RUNPOD_PORT} -o StrictHostKeyChecking=no root@${RUNPOD_IP} << 'ENDSSH'
cd /workspace/flashcore_llama
mkdir -p flashcore/cuda flashcore/fast
mv -f attention_cuda_wmma.cu flashcore/fast/ 2>/dev/null || true
mv -f test_hopper_kernel.cu flashcore/cuda/ 2>/dev/null || true
chmod +x build_cuda_simple.sh
./build_cuda_simple.sh
ENDSSH

echo ""
echo "========================================="
echo "ITERATION 1 COMPLETE"
echo "========================================="

