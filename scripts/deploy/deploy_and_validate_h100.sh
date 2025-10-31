#!/bin/bash
# Deploy WMMA kernel + NVIDIA toolchain to H100 and run full validation

set -e

RUNPOD_IP="${RUNPOD_IP:-154.57.34.90}"
RUNPOD_PORT="${RUNPOD_PORT:-14727}"

echo "========================================"
echo "WMMA KERNEL VALIDATION (H100)"
echo "========================================"
echo ""
echo "Target: ${RUNPOD_IP}:${RUNPOD_PORT}"
echo ""

#==============================================================================
# STEP 1: DEPLOY FILES
#==============================================================================

echo "[1/3] Deploying files..."
scp -P ${RUNPOD_PORT} -o StrictHostKeyChecking=no \
    flashcore/fast/attention_cuda_wmma.cu \
    flashcore/cuda/test_hopper_kernel.cu \
    tools/run_debug_profile.sh \
    root@${RUNPOD_IP}:/workspace/flashcore_llama/

echo "✅ Deployed"
echo ""

#==============================================================================
# STEP 2: RUN BASELINE (Correctness + Performance)
#==============================================================================

echo "[2/3] Running baseline validation..."
ssh -p ${RUNPOD_PORT} -o StrictHostKeyChecking=no root@${RUNPOD_IP} << 'ENDSSH'
cd /workspace/flashcore_llama
mkdir -p flashcore/cuda flashcore/fast tools build/bin
mv -f attention_cuda_wmma.cu flashcore/fast/ 2>/dev/null || true
mv -f test_hopper_kernel.cu flashcore/cuda/ 2>/dev/null || true
mv -f run_debug_profile.sh tools/ 2>/dev/null || true
chmod +x tools/run_debug_profile.sh

echo ""
echo "========================================="
echo "BASELINE: Correctness + Performance"
echo "========================================="
RUN_BASELINE=1 ./tools/run_debug_profile.sh
ENDSSH

echo ""

#==============================================================================
# STEP 3: RUN SANITIZER (Memory + Sync Validation)
#==============================================================================

echo "[3/3] Running compute-sanitizer..."
ssh -p ${RUNPOD_PORT} -o StrictHostKeyChecking=no root@${RUNPOD_IP} << 'ENDSSH'
cd /workspace/flashcore_llama

echo ""
echo "========================================="
echo "SANITIZER: Memory + Sync Checks"
echo "========================================="
RUN_BASELINE=0 RUN_SANITIZER=1 ./tools/run_debug_profile.sh
ENDSSH

echo ""
echo "========================================"
echo "VALIDATION COMPLETE"
echo "========================================"
echo ""
echo "Optional: Run profiler for detailed metrics"
echo "  ssh -p ${RUNPOD_PORT} root@${RUNPOD_IP}"
echo "  cd /workspace/flashcore_llama"
echo "  RUN_PROFILER=1 ./tools/run_debug_profile.sh"
echo ""

