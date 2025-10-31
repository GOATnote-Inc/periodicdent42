#!/bin/bash
# Deploy Stage 5 to RunPod H100 and run Stage 1 validation

set -e

echo "========================================="
echo "DEPLOYING STAGE 5 TO H100"
echo "========================================="
echo ""

# RunPod connection details
RUNPOD_IP="154.57.34.90"
RUNPOD_PORT="14727"
RUNPOD_HOST="root@${RUNPOD_IP}"

echo "Target: ${RUNPOD_HOST}:${RUNPOD_PORT}"
echo ""

# 1. Deploy Stage 5 kernel
echo "[1/4] Deploying attention_stage5_warpspec.py..."
scp -P ${RUNPOD_PORT} -o StrictHostKeyChecking=no \
    flashcore/fast/attention_stage5_warpspec.py \
    ${RUNPOD_HOST}:/workspace/flashcore_llama/flashcore/fast/
echo "✅ Stage 5 kernel deployed"
echo ""

# 2. Deploy validator
echo "[2/4] Deploying stage_validator.py..."
ssh -p ${RUNPOD_PORT} -o StrictHostKeyChecking=no ${RUNPOD_HOST} \
    "mkdir -p /workspace/flashcore_llama/flashcore/validation"

scp -P ${RUNPOD_PORT} -o StrictHostKeyChecking=no \
    flashcore/validation/stage_validator.py \
    ${RUNPOD_HOST}:/workspace/flashcore_llama/flashcore/validation/
echo "✅ Validator deployed"
echo ""

# 3. Create __init__.py for validation module
echo "[3/4] Creating validation module __init__.py..."
ssh -p ${RUNPOD_PORT} -o StrictHostKeyChecking=no ${RUNPOD_HOST} \
    "touch /workspace/flashcore_llama/flashcore/validation/__init__.py"
echo "✅ Module structure ready"
echo ""

# 4. Run Stage 1 validation
echo "[4/4] Running Stage 1 validation..."
echo "========================================="
echo "STAGE 1: PRODUCER/CONSUMER ARCHITECTURE"
echo "========================================="
echo ""

ssh -p ${RUNPOD_PORT} -o StrictHostKeyChecking=no ${RUNPOD_HOST} << 'ENDSSH'
cd /workspace/flashcore_llama
export PYTHONPATH=/workspace/flashcore_llama:$PYTHONPATH

echo "Environment:"
echo "  Python: $(python3 --version)"
echo "  PyTorch: $(python3 -c 'import torch; print(torch.__version__)')"
echo "  Triton: $(python3 -c 'import triton; print(triton.__version__)')"
echo "  CUDA: $(python3 -c 'import torch; print(torch.version.cuda)')"
echo "  GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo ""

echo "Running Stage 1 Validation (Correctness Gate)..."
python3 -m flashcore.validation.stage_validator --stage 1

ENDSSH

echo ""
echo "========================================="
echo "DEPLOYMENT AND VALIDATION COMPLETE"
echo "========================================="

