#!/bin/bash
# Phase 1 Validation: Hopper-Native Skeleton on H100
# Tests: Correctness → Basic Performance → Sanitizer

set -e

# RunPod H100 configuration
RUNPOD_IP="154.57.34.90"
RUNPOD_PORT="14727"
RUNPOD_USER="root"
REMOTE_DIR="/workspace/flashcore_hopper"

echo "========================================"
echo "PHASE 1: HOPPER-NATIVE VALIDATION (H100)"
echo "========================================"
echo ""
echo "Target: ${RUNPOD_IP}:${RUNPOD_PORT}"
echo "Remote: ${REMOTE_DIR}"
echo ""

# Check SSH connectivity
echo "[0/4] Checking H100 connectivity..."
if ! ssh -p ${RUNPOD_PORT} -o ConnectTimeout=5 ${RUNPOD_USER}@${RUNPOD_IP} "echo 'Connected'" > /dev/null 2>&1; then
    echo "❌ ERROR: Cannot connect to H100"
    echo "   Check RunPod 'Connect' tab for current IP/Port"
    exit 1
fi
echo "✅ H100 online"
echo ""

# Create remote directory structure
ssh -p ${RUNPOD_PORT} ${RUNPOD_USER}@${RUNPOD_IP} "mkdir -p ${REMOTE_DIR}/{flashcore/fast,flashcore/cuda,tools,build/bin}"

# Deploy files
echo "[1/2] Deploying kernel and validation system..."
FILES=(
    "flashcore/fast/attention_hopper_tma.cu"
    "flashcore/cuda/test_hopper_kernel.cu"
    "tools/validate_hopper.sh"
)

for file in "${FILES[@]}"; do
    echo "  - $file"
    scp -P ${RUNPOD_PORT} -q ${file} ${RUNPOD_USER}@${RUNPOD_IP}:${REMOTE_DIR}/${file}
done
echo "✅ Deployed"
echo ""

# Make scripts executable
ssh -p ${RUNPOD_PORT} ${RUNPOD_USER}@${RUNPOD_IP} "chmod +x ${REMOTE_DIR}/tools/validate_hopper.sh"

# Run structured validation
echo "[2/2] Running structured validation on H100..."
ssh -p ${RUNPOD_PORT} ${RUNPOD_USER}@${RUNPOD_IP} "cd ${REMOTE_DIR} && ./tools/validate_hopper.sh" 2>&1 | tee phase1_validation.log

if [ ${PIPESTATUS[0]} -eq 0 ]; then
    echo ""
    echo "✅ PHASE 1 VALIDATION PASSED"
else
    echo ""
    echo "❌ PHASE 1 VALIDATION FAILED - see phase1_validation.log"
    exit 1
fi
echo ""

echo "========================================"
echo "PHASE 1 VALIDATION COMPLETE"
echo "========================================"
echo ""
echo "Logs:"
echo "  - phase1_build.log (build + correctness)"
echo "  - phase1_sanitizer.log (memory validation)"
echo ""
echo "Next steps:"
echo "  1. If correctness passed: Deploy NSight Compute profiling"
echo "  2. Profile command:"
echo "     ssh -p ${RUNPOD_PORT} ${RUNPOD_USER}@${RUNPOD_IP}"
echo "     cd ${REMOTE_DIR}"
echo "     ./tools/ncu_validate.sh ./build/bin/test_hopper"
echo ""

