#!/bin/bash
# RunPod H100 - Zero-config deployment
# Usage: ./RUNPOD_QUICKSTART.sh <pod-id>

POD_ID=${1:-""}
if [ -z "$POD_ID" ]; then
    echo "Usage: ./RUNPOD_QUICKSTART.sh <your-pod-id>"
    echo "Get pod ID from RunPod dashboard"
    exit 1
fi

RUNPOD_SSH="ssh root@${POD_ID}-ssh.runpod.io"
RUNPOD_SCP="scp -r"

echo "🚀 Deploying to RunPod pod: $POD_ID"

# Package
tar czf runpod_deploy.tar.gz \
    flashcore/fast/*.cu \
    test_wgmma_single_corrected.cu \
    build_test_wgmma_corrected.sh \
    benchmark_vs_sglang.py

# Deploy
$RUNPOD_SCP runpod_deploy.tar.gz root@${POD_ID}-ssh.runpod.io:/workspace/

# Execute
$RUNPOD_SSH << 'EOF'
cd /workspace
tar xzf runpod_deploy.tar.gz
chmod +x build_test_wgmma_corrected.sh benchmark_vs_sglang.py

echo "=== STEP 1: Build ==="
./build_test_wgmma_corrected.sh 2>&1 | grep -E "registers|spill|✅|❌|Build"

echo ""
echo "=== STEP 2: Validate Step 1 (2.8-3.5 TFLOPS) ==="
./build/bin/test_wgmma_corrected | grep -E "Median:|Status:|SUCCESS|FAIL|Error"

echo ""
echo "=== STEP 3: System Info ==="
nvidia-smi --query-gpu=name,compute_cap,memory.total --format=csv,noheader

echo ""
echo "✅ Deployment complete. Check results above."
EOF

echo ""
echo "📊 Next: Iterate to 55-65 TFLOPS"
echo "   ssh root@${POD_ID}-ssh.runpod.io"

