#!/bin/bash
# Auto-deploy to RunPod H100
set -e

# Check for RunPod credentials
if [ -f ~/.runpod_credentials ]; then
    source ~/.runpod_credentials
elif [ ! -z "$RUNPOD_POD_ID" ]; then
    POD_ID=$RUNPOD_POD_ID
elif [ ! -z "$1" ]; then
    POD_ID=$1
else
    echo "⚠️  Provide RunPod pod ID:"
    echo ""
    echo "Option 1: Pass as argument"
    echo "  ./deploy_now.sh <pod-id>"
    echo ""
    echo "Option 2: Set environment variable"
    echo "  export RUNPOD_POD_ID=<pod-id>"
    echo "  ./deploy_now.sh"
    echo ""
    echo "Option 3: Save credentials"
    echo "  echo 'POD_ID=<pod-id>' > ~/.runpod_credentials"
    echo "  ./deploy_now.sh"
    echo ""
    echo "Get pod ID from: https://www.runpod.io/console/pods"
    exit 1
fi

echo "🚀 Deploying to RunPod pod: $POD_ID"
echo ""

# Package
echo "📦 Packaging files..."
tar czf deploy.tar.gz \
    flashcore/fast/attention_phase6_wgmma_corrected.cu \
    flashcore/fast/attention_phase6_step2_multi.cu \
    flashcore/fast/attention_phase6_step3_pipeline.cu \
    test_wgmma_single_corrected.cu \
    build_test_wgmma_corrected.sh \
    Makefile \
    iterate_h100.sh \
    benchmark_vs_sglang.py

# Deploy
echo "📤 Uploading to RunPod..."
scp -o StrictHostKeyChecking=no deploy.tar.gz root@${POD_ID}-ssh.runpod.io:/workspace/ 2>&1 | grep -v "Warning:"

# Execute
echo ""
echo "⚡ Building and testing on H100..."
ssh -o StrictHostKeyChecking=no root@${POD_ID}-ssh.runpod.io << 'ENDSSH'
cd /workspace
tar xzf deploy.tar.gz
chmod +x *.sh

echo "=== BUILD ==="
nvcc -arch=sm_90a -O3 --use_fast_math -lineinfo --ptxas-options=-v \
    -I. test_wgmma_single_corrected.cu \
    -o test_step1 2>&1 | grep -E "registers|spill|error|warning" || echo "Build OK"

echo ""
echo "=== GPU INFO ==="
nvidia-smi --query-gpu=name,compute_cap,memory.total --format=csv,noheader

echo ""
echo "=== STEP 1 TEST (Target: 2.8-3.5 TFLOPS) ==="
./test_step1 2>&1 | grep -E "Median.*TFLOPS|Status.*EXCELLENT|Status.*PASS|Status.*FAIL|Max Error"

echo ""
echo "✅ Deployment complete"
ENDSSH

echo ""
echo "📊 To iterate:"
echo "  ssh root@${POD_ID}-ssh.runpod.io"
echo "  cd /workspace"
echo "  make test"
echo ""
echo "🎯 Next: Steps 2-5 to reach 55-65 TFLOPS"

# Cleanup
rm -f deploy.tar.gz

