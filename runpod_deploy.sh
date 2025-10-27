#!/bin/bash
# RunPod H100 Deployment - Zero fluff, maximum action
set -e

RUNPOD_IP=${1:-"your.runpod.ip"}
RUNPOD_PORT=${2:-"22"}

echo "🚀 Deploying to RunPod H100..."

# Build package
tar czf deploy.tar.gz \
  flashcore/fast/attention_phase6_wgmma_corrected.cu \
  test_wgmma_single_corrected.cu \
  build_test_wgmma_corrected.sh

# Deploy and run
scp -P $RUNPOD_PORT deploy.tar.gz root@$RUNPOD_IP:/workspace/
ssh -p $RUNPOD_PORT root@$RUNPOD_IP << 'ENDSSH'
cd /workspace
tar xzf deploy.tar.gz
chmod +x build_test_wgmma_corrected.sh
./build_test_wgmma_corrected.sh 2>&1 | grep -E "registers|spill|✅|❌"
./build/bin/test_wgmma_corrected | grep -E "TFLOPS|Status|SUCCESS|FAIL"
ENDSSH

echo "✅ Done. Check output above."

