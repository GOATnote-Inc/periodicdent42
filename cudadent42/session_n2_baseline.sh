#!/bin/bash
# Session N+2 Automated Baseline Script
# Gets to working 0.09× speedup in 15 minutes (vs 60+ min Session N+1)
# 
# Usage (from local machine):
#   cd ~/periodicdent42/cudadent42
#   bash session_n2_baseline.sh

set -e  # Exit on error

PROJECT_ID="periodicdent42"
ZONE="us-central1-a"
INSTANCE="cudadent42-l4-dev"

echo "======================================================================"
echo "Session N+2 Baseline Automation"
echo "Target: 0.09× speedup in 15 minutes (Pattern 6: Git Bisect)"
echo "======================================================================"
echo ""

#------------------------------------------------------------------------------
# Step 1: Start GPU and Wait for Boot (2 min)
#------------------------------------------------------------------------------
echo "⏱️  [1/6] Starting GPU and waiting for boot..."
echo ""

# Check if already running
STATUS=$(gcloud compute instances describe ${INSTANCE} --zone=${ZONE} --project=${PROJECT_ID} --format="value(status)")
if [ "$STATUS" = "RUNNING" ]; then
    echo "✅ GPU already running"
else
    echo "🔄 Starting GPU (this takes ~30 seconds)..."
    gcloud compute instances start ${INSTANCE} --zone=${ZONE} --project=${PROJECT_ID} 2>&1 | grep -E "Starting|done" || true
    echo "⏳ Waiting 30 seconds for SSH to be ready..."
    sleep 30
fi

# Get external IP
EXTERNAL_IP=$(gcloud compute instances describe ${INSTANCE} --zone=${ZONE} --project=${PROJECT_ID} --format="value(networkInterfaces[0].accessConfigs[0].natIP)")
echo "✅ GPU ready at ${EXTERNAL_IP}"
echo ""

#------------------------------------------------------------------------------
# Step 2: Checkout Last Working Commit on GPU (1 min)
#------------------------------------------------------------------------------
echo "⏱️  [2/6] Checking out commit 5b4c0c8 (last working build)..."
echo ""

gcloud compute ssh ${INSTANCE} --zone=${ZONE} --project=${PROJECT_ID} --command="
cd ~/periodicdent42/cudadent42
git fetch origin
git checkout 5b4c0c8 2>&1 | tail -3
echo '✅ Checked out commit 5b4c0c8'
echo ''
echo 'Files present:'
ls -lh setup.py benches/bench_correctness_and_speed.py python/flashmoe_science/csrc/build_config.h | awk '{print \$9, \$5}'
"

echo ""

#------------------------------------------------------------------------------
# Step 3: Measure PyTorch Baseline (2 min)
#------------------------------------------------------------------------------
echo "⏱️  [3/6] Measuring PyTorch baseline @ S=128..."
echo ""

gcloud compute ssh ${INSTANCE} --zone=${ZONE} --project=${PROJECT_ID} --command="
python3 << 'EOF'
import torch
import torch.nn.functional as F

Q = K = V = torch.randn(1, 1, 128, 64, dtype=torch.float16, device='cuda')

# Warmup
for _ in range(10):
    _ = F.scaled_dot_product_attention(Q, K, V)
torch.cuda.synchronize()

# Measure
start = torch.cuda.Event(enable_timing=True)
end = torch.cuda.Event(enable_timing=True)
start.record()
for _ in range(100):
    O = F.scaled_dot_product_attention(Q, K, V)
end.record()
torch.cuda.synchronize()

baseline_ms = start.elapsed_time(end) / 100
print(f'✅ PyTorch baseline @ S=128: {baseline_ms:.3f} ms')
print(f'   Target for 0.5× speedup: < {baseline_ms * 2:.3f} ms (ours)')
print(f'   Target for 1.0× speedup: < {baseline_ms:.3f} ms (ours)')
EOF
"

echo ""

#------------------------------------------------------------------------------
# Step 4: Build Extension (5 min)
#------------------------------------------------------------------------------
echo "⏱️  [4/6] Building CUDA extension (this takes ~2-3 minutes)..."
echo ""

gcloud compute ssh ${INSTANCE} --zone=${ZONE} --project=${PROJECT_ID} --command="
cd ~/periodicdent42/cudadent42

echo '🧹 Cleaning previous build...'
python3 setup.py clean --all 2>&1 | tail -3
rm -rf build/ dist/ *.egg-info flashmoe_science.*.so flashmoe_science/_C.*.so

echo ''
echo '🔨 Building extension...'
python3 setup.py build_ext --inplace 2>&1 | tail -40

echo ''
echo '🔍 Checking for .so file...'
if [ -f 'flashmoe_science/_C.cpython-310-x86_64-linux-gnu.so' ]; then
    ls -lh flashmoe_science/_C.*.so | awk '{print \"✅\", \$9, \$5}'
else
    echo '❌ Extension not built'
    exit 1
fi
"

if [ $? -ne 0 ]; then
    echo ""
    echo "❌ Build failed. Check error messages above."
    echo "   Tip: SSH to GPU and check build.log"
    exit 1
fi

echo ""

#------------------------------------------------------------------------------
# Step 5: Test Import (1 min)
#------------------------------------------------------------------------------
echo "⏱️  [5/6] Testing extension import..."
echo ""

gcloud compute ssh ${INSTANCE} --zone=${ZONE} --project=${PROJECT_ID} --command="
cd ~/periodicdent42/cudadent42
export PYTHONPATH=\"\${PYTHONPATH}:\$(pwd)/python\"
export LD_LIBRARY_PATH=\"/usr/local/lib/python3.10/dist-packages/torch/lib:\${LD_LIBRARY_PATH}\"

python3 -c 'import flashmoe_science._C; print(\"✅ Gate 1 PASSED: Extension imported successfully\")'
"

if [ $? -ne 0 ]; then
    echo ""
    echo "❌ Import failed. Check error messages above."
    exit 1
fi

echo ""

#------------------------------------------------------------------------------
# Step 6: Run Benchmark (3 min)
#------------------------------------------------------------------------------
echo "⏱️  [6/6] Running benchmark @ S=128..."
echo ""

gcloud compute ssh ${INSTANCE} --zone=${ZONE} --project=${PROJECT_ID} --command="
cd ~/periodicdent42/cudadent42
export PYTHONPATH=\"\${PYTHONPATH}:\$(pwd)/python\"
export LD_LIBRARY_PATH=\"/usr/local/lib/python3.10/dist-packages/torch/lib:\${LD_LIBRARY_PATH}\"

python3 benches/bench_correctness_and_speed.py --config small 2>&1 | grep -E '(Config:|PyTorch:|Ours:|Speedup:|Max diff:)'
"

echo ""
echo "======================================================================"
echo "✅ Baseline Complete!"
echo "======================================================================"
echo ""
echo "Expected speedup: 0.09-0.15× (reproducible baseline from Session N)"
echo ""
echo "Next steps:"
echo "  1. If speedup < 0.5× → Profile with Nsight Compute (mandatory)"
echo "  2. Identify bottleneck from profile"
echo "  3. Fix ONE thing"
echo "  4. Rebuild and re-measure"
echo ""
echo "Commands:"
echo "  # SSH to GPU"
echo "  gcloud compute ssh ${INSTANCE} --zone=${ZONE} --project=${PROJECT_ID}"
echo ""
echo "  # Profile tiny config (on GPU)"
echo "  cd ~/periodicdent42/cudadent42"
echo "  ncu --set full python3 benches/bench_correctness_and_speed.py --config tiny"
echo ""
echo "See SESSION_N2_QUICK_START.md for full profiling guide."
echo ""

