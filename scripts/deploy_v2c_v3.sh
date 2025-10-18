#!/usr/bin/env bash
#
# Deploy V2c-v3 to GPU Instance and Run Tests
#
# Usage:
#   bash scripts/deploy_v2c_v3.sh
#
# This script:
#   1. Pushes latest code to GitHub
#   2. SSHs to GPU instance (cudadent42-l4-dev)
#   3. Pulls latest code
#   4. Runs V2c-v3 tests
#   5. Captures results
#
# Prerequisites:
#   - gcloud CLI installed and configured
#   - SSH access to cudadent42-l4-dev
#   - Git remote configured

set -euo pipefail

# Configuration
GPU_INSTANCE="cudadent42-l4-dev"
GPU_ZONE="us-central1-a"
GPU_PROJECT="periodicdent42"  # Update if different

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "V2c-v3 GPU Deployment & Testing"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

# Step 1: Commit and push latest code
echo "📦 Step 1: Pushing latest code to GitHub..."
echo ""

cd "$(dirname "$0")/.."

if [[ -n $(git status --porcelain) ]]; then
    echo "⚠️  Uncommitted changes detected. Please commit first."
    echo ""
    git status --short
    echo ""
    read -p "Commit now? (y/n) " -n 1 -r
    echo ""
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        git add -A
        read -p "Commit message: " commit_msg
        git commit -m "$commit_msg"
        git push
        echo "✅ Code pushed"
    else
        echo "❌ Deployment cancelled"
        exit 1
    fi
else
    git push || true
    echo "✅ Code is up to date"
fi

echo ""

# Step 2: Check GPU instance availability
echo "🔍 Step 2: Checking GPU instance..."
echo ""

if ! gcloud compute instances describe "$GPU_INSTANCE" --zone="$GPU_ZONE" &>/dev/null; then
    echo "❌ GPU instance '$GPU_INSTANCE' not found in zone '$GPU_ZONE'"
    echo ""
    echo "Available instances:"
    gcloud compute instances list --format="table(name,zone,status)"
    exit 1
fi

INSTANCE_STATUS=$(gcloud compute instances describe "$GPU_INSTANCE" --zone="$GPU_ZONE" --format="get(status)")
echo "   Instance: $GPU_INSTANCE"
echo "   Zone: $GPU_ZONE"
echo "   Status: $INSTANCE_STATUS"
echo ""

if [[ "$INSTANCE_STATUS" != "RUNNING" ]]; then
    echo "⚠️  Instance is not running. Start it?"
    read -p "Start instance? (y/n) " -n 1 -r
    echo ""
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo "🚀 Starting instance..."
        gcloud compute instances start "$GPU_INSTANCE" --zone="$GPU_ZONE"
        echo "   Waiting for instance to start..."
        sleep 30
        echo "✅ Instance started"
    else
        echo "❌ Deployment cancelled"
        exit 1
    fi
fi

echo ""

# Step 3: Deploy and test on GPU
echo "🧪 Step 3: Running tests on GPU instance..."
echo ""

# Create remote test script
REMOTE_SCRIPT=$(cat <<'EOF'
#!/bin/bash
set -euo pipefail

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "V2c-v3 Testing on GPU Instance"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

cd ~/periodicdent42
echo "📍 Working directory: $(pwd)"
echo ""

# Pull latest code
echo "📥 Pulling latest code..."
git pull
echo "✅ Code updated"
echo ""

# Activate environment
if [[ -f ~/venv/bin/activate ]]; then
    source ~/venv/bin/activate
    echo "✅ Virtual environment activated"
elif [[ -f ~/.conda/etc/profile.d/conda.sh ]]; then
    source ~/.conda/etc/profile.d/conda.sh
    conda activate base
    echo "✅ Conda environment activated"
else
    echo "⚠️  No virtual environment found, using system Python"
fi

echo ""

# Check environment
echo "🔍 Environment Check:"
echo "   Python: $(python3 --version)"
echo "   PyTorch: $(python3 -c 'import torch; print(torch.__version__)')"
echo "   CUDA Available: $(python3 -c 'import torch; print(torch.cuda.is_available())')"
if python3 -c 'import torch; torch.cuda.is_available()' &>/dev/null; then
    echo "   GPU: $(python3 -c 'import torch; print(torch.cuda.get_device_name(0))')"
fi
echo ""

# Navigate to evo-sdpa
cd evo-sdpa

# Check if test script exists
if [[ ! -f TEST_V2C_V3.sh ]]; then
    echo "❌ TEST_V2C_V3.sh not found"
    echo "   Looking for: $(pwd)/TEST_V2C_V3.sh"
    exit 1
fi

# Run tests
echo "🧪 Running V2c-v3 tests..."
echo ""
bash TEST_V2C_V3.sh | tee v2c_v3_test_results.log

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Testing Complete!"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "Results saved to: evo-sdpa/v2c_v3_test_results.log"
echo ""
EOF
)

# Execute remote script
gcloud compute ssh "$GPU_INSTANCE" --zone="$GPU_ZONE" --command="$REMOTE_SCRIPT"

echo ""

# Step 4: Download results
echo "📥 Step 4: Downloading test results..."
echo ""

mkdir -p evidence/v2c_v3
gcloud compute scp "$GPU_INSTANCE:~/periodicdent42/evo-sdpa/v2c_v3_test_results.log" \
    evidence/v2c_v3/ --zone="$GPU_ZONE" || true

if [[ -f evidence/v2c_v3/v2c_v3_test_results.log ]]; then
    echo "✅ Results downloaded to: evidence/v2c_v3/v2c_v3_test_results.log"
    echo ""
    
    # Show summary
    echo "📊 Test Summary:"
    echo ""
    grep -E "(PASS|FAIL|SUMMARY)" evidence/v2c_v3/v2c_v3_test_results.log || true
else
    echo "⚠️  Could not download results (may not exist yet)"
fi

echo ""

# Step 5: Next steps
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "✅ Deployment Complete!"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "📋 Next Steps:"
echo ""

if [[ -f evidence/v2c_v3/v2c_v3_test_results.log ]]; then
    if grep -q "5/5 tests passed" evidence/v2c_v3/v2c_v3_test_results.log 2>/dev/null; then
        echo "✅ All tests passed!"
        echo ""
        echo "   → Proceed to Iteration 4: WMMA + K^T"
        echo "   → Target: 800-1200 μs, 100% correctness"
        echo "   → Estimated time: 1-2 hours"
        echo ""
    else
        echo "❌ Some tests failed"
        echo ""
        echo "   → Review results: evidence/v2c_v3/v2c_v3_test_results.log"
        echo "   → Debug plan: evo-sdpa/V2C_ITER3_STATUS.md"
        echo "   → SSH to GPU: gcloud compute ssh $GPU_INSTANCE --zone=$GPU_ZONE"
        echo ""
    fi
else
    echo "   → Review full logs on GPU instance"
    echo "   → SSH: gcloud compute ssh $GPU_INSTANCE --zone=$GPU_ZONE"
    echo "   → Log: ~/periodicdent42/evo-sdpa/v2c_v3_test_results.log"
    echo ""
fi

echo "📚 Documentation:"
echo "   - V2C_SESSION_SUMMARY_OCT18.md (full session overview)"
echo "   - V2C_ITER3_STATUS.md (iteration 3 details)"
echo "   - V2C_ITERATION_LOG.md (complete timeline)"
echo ""


