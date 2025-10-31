#!/bin/bash
# Deploy Issue #6638 Reproducer to H100
set -euo pipefail

RUNPOD_IP="${1:-154.57.34.90}"
RUNPOD_PORT="${2:-35960}"
SSH_OPTS="-o StrictHostKeyChecking=no -o TCPKeepAlive=yes -o ServerAliveInterval=20"

echo "=========================================="
echo "DEPLOYING ISSUE #6638 REPRODUCER TO H100"
echo "=========================================="

# Upload reproducer
echo "📦 Uploading reproducer..."
scp -P "$RUNPOD_PORT" $SSH_OPTS \
    tma-issue-6638-reproducer/reproduce_6638.py \
    root@"$RUNPOD_IP":/root/

# Run test on H100
echo "🔬 Running reproducer on H100..."
ssh -p "$RUNPOD_PORT" $SSH_OPTS root@"$RUNPOD_IP" 'bash -s' <<'REMOTE'
set -euxo pipefail

cd /root

# Ensure latest PyTorch/Triton
python3 -c "import torch; print(f'PyTorch: {torch.__version__}')"
python3 -c "import triton; print(f'Triton: {triton.__version__}')"

# Run reproducer
echo ""
echo "=========================================="
echo "EXECUTING ISSUE #6638 REPRODUCER"
echo "=========================================="
python3 reproduce_6638.py 2>&1 | tee reproduce_6638_results.txt

exit 0
REMOTE

# Download results
echo "⬇️  Downloading results..."
scp -P "$RUNPOD_PORT" $SSH_OPTS \
    root@"$RUNPOD_IP":/root/reproduce_6638_results.txt \
    tma-issue-6638-reproducer/

echo ""
echo "=========================================="
echo "RESULTS"
echo "=========================================="
cat tma-issue-6638-reproducer/reproduce_6638_results.txt | tail -40

