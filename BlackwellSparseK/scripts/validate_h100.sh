#!/bin/bash
# ============================================================================
# RunPod H100 Validation Script
# ============================================================================
# Deploys BlackwellSparseK to RunPod H100 and runs validation tests
# ============================================================================

set -e

# Configuration (override with environment variables)
RUNPOD_IP=${RUNPOD_IP:-""}
RUNPOD_PORT=${RUNPOD_PORT:-""}
RUNPOD_USER=${RUNPOD_USER:-"root"}

# Check configuration
if [ -z "$RUNPOD_IP" ] || [ -z "$RUNPOD_PORT" ]; then
    echo "❌ Error: RUNPOD_IP and RUNPOD_PORT must be set"
    echo ""
    echo "Usage:"
    echo "  export RUNPOD_IP=154.57.34.90"
    echo "  export RUNPOD_PORT=23673"
    echo "  ./scripts/validate_h100.sh"
    exit 1
fi

SSH_HOST="${RUNPOD_USER}@${RUNPOD_IP}"
SSH_OPTS="-p ${RUNPOD_PORT} -o StrictHostKeyChecking=no -o TCPKeepAlive=yes -o ServerAliveInterval=20"

echo "======================================"
echo "BlackwellSparseK H100 Validation"
echo "======================================"
echo "Host: ${SSH_HOST}"
echo "Port: ${RUNPOD_PORT}"
echo ""

# Test SSH connection with retries
echo "📡 Testing SSH connection..."
MAX_RETRIES=10
RETRY=0

while [ $RETRY -lt $MAX_RETRIES ]; do
    if ssh ${SSH_OPTS} ${SSH_HOST} "echo 'Connection OK'" &> /dev/null; then
        echo "✅ SSH connection successful"
        break
    fi
    
    RETRY=$((RETRY + 1))
    echo "⏳ Retry $RETRY/$MAX_RETRIES..."
    sleep 5
done

if [ $RETRY -eq $MAX_RETRIES ]; then
    echo "❌ SSH connection failed after $MAX_RETRIES attempts"
    exit 1
fi

# Create workspace
echo ""
echo "📁 Creating workspace..."
ssh ${SSH_OPTS} ${SSH_HOST} "mkdir -p /workspace/BlackwellSparseK"

# Upload source code
echo ""
echo "📤 Uploading source code..."
scp ${SSH_OPTS} -r \
    src/ \
    tests/ \
    benchmarks/ \
    examples/ \
    pyproject.toml \
    setup.py \
    ${SSH_HOST}:/workspace/BlackwellSparseK/

# Install package
echo ""
echo "📦 Installing BlackwellSparseK..."
ssh ${SSH_OPTS} ${SSH_HOST} "cd /workspace/BlackwellSparseK && pip install -e .[dev,bench]"

# Verify GPU
echo ""
echo "🔍 Verifying H100 GPU..."
ssh ${SSH_OPTS} ${SSH_HOST} "nvidia-smi --query-gpu=name,compute_cap,memory.total --format=csv,noheader"

# Run tests
echo ""
echo "🧪 Running test suite..."
ssh ${SSH_OPTS} ${SSH_HOST} "cd /workspace/BlackwellSparseK && pytest tests/ -v"

# Run benchmarks
echo ""
echo "📊 Running performance benchmarks..."
ssh ${SSH_OPTS} ${SSH_HOST} "cd /workspace/BlackwellSparseK && python benchmarks/perf.py --save-results"

# Download results
echo ""
echo "📥 Downloading results..."
mkdir -p results/h100_validation
scp ${SSH_OPTS} -r ${SSH_HOST}:/workspace/BlackwellSparseK/results/* results/h100_validation/ || true

echo ""
echo "======================================"
echo "✅ H100 Validation Complete!"
echo "======================================"
echo "Results: results/h100_validation/"
echo ""

