#!/usr/bin/env bash
#
# Deploy FP8 Stage C Fixes and Run Priority 1.3 Validation
#

set -euo pipefail

GPU_INSTANCE="cudadent42-l4-dev"
GPU_ZONE="us-west1-c"

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "🚀 FP8 Stage C Validation (Priority 1.3)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "  Instance: $GPU_INSTANCE"
echo "  Zone: $GPU_ZONE"
echo "  Test: Correctness validation (mission shape)"
echo ""

# Create remote script
REMOTE_SCRIPT=$(cat <<'EOF'
#!/bin/bash
set -euo pipefail

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Priority 1.3: Correctness Validation (On GPU)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

cd ~/periodicdent42

# Pull latest code
echo "📥 Pulling latest code..."
git pull
echo ""

# Setup environment
if [[ ! -d venv ]]; then
    echo "📦 Creating virtual environment..."
    python3 -m venv venv
fi

source venv/bin/activate
echo "✅ Virtual environment activated"
echo ""

# Install dependencies if needed
if ! python3 -c "import torch" &>/dev/null; then
    echo "📦 Installing dependencies..."
    pip install torch pytest numpy
fi
echo ""

# Check environment
echo "🔍 Environment:"
echo "   Python: $(python3 --version)"
echo "   PyTorch: $(python3 -c 'import torch; print(torch.__version__)')"
echo "   CUDA: $(python3 -c 'import torch; print(torch.cuda.is_available())')"
echo "   GPU: $(python3 -c 'import torch; print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A")')"
echo ""

# Run Priority 1.3: Quick validation (10 iters)
echo "🧪 Running Priority 1.3: Correctness Validation"
echo "   Shape: mission (B=1, H=8, S=512, D=64)"
echo "   Iterations: 10"
echo ""

python scripts/bench_fp8_stage_c.py --shapes mission --iters 10

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Priority 1.3 Complete!"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

EOF
)

# Execute on GPU instance
echo "🚀 Executing on GPU instance..."
echo ""

gcloud compute ssh "$GPU_INSTANCE" --zone="$GPU_ZONE" --command="$REMOTE_SCRIPT"

echo ""
echo "✅ Validation complete!"

