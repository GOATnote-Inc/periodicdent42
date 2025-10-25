#!/bin/bash
#
# Fix: Upgrade PyTorch to 2.5.0+ for torch.nn.attention support
#
set -e

echo "════════════════════════════════════════════════════════════════════════════════"
echo "FIX: Upgrade PyTorch for SDPA Support"
echo "════════════════════════════════════════════════════════════════════════════════"
echo ""

cd ~/periodicdent42
source ~/venv/bin/activate

# Current version
echo "Current PyTorch: $(python -c 'import torch; print(torch.__version__)')"
echo ""

# Install build dependencies
echo "Step 1: Install build dependencies (wheel, packaging, ninja)..."
pip install --upgrade wheel packaging ninja setuptools
echo ""

# Upgrade PyTorch to 2.5.0
echo "Step 2: Upgrade PyTorch to 2.5.0+cu121..."
pip install --upgrade torch==2.5.0+cu121 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
echo ""

# Verify
echo "Step 3: Verify PyTorch upgrade..."
python - <<'PY'
import torch
print(f"✅ PyTorch: {torch.__version__}")
print(f"✅ CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"✅ GPU: {torch.cuda.get_device_name(0)}")
    cc = torch.cuda.get_device_capability()
    print(f"✅ Compute Capability: sm_{cc[0]}{cc[1]}")

# Check for torch.nn.attention
try:
    from torch.nn.attention import sdpa_kernel, SDPBackend
    print("✅ torch.nn.attention available")
except ImportError:
    print("❌ torch.nn.attention NOT available (need PyTorch >= 2.2)")
PY

echo ""
echo "✅ PyTorch upgrade complete!"

