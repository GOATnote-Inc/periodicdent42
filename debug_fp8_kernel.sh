#!/usr/bin/env bash
#
# Debug FP8 Stage C Kernel - Detailed Diagnostics
#

set -euo pipefail

GPU_INSTANCE="cudadent42-l4-dev"
GPU_ZONE="us-west1-c"

REMOTE_SCRIPT=$(cat <<'EOF'
#!/bin/bash
set -euo pipefail

cd ~/periodicdent42
source venv/bin/activate

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "FP8 Stage C Kernel - Detailed Diagnostics"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

# Test 1: Quantizer fix
echo "🔍 Test 1: Quantizer Scale Bug Fix"
python3 test_quantizer_fix.py
echo ""

# Test 2: WMMA unit tests
echo "🔍 Test 2: WMMA Unit Tests"
python3 -m pytest tests/test_fp8_stage_c_wmma.py -xvs
echo ""

# Test 3: Detailed correctness with smaller tensor
echo "🔍 Test 3: Small Tensor Test (B=1, H=1, S=16, D=64)"
python3 <<PYEOF
import sys
sys.path.insert(0, 'cudadent42')
import torch
import torch.nn.functional as F
from bench.sdpa_fp8_stage_c_wmma import sdpa_fp8_stage_c_wmma_forward

torch.manual_seed(42)
Q = torch.randn(1, 1, 16, 64, device='cuda', dtype=torch.float16)
K = torch.randn_like(Q)
V = torch.randn_like(Q)

try:
    out = sdpa_fp8_stage_c_wmma_forward(Q, K, V)
    ref = F.scaled_dot_product_attention(Q.float(), K.float(), V.float()).to(torch.float16)
    
    diff = (out - ref).abs()
    print(f"   Max abs error: {diff.max().item():.2e}")
    print(f"   Mean abs error: {diff.mean().item():.2e}")
    print(f"   % elements > 0.05: {(diff > 0.05).float().mean().item() * 100:.1f}%")
    print(f"   Output sample: {out[0,0,:3,0].cpu().tolist()}")
    print(f"   Ref sample: {ref[0,0,:3,0].cpu().tolist()}")
except Exception as e:
    print(f"   ❌ Error: {e}")
    import traceback
    traceback.print_exc()
PYEOF

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Diagnostics Complete"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

EOF
)

gcloud compute ssh "$GPU_INSTANCE" --zone="$GPU_ZONE" --command="$REMOTE_SCRIPT"

