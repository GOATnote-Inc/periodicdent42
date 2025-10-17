#!/bin/bash
#
# TDD: Test-Driven Baseline Execution on L4 (sm_89)
#
# Stern directive: NO QUITTING!
# - On failure: print logs, try next baseline, continue
# - Test all baselines systematically
# - Pick champion at the end
#

set -e  # Exit on error, but we'll catch and continue

echo "════════════════════════════════════════════════════════════════════════════════"
echo "TDD BASELINE EXECUTION ON L4 (sm_89)"
echo "════════════════════════════════════════════════════════════════════════════════"
echo ""
echo "Directive: NO QUITTING - systematic fallback on failures!"
echo ""

# Step 1: Verify environment
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "STEP 1: Environment Verification"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

cd ~/periodicdent42
source ~/venv/bin/activate
export PYTHONPATH="$(pwd):${PYTHONPATH:-}"

echo "Working directory: $(pwd)"
echo "Python: $(which python)"
echo "PyTorch: $(python -c 'import torch; print(torch.__version__)')"
echo ""

# CUDA check
python - <<'PY' || { echo "⚠️  CUDA check failed, continuing..."; }
import torch
assert torch.cuda.is_available(), "CUDA not available"
cc = torch.cuda.get_device_capability()
print(f"✅ GPU: {torch.cuda.get_device_name(0)}")
print(f"✅ Compute Capability: sm_{cc[0]}{cc[1]}")
if cc[0]*10+cc[1] != 89:
    print(f"⚠️  Expected sm_89 (Ada/L4), got sm_{cc[0]}{cc[1]}")
PY

echo ""

# Step 2: Test PyTorch SDPA backends (should work out-of-box)
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "STEP 2: Test PyTorch SDPA Backends (No Installation Required)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

python - <<'PY' || { echo "⚠️  Backend check failed, continuing..."; }
import torch
from torch.nn.attention import sdpa_kernel, SDPBackend

print("Checking SDPA backend availability...")
backends = [
    ("FLASH_ATTENTION", SDPBackend.FLASH_ATTENTION),
    ("CUDNN_ATTENTION", SDPBackend.CUDNN_ATTENTION),
    ("EFFICIENT_ATTENTION", SDPBackend.EFFICIENT_ATTENTION),
    ("MATH", SDPBackend.MATH),
]

available = []
for name, backend in backends:
    try:
        # Try to enable the backend
        with sdpa_kernel(backend):
            pass
        print(f"  ✅ {name}")
        available.append(name)
    except Exception as e:
        print(f"  ❌ {name}: {e}")

print(f"\n✅ Available SDPA backends: {available}")
PY

echo ""

# Step 3: Quick smoke test (no FA-2 yet)
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "STEP 3: Smoke Test (PyTorch SDPA Only)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

python scripts/bench_baselines.py --iters 10 || {
    echo "⚠️  Initial benchmark failed, checking logs..."
    echo "Continuing with FA-2 installation..."
}

echo ""

# Step 4: Install FlashAttention-2
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "STEP 4: Install FlashAttention-2 (flash-attn==2.8.3)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

export TORCH_CUDA_ARCH_LIST="8.9"
export MAX_JOBS=4

# Check if already installed
if python -c "import flash_attn" 2>/dev/null; then
    echo "✅ flash-attn already installed"
    python -c "import flash_attn; print(f'   Version: {flash_attn.__version__}')"
else
    echo "Installing flash-attn==2.8.3 (this may take 10-15 minutes)..."
    echo "TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST}"
    echo ""
    
    pip install --upgrade flash-attn==2.8.3 --no-build-isolation 2>&1 | tee /tmp/flash_attn_install.log || {
        echo ""
        echo "⚠️  FA-2 installation failed!"
        echo "Last 50 lines of log:"
        tail -50 /tmp/flash_attn_install.log
        echo ""
        echo "Continuing without FA-2 - PyTorch SDPA baselines should still work!"
    }
fi

echo ""

# Step 5: Full benchmark (all baselines)
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "STEP 5: Full Benchmark (All Available Baselines)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

python scripts/bench_baselines.py --iters 50 2>&1 | tee evidence/baseline_benchmark_l4.log || {
    echo ""
    echo "⚠️  Benchmark encountered errors, but continuing..."
    echo "Check evidence/baseline_benchmark_l4.log for details"
}

echo ""

# Step 6: Summary
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "STEP 6: Summary"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

if [ -f evidence/baseline_benchmark_l4.log ]; then
    echo "✅ Benchmark log saved: evidence/baseline_benchmark_l4.log"
    echo ""
    echo "Champion baseline:"
    grep "🏆" evidence/baseline_benchmark_l4.log || echo "  (see log for details)"
else
    echo "⚠️  No benchmark log generated"
fi

echo ""
echo "════════════════════════════════════════════════════════════════════════════════"
echo "TDD EXECUTION COMPLETE"
echo "════════════════════════════════════════════════════════════════════════════════"
echo ""
echo "Next steps:"
echo "  1. Review evidence/baseline_benchmark_l4.log"
echo "  2. Profile champion with NCU"
echo "  3. Iterate to beat champion performance"
echo ""

