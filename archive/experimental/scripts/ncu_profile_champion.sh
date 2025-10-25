#!/bin/bash
#
# NCU Profile: pytorch_sdpa_efficient (xFormers Champion)
#
# Goal: Understand bottlenecks BEFORE optimizing
# - Memory-bound? (DRAM bandwidth utilization)
# - Compute-bound? (SM utilization, FLOPs)
# - Tensor Core utilization? (TC active %)
#

set -e

echo "════════════════════════════════════════════════════════════════════════════════"
echo "NCU PROFILING: Champion Baseline (pytorch_sdpa_efficient)"
echo "════════════════════════════════════════════════════════════════════════════════"
echo ""

cd ~/periodicdent42
source ~/venv/bin/activate
export PYTHONPATH="$(pwd):${PYTHONPATH:-}"

# Ensure NCU permissions are set
echo "Step 1: Verify NCU permissions..."
if [ -f /proc/sys/kernel/perf_event_paranoid ]; then
    paranoid=$(cat /proc/sys/kernel/perf_event_paranoid)
    if [ "$paranoid" -gt 2 ]; then
        echo "⚠️  NCU may require elevated permissions"
        echo "   Current perf_event_paranoid: $paranoid"
        echo "   May need: sudo modprobe nvidia NVreg_RestrictProfilingToAdminUsers=0"
    else
        echo "✅ NCU permissions OK (perf_event_paranoid=$paranoid)"
    fi
else
    echo "⚠️  Cannot check perf_event_paranoid"
fi
echo ""

# Step 2: Quick metrics (fast)
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "STEP 2: Quick Metrics (Memory + Compute)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

# Use full path to NCU
NCU_BIN="/usr/local/cuda/bin/ncu"
if [ ! -f "$NCU_BIN" ]; then
    echo "❌ NCU not found at $NCU_BIN"
    echo "   Trying 'ncu' in PATH..."
    NCU_BIN="ncu"
fi

$NCU_BIN --metrics \
  gpu__time_duration.sum,\
  sm__throughput.avg.pct_of_peak_sustained_elapsed,\
  dram__throughput.avg.pct_of_peak_sustained_elapsed,\
  sm__pipe_tensor_op_hmma_cycles_active.avg.pct_of_peak_sustained_elapsed \
  --target-processes all \
  --kernel-name regex:".*" \
  --launch-count 1 \
  --csv \
  python - <<'PY' 2>&1 | tee evidence/ncu_champion_quick.log
import torch
from baselines import registry

# Get champion
champion = registry.get("pytorch_sdpa_efficient")

# Test data
q = torch.randn(1, 8, 512, 64, device='cuda', dtype=torch.float16)
k = torch.randn(1, 8, 512, 64, device='cuda', dtype=torch.float16)
v = torch.randn(1, 8, 512, 64, device='cuda', dtype=torch.float16)

# Warmup
for _ in range(5):
    _ = champion.fn(q, k, v, causal=False, dropout_p=0.0)

# Profile
torch.cuda.synchronize()
out = champion.fn(q, k, v, causal=False, dropout_p=0.0)
torch.cuda.synchronize()
PY

echo ""
echo "✅ Quick metrics saved: evidence/ncu_champion_quick.log"
echo ""

# Step 3: Full profile (comprehensive, slower)
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "STEP 3: Full Profile (Comprehensive Analysis)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "This may take 5-10 minutes..."
echo ""

$NCU_BIN --set full \
  --target-processes all \
  --kernel-name regex:".*" \
  --launch-count 1 \
  --force-overwrite \
  -o evidence/ncu_champion_full \
  python - <<'PY' 2>&1 | tee evidence/ncu_champion_full.log
import torch
from baselines import registry

# Get champion
champion = registry.get("pytorch_sdpa_efficient")

# Test data
q = torch.randn(1, 8, 512, 64, device='cuda', dtype=torch.float16)
k = torch.randn(1, 8, 512, 64, device='cuda', dtype=torch.float16)
v = torch.randn(1, 8, 512, 64, device='cuda', dtype=torch.float16)

# Warmup
for _ in range(5):
    _ = champion.fn(q, k, v, causal=False, dropout_p=0.0)

# Profile
torch.cuda.synchronize()
out = champion.fn(q, k, v, causal=False, dropout_p=0.0)
torch.cuda.synchronize()

print("✅ Profile complete")
PY

echo ""
echo "✅ Full profile saved:"
echo "   - evidence/ncu_champion_full.ncu-rep (binary, open in Nsight Compute UI)"
echo "   - evidence/ncu_champion_full.log (text)"
echo ""

# Step 4: Summary analysis
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "STEP 4: Summary Analysis"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

if [ -f evidence/ncu_champion_quick.log ]; then
    echo "Quick Metrics Summary:"
    echo ""
    grep -E "(Duration|Throughput|sm__|dram__|tensor)" evidence/ncu_champion_quick.log | head -20 || echo "  (see evidence/ncu_champion_quick.log for details)"
    echo ""
fi

echo "Full report: Open evidence/ncu_champion_full.ncu-rep in Nsight Compute UI"
echo ""

# Step 5: Key bottleneck indicators
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "STEP 5: Bottleneck Analysis Guide"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "Interpretation:"
echo ""
echo "1. MEMORY-BOUND (optimize memory access):"
echo "   - DRAM throughput > 60%"
echo "   - SM throughput < 40%"
echo "   → Solutions: coalescing, caching, prefetch, reduce DRAM"
echo ""
echo "2. COMPUTE-BOUND (optimize ALU/FPU):"
echo "   - SM throughput > 60%"
echo "   - DRAM throughput < 40%"
echo "   - Tensor Core < 10%"
echo "   → Solutions: increase arithmetic intensity, use TC"
echo ""
echo "3. TENSOR CORE UNDERUTILIZED:"
echo "   - Tensor Core activity < 50%"
echo "   - Problem: Not using TC or wrong data layout"
echo "   → Solutions: WMMA, ensure FP16/BF16, 16x16x16 tiles"
echo ""
echo "4. BALANCED (well-optimized):"
echo "   - SM throughput: 50-80%"
echo "   - DRAM throughput: 30-50%"
echo "   - Tensor Core: > 50% (if applicable)"
echo "   → Hard to improve further without major changes"
echo ""

echo "════════════════════════════════════════════════════════════════════════════════"
echo "NCU PROFILING COMPLETE"
echo "════════════════════════════════════════════════════════════════════════════════"
echo ""
echo "Evidence:"
echo "  - evidence/ncu_champion_quick.log (text summary)"
echo "  - evidence/ncu_champion_full.ncu-rep (full profile, open in NCU UI)"
echo "  - evidence/ncu_champion_full.log (text log)"
echo ""
echo "Next: Analyze bottlenecks → Plan targeted optimizations"
echo ""

