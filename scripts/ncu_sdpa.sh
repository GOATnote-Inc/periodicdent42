#!/usr/bin/env bash
#
# Stage-5 NCU Profiling Script
# =============================
# One-click profiling with key metrics for compute-bound diagnosis
#

set -euo pipefail

echo "======================================================================"
echo "Stage-5 NCU Profiling"
echo "======================================================================"

# Ensure we're on L4
if ! command -v ncu &> /dev/null; then
    echo "❌ ncu not found. Run this on L4 GPU instance."
    exit 1
fi

# Warmup run (silent)
echo "Warming up kernel..."
python scripts/bench_sdpa.py --iters 20 --warmup 10 --shapes mission >/dev/null 2>&1 || true

# NCU profiling with key metrics
echo ""
echo "Running NCU profiling (60 iters)..."
ncu --target-processes all \
    --metrics sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_active,\
smsp__inst_executed_pipe_tensor.sum,\
smsp__inst_executed_pipe_fma.sum,\
dram__throughput.avg.pct_of_peak_sustained_elapsed,\
smsp__cycles_active.sum,\
sm__warps_active.avg.pct_of_peak_sustained_active \
    --section SpeedOfLight \
    --set full \
    --export kbench/ncu_stage5 \
    --force-overwrite \
    python scripts/bench_sdpa.py --iters 60 --warmup 10 --shapes mission

echo ""
echo "======================================================================"
echo "NCU Results saved to: kbench/ncu_stage5.ncu-rep"
echo "======================================================================"
echo ""
echo "Key Metrics to Check:"
echo "  - sm__pipe_tensor_cycles_active >= 50%  → Tensor Cores busy (compute-bound)"
echo "  - dram__throughput < 50% peak         → Memory not saturated"
echo "  - smsp__cycles_active                  → Total active cycles"
echo ""
echo "If both conditions met: kernel is compute-bound (Stage-4 hypothesis confirmed)"
echo "If DRAM throughput high: memory is bottleneck (revisit cp.async pipeline)"

