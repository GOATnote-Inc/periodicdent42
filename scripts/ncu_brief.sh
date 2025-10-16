#!/usr/bin/env bash
# Nsight Compute Brief Metrics - Fast & Robust
# Usage: scripts/ncu_brief.sh <python_script_or_binary> [args...]

set -euo pipefail

# Run Nsight Compute with minimal, high-value metrics
# Target: < 30s overhead per kernel
ncu --target-processes=all \
    --replay-mode kernel \
    --set full \
    --metrics \
sm__warps_active.avg.pct_of_peak_sustained_active,\
sm__pipe_tensor_active.avg.pct_of_peak_sustained_active,\
dram__throughput.avg.pct_of_peak_sustained_elapsed \
    --timeout 30 \
    --csv \
    --page raw \
    "$@" 2>&1 || {
    echo "WARNING: Nsight Compute failed or timed out" >&2
    exit 0  # Don't fail the sweep on NCU timeout
}

