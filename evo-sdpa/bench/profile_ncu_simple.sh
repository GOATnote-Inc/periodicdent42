#!/bin/bash
# Simplified NCU Profiling Script for V2c-v7a - Essential Metrics Only
# Goal: Quick analysis of why cp.async overlap didn't help

set -e

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "NCU PROFILING: V2c-v7a Essential Metrics (FAST)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# Activate venv
source ~/venv/bin/activate

# Navigate to evo-sdpa
cd ~/periodicdent42/evo-sdpa

# Output directory
mkdir -p ncu_results
OUTPUT_DIR="ncu_results"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

echo ""
echo "📊 Profiling V2c-v7a kernel (mission shape: 1,8,512,64)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# Run with essential metrics only (much faster than --set full)
sudo /usr/local/cuda/bin/ncu \
    --target-processes all \
    --kernel-name-base function \
    --kernel-name regex:"sdpa_fused_v2c_v7a" \
    --metrics sm__pipe_tensor_active.avg.pct_of_peak_sustained_active \
    --metrics dram__throughput.avg.pct_of_peak_sustained_elapsed \
    --metrics smsp__inst_executed_op_cp_async.sum \
    --metrics l1tex__data_bank_conflicts_pipe_lsu_mem_shared.sum \
    --metrics smsp__warps_active.avg.pct_of_peak_sustained_active \
    --metrics launch__registers_per_thread \
    --metrics launch__shared_mem_per_block_dynamic \
    --metrics smsp__warp_issue_stalled_long_scoreboard_per_warp_active.pct \
    --metrics smsp__warp_issue_stalled_barrier_per_warp_active.pct \
    --csv \
    python3 -c "
from bench.bench_sdpa import build_ext, run_case
import torch
print('[Building kernel...]')
mod = build_ext()
print('[Running kernel (1,8,512,64)...]')
run_case(mod, B=1, H=8, L=512, d=64, causal=False, verbose=False, iters=1)
print('[Complete]')
" > "${OUTPUT_DIR}/ncu_essential_${TIMESTAMP}.csv" 2>&1

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "✅ NCU Profiling Complete"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "Results saved to: ${OUTPUT_DIR}/ncu_essential_${TIMESTAMP}.csv"
echo ""
echo "To view results:"
echo "  python3 bench/parse_ncu_simple.py"

