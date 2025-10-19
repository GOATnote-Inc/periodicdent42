#!/bin/bash
# NCU Profiling Script for V2c-v7a Phase 1 Analysis
# Goal: Understand why cp.async overlap provided minimal speedup (1.01× vs expected 1.3-1.7×)

set -e

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "NCU PROFILING: V2c-v7a Phase 1 Analysis"
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
echo "📊 Profile 1: Tensor Core & Memory Utilization"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# Profile 1: Core utilization metrics
ncu --set full \
    --target-processes all \
    --kernel-name "sdpa_fused_v2c_v7a_kernel" \
    --metrics sm__pipe_tensor_active.avg.pct_of_peak_sustained_active \
    --metrics sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_active \
    --metrics dram__throughput.avg.pct_of_peak_sustained_elapsed \
    --metrics dram__bytes.sum.per_second \
    --metrics l1tex__t_bytes_pipe_lsu_mem_global_op_ld.sum.per_second \
    --metrics l1tex__t_bytes_pipe_lsu_mem_global_op_st.sum.per_second \
    --metrics smsp__warps_active.avg.pct_of_peak_sustained_active \
    --metrics sm__cycles_elapsed.avg \
    --metrics sm__cycles_elapsed.sum \
    --csv \
    python3 -c "
from bench_sdpa import build_ext, run_case
import torch
mod = build_ext()
# Profile mission shape only
run_case(mod, B=1, H=8, L=512, d=64, causal=False, verbose=False, iters=1)
" > "${OUTPUT_DIR}/profile1_utilization_${TIMESTAMP}.csv" 2>&1

echo "✅ Profile 1 complete"

echo ""
echo "📊 Profile 2: cp.async & Memory Pipeline"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# Profile 2: Async copy metrics
ncu --set full \
    --target-processes all \
    --kernel-name "sdpa_fused_v2c_v7a_kernel" \
    --metrics smsp__inst_executed_op_cp_async.sum \
    --metrics smsp__inst_executed_op_ldgsts.sum \
    --metrics smsp__sass_thread_inst_executed_op_memory_128b.sum \
    --metrics smsp__sass_thread_inst_executed_op_memory_64b.sum \
    --metrics smsp__sass_thread_inst_executed_op_memory_32b.sum \
    --metrics lts__t_sectors_srcunit_tex_op_read.sum \
    --metrics lts__t_sectors_srcunit_tex_op_write.sum \
    --csv \
    python3 -c "
from bench_sdpa import build_ext, run_case
import torch
mod = build_ext()
run_case(mod, B=1, H=8, L=512, d=64, causal=False, verbose=False, iters=1)
" > "${OUTPUT_DIR}/profile2_async_${TIMESTAMP}.csv" 2>&1

echo "✅ Profile 2 complete"

echo ""
echo "📊 Profile 3: SMEM Bank Conflicts & Occupancy"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# Profile 3: SMEM and occupancy
ncu --set full \
    --target-processes all \
    --kernel-name "sdpa_fused_v2c_v7a_kernel" \
    --metrics l1tex__data_bank_conflicts_pipe_lsu_mem_shared.sum \
    --metrics l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum \
    --metrics l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_st.sum \
    --metrics sm__maximum_warps_per_active_cycle_pct \
    --metrics launch__occupancy_limit_blocks \
    --metrics launch__occupancy_limit_registers \
    --metrics launch__occupancy_limit_shared_mem \
    --metrics launch__registers_per_thread \
    --metrics launch__shared_mem_per_block_static \
    --metrics launch__shared_mem_per_block_dynamic \
    --csv \
    python3 -c "
from bench_sdpa import build_ext, run_case
import torch
mod = build_ext()
run_case(mod, B=1, H=8, L=512, d=64, causal=False, verbose=False, iters=1)
" > "${OUTPUT_DIR}/profile3_smem_occupancy_${TIMESTAMP}.csv" 2>&1

echo "✅ Profile 3 complete"

echo ""
echo "📊 Profile 4: Stall Analysis"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# Profile 4: Stall reasons
ncu --set full \
    --target-processes all \
    --kernel-name "sdpa_fused_v2c_v7a_kernel" \
    --metrics smsp__warp_issue_stalled_math_pipe_throttle_per_warp_active.pct \
    --metrics smsp__warp_issue_stalled_mio_throttle_per_warp_active.pct \
    --metrics smsp__warp_issue_stalled_long_scoreboard_per_warp_active.pct \
    --metrics smsp__warp_issue_stalled_short_scoreboard_per_warp_active.pct \
    --metrics smsp__warp_issue_stalled_wait_per_warp_active.pct \
    --metrics smsp__warp_issue_stalled_barrier_per_warp_active.pct \
    --metrics smsp__warp_issue_stalled_drain_per_warp_active.pct \
    --csv \
    python3 -c "
from bench_sdpa import build_ext, run_case
import torch
mod = build_ext()
run_case(mod, B=1, H=8, L=512, d=64, causal=False, verbose=False, iters=1)
" > "${OUTPUT_DIR}/profile4_stalls_${TIMESTAMP}.csv" 2>&1

echo "✅ Profile 4 complete"

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "✅ NCU Profiling Complete"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "Results saved to: ${OUTPUT_DIR}/"
echo "Timestamp: ${TIMESTAMP}"
echo ""
echo "Next: Run parse_ncu_results.py to analyze and extract I3 insights"

