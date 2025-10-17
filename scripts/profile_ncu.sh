#!/bin/bash
# Nsight Compute profiling wrapper with standard metrics
# Usage: ./scripts/profile_ncu.sh <kernel_binary> <output_name>

set -e

export PATH="/usr/local/cuda/bin:$PATH"

KERNEL_CMD="${1:-python bench/run_attn.py --shape S512_D64}"
OUTPUT_NAME="${2:-ncu_profile}"
EVIDENCE_DIR="evidence"

mkdir -p "$EVIDENCE_DIR"

echo "🔬 Profiling with Nsight Compute..."
echo "   Command: $KERNEL_CMD"
echo "   Output: $EVIDENCE_DIR/${OUTPUT_NAME}.csv"
echo ""

# Brief, fast metrics (minimal overhead)
ncu --target-processes all \
  --replay-mode kernel \
  --metrics \
sm__warps_active.avg.pct_of_peak_sustained_active,\
sm__pipe_tensor_active.avg.pct_of_peak_sustained_active,\
dram__throughput.avg.pct_of_peak_sustained_elapsed,\
sm__throughput.avg.pct_of_peak_sustained_elapsed,\
l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum,\
l1tex__t_sectors_pipe_lsu_mem_global_op_st.sum \
  --csv \
  --log-file "$EVIDENCE_DIR/${OUTPUT_NAME}.txt" \
  --export "$EVIDENCE_DIR/${OUTPUT_NAME}" \
  $KERNEL_CMD

echo ""
echo "✅ Profile saved to: $EVIDENCE_DIR/${OUTPUT_NAME}.*"
echo ""
echo "📊 Key Metrics:"
echo "   - sm__warps_active: Warp occupancy (target >60%)"
echo "   - sm__pipe_tensor_active: Tensor Core utilization"
echo "   - dram__throughput: Memory bandwidth usage"
echo "   - sm__throughput: Compute utilization"

