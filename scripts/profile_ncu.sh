#!/usr/bin/env bash
#
# NCU one-touch profiling for FP8 SDPA Stage-C WMMA kernel
#
# Usage:
#   scripts/profile_ncu.sh small
#   scripts/profile_ncu.sh mission
#

set -euo pipefail

# Parse arguments
SHAPE="${1:-mission}"
SEED="${2:-0}"
ITERS="${3:-3}"

# Output directory
TIMESTAMP=$(date +%Y%m%d-%H%M%S)
OUT_DIR="results/fp8_wmma_baseline/${TIMESTAMP}-${SHAPE}-ncu"
mkdir -p "$OUT_DIR"

echo ""
echo "================================================================================"
echo "NCU Profiling: FP8 SDPA Stage-C WMMA"
echo "================================================================================"
echo "  Shape:      $SHAPE"
echo "  Seed:       $SEED"
echo "  Iterations: $ITERS"
echo "  Output:     $OUT_DIR"
echo "================================================================================"
echo ""

# NCU metrics (Tensor Core focus)
METRICS="sm__warps_active.avg.pct_of_peak_sustained_active,\
smsp__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_active,\
smsp__inst_executed_pipe_tensor.sum,\
smsp__sass_thread_inst_executed_op_hmma_pred_on.sum,\
l1tex__t_bytes.sum,\
lts__t_bytes.sum,\
dram__bytes.sum,\
smsp__sass_average_data_bytes_per_sector_mem_global_load.pct,\
sm__throughput.avg.pct_of_peak_sustained_elapsed,\
smsp__sass_inst_executed_op_memory_32bit.sum,\
smsp__sass_inst_executed_op_memory_64bit.sum,\
smsp__sass_inst_executed_op_memory_128bit.sum"

# Run NCU
ncu --target-processes all \
    --set full \
    --metrics "$METRICS" \
    --export "$OUT_DIR/profile" \
    --profile-from-start off \
    --force-overwrite \
    python -m tasks.fp8_sdpa_stage_c_wmma.runner \
      --shapes "$SHAPE" \
      --seeds "$SEED" \
      --iters "$ITERS" \
      --no-build

# Save metadata
echo "$OUT_DIR/profile.ncu-rep" > "$OUT_DIR/profile_path.txt"

echo ""
echo "================================================================================"
echo "✅ Profiling complete!"
echo "================================================================================"
echo "  Report: $OUT_DIR/profile.ncu-rep"
echo "  View:   ncu-ui $OUT_DIR/profile.ncu-rep"
echo "================================================================================"
echo ""
