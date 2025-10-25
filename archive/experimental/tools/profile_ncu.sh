#!/bin/bash
#
# NCU Profiling Script for FP8 Stage C WMMA Kernel (EvoEngineer Framework)
# ===========================================================================
#
# This script runs Nsight Compute (NCU) profiling on the FP8 Stage C kernel
# for specified shapes, collecting key metrics for compute-bound vs memory-bound
# analysis.
#
# Usage:
#   ./tools/profile_ncu.sh [SHAPE] [ITERS]
#
# Example:
#   ./tools/profile_ncu.sh mission 100
#   ./tools/profile_ncu.sh long 50
#
# Requirements:
#   - NVIDIA L4 GPU (or compatible Ada/Ampere)
#   - CUDA 12.x with NCU installed
#   - Sudo access (NCU requires elevated permissions for some metrics)
#

set -e

# Configuration
SHAPE=${1:-mission}
ITERS=${2:-100}
WARMUP=20
OUTPUT_DIR="./runs"
PYTHON_CMD="${HOME}/venv/bin/python3"

# Ensure output directory exists
mkdir -p "${OUTPUT_DIR}"

# NCU metrics to collect (EvoEngineer Priority 1)
# These metrics tell us:
#   - sm__pipe_tensor_active: Are Tensor Cores being used?
#   - dram__throughput: Are we memory-bound?
#   - smsp__warps_active: What's our occupancy?
#   - l1tex__t_bytes: L1 cache traffic
#   - lts__t_bytes: L2 cache traffic
METRICS="sm__pipe_tensor_active.avg.pct_of_peak_sustained_active,\
dram__throughput.avg.pct_of_peak_sustained_elapsed,\
smsp__warps_active.avg.pct_of_peak_sustained_active,\
l1tex__t_bytes.sum.per_second,\
lts__t_bytes.sum.per_second,\
smsp__inst_executed_pipe_tensor.sum,\
smsp__inst_executed_pipe_fma.sum,\
l1tex__data_bank_conflicts_pipe_lsu_mem_shared.sum"

# Output files
CSV_OUT="${OUTPUT_DIR}/ncu_${SHAPE}.csv"
LOG_OUT="${OUTPUT_DIR}/ncu_${SHAPE}.log"

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "🔍 NCU Profiling for FP8 Stage C WMMA Kernel (EvoEngineer Framework)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  Shape:          $SHAPE"
echo "  Iterations:     $ITERS (warmup: $WARMUP)"
echo "  Output CSV:     $CSV_OUT"
echo "  Output Log:     $LOG_OUT"
echo ""

# Check if ncu is available
if ! command -v ncu &> /dev/null; then
    echo "❌ Error: 'ncu' (Nsight Compute) not found in PATH"
    echo "   Please install CUDA Toolkit with NSight Compute"
    echo "   Typical location: /usr/local/cuda/bin/ncu"
    exit 1
fi

# Check if Python script exists
BENCH_SCRIPT="scripts/bench_fp8_stage_c.py"
if [[ ! -f "$BENCH_SCRIPT" ]]; then
    echo "❌ Error: Benchmark script not found: $BENCH_SCRIPT"
    exit 1
fi

# Check for sudo (NCU requires elevated permissions for some metrics)
echo "⚠️  Note: NCU profiling may require sudo for some metrics"
echo "   If profiling fails, try running with sudo: sudo $0 $SHAPE $ITERS"
echo ""

# Run NCU profiling
echo "🚀 Running NCU profiling..."
echo "   (This may take several minutes...)"
echo ""

# NCU command
# - --target-processes all: Profile all processes (handle multi-process cases)
# - --replay-mode kernel: Replay kernels multiple times for accurate metrics
# - --set full: Full metric set (detailed)
# - --kernel-name regex: Only profile our FP8 Stage C kernel
# - --csv: Output in CSV format for easy parsing
ncu --target-processes all \
    --replay-mode kernel \
    --set full \
    --kernel-name regex:sdpa_fp8_stage_c_wmma.* \
    --metrics "$METRICS" \
    --csv \
    --log-file "$LOG_OUT" \
    --export "$CSV_OUT" \
    "$PYTHON_CMD" "$BENCH_SCRIPT" \
        --shapes "$SHAPE" \
        --iters "$ITERS" \
        --warmup "$WARMUP" \
        --skip-correctness

echo ""
echo "✅ NCU profiling complete!"
echo ""
echo "📊 Results:"
echo "   CSV:  $CSV_OUT"
echo "   Log:  $LOG_OUT"
echo ""
echo "🔍 Next steps:"
echo "   1. Parse CSV with: python tools/parse_ncu_results.py $CSV_OUT"
echo "   2. Check for:"
echo "      - sm__pipe_tensor_active > 50% → compute-bound (good!)"
echo "      - dram__throughput < 70% → not memory-bound (good!)"
echo "      - smsp__warps_active > 40% → good occupancy"
echo ""

