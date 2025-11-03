#!/bin/bash
# NCU Validation Script (Burn Methodology)
# Based on BlackwellSparseK iterations 0-9

set -e

KERNEL=${1:-"baseline"}
MODE=${2:-"quick"}

echo "════════════════════════════════════════════════════════════════════════════════"
echo "  DHP NCU VALIDATION (Burn Methodology)"
echo "════════════════════════════════════════════════════════════════════════════════"
echo ""
echo "Kernel: $KERNEL"
echo "Mode:   $MODE"
echo ""

# Metrics from our burn iterations
NCU_METRICS="gpu__time_duration.sum,\
sm__throughput.avg.pct_of_peak_sustained_elapsed,\
dram__bytes_read.sum,\
dram__bytes_write.sum,\
smsp__sass_thread_inst_executed.sum"

if [ "$MODE" = "full" ]; then
    # Add detailed metrics for deep analysis
    NCU_METRICS+=",\
sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_active,\
l2_tex_read_hit_rate.pct,\
launch__registers_per_thread"
fi

# Run NCU (requires sudo for performance counters)
echo "Running NCU profiling..."
echo ""

sudo /usr/local/cuda-13.0/bin/ncu \
    --metrics "$NCU_METRICS" \
    --target-processes all \
    --csv \
    --log-file "audits/${KERNEL}_ncu.csv" \
    python3 "benchmarks/bench_${KERNEL}.py" 2>&1 | \
    grep -E "(gpu__time|sm__throughput|dram__|sass_thread|tensor_cycles|registers)" | \
    tail -30

echo ""
echo "════════════════════════════════════════════════════════════════════════════════"
echo "  NCU Profile saved to: audits/${KERNEL}_ncu.csv"
echo "════════════════════════════════════════════════════════════════════════════════"

