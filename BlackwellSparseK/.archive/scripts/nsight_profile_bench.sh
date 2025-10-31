#!/usr/bin/env bash
# Quick Nsight Compute profiling for make bench
# Exports metrics to benchmarks/metrics/bench_metrics.json

set -e

METRICS_DIR="benchmarks/metrics"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
REPORT_FILE="${METRICS_DIR}/bench_${TIMESTAMP}.ncu-rep"
JSON_FILE="${METRICS_DIR}/bench_metrics.json"

mkdir -p "$METRICS_DIR"

echo "⚡ Running BlackwellSparseK benchmark with Nsight profiling..."
echo "Report: $REPORT_FILE"

# Run Nsight Compute with key metrics
ncu \
  --target-processes all \
  --kernel-name-base mangled \
  --launch-skip 10 \
  --launch-count 100 \
  --metrics \
    sm__cycles_elapsed.avg,\
    sm__cycles_elapsed.sum,\
    gpu__time_duration.sum,\
    sm__warps_active.avg.pct_of_peak_sustained_active,\
    dram__bytes.sum,\
    dram__throughput.avg.pct_of_peak_sustained_elapsed,\
    l1tex__t_bytes_pipe_lsu_mem_global_op_ld.sum,\
    l1tex__t_bytes_pipe_lsu_mem_global_op_st.sum,\
    smsp__sass_thread_inst_executed_op_fadd_pred_on.sum,\
    smsp__sass_thread_inst_executed_op_fmul_pred_on.sum,\
    smsp__sass_thread_inst_executed_op_ffma_pred_on.sum,\
    sm__sass_inst_executed_op_memory_128b.sum \
  --csv \
  --log-file "${METRICS_DIR}/bench_${TIMESTAMP}.csv" \
  --export "$REPORT_FILE" \
  python benchmarks/perf.py --seq 4096 --heads 96 --device cuda --batch 1

echo ""
echo "📊 Parsing metrics to JSON..."
python3 scripts/parse_ncu_metrics.py \
  --csv "${METRICS_DIR}/bench_${TIMESTAMP}.csv" \
  --output "$JSON_FILE" \
  --timestamp "$TIMESTAMP"

echo ""
echo "✅ Profiling complete"
echo "📄 Metrics: $JSON_FILE"
echo ""

# Print summary
if [ -f "$JSON_FILE" ]; then
    python3 -c "
import json
with open('$JSON_FILE') as f:
    data = json.load(f)
    latest = data['runs'][-1] if 'runs' in data and data['runs'] else {}
    print('═' * 60)
    print('PERFORMANCE SUMMARY')
    print('═' * 60)
    print(f\"Kernel Duration:     {latest.get('kernel_duration_us', 'N/A')} μs\")
    print(f\"SM Efficiency:       {latest.get('sm_efficiency_pct', 'N/A')}%\")
    print(f\"DRAM Throughput:     {latest.get('dram_throughput_pct', 'N/A')}% of peak\")
    print(f\"Global Memory (GB):  {latest.get('global_memory_gb', 'N/A')}\")
    print(f\"Compute TFLOPS:      {latest.get('compute_tflops', 'N/A')}\")
    print('═' * 60)
"
fi

echo ""
echo "💡 View full report: nsight-compute $REPORT_FILE"

