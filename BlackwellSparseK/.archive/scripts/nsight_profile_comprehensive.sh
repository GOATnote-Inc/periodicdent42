#!/usr/bin/env bash
# Comprehensive Nsight Compute profiling
# Full metrics set for deep performance analysis

set -e

METRICS_DIR="benchmarks/metrics"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
REPORT_FILE="${METRICS_DIR}/comprehensive_${TIMESTAMP}.ncu-rep"
JSON_FILE="${METRICS_DIR}/comprehensive_metrics.json"

mkdir -p "$METRICS_DIR"

echo "📊 Running COMPREHENSIVE Nsight Compute profiling..."
echo "Report: $REPORT_FILE"
echo ""

# Full metrics set (includes Tensor Core, FP8, memory hierarchy)
ncu \
  --target-processes all \
  --kernel-name-base mangled \
  --launch-skip 5 \
  --launch-count 50 \
  --set full \
  --metrics \
    sm__cycles_elapsed.avg,\
    sm__cycles_elapsed.sum,\
    gpu__time_duration.sum,\
    sm__warps_active.avg.pct_of_peak_sustained_active,\
    sm__sass_inst_executed_op_memory_128b.sum,\
    dram__bytes.sum,\
    dram__throughput.avg.pct_of_peak_sustained_elapsed,\
    l1tex__t_bytes_pipe_lsu_mem_global_op_ld.sum,\
    l1tex__t_bytes_pipe_lsu_mem_global_op_st.sum,\
    l2_tex_read_throughput.avg.pct_of_peak_sustained_elapsed,\
    l2_tex_write_throughput.avg.pct_of_peak_sustained_elapsed,\
    smsp__sass_thread_inst_executed_op_fadd_pred_on.sum,\
    smsp__sass_thread_inst_executed_op_fmul_pred_on.sum,\
    smsp__sass_thread_inst_executed_op_ffma_pred_on.sum,\
    smsp__sass_thread_inst_executed_op_dfma_pred_on.sum,\
    sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_active,\
    smsp__inst_executed_pipe_tensor.sum,\
    smsp__inst_executed_pipe_tensor_op_hmma.sum,\
    smsp__inst_executed_pipe_tensor_op_dmma.sum,\
    sm__sass_inst_executed_op_fp8_pred_on.sum,\
    derived__sm__sass_active_cycles_pct,\
    derived__memory_l1_wavefronts_shared,\
    derived__memory_l2_theoretical_sectors_global \
  --csv \
  --log-file "${METRICS_DIR}/comprehensive_${TIMESTAMP}.csv" \
  --export "$REPORT_FILE" \
  python benchmarks/perf.py --seq 4096 --heads 96 --device cuda --batch 1

echo ""
echo "📊 Parsing comprehensive metrics to JSON..."
python3 scripts/parse_ncu_metrics.py \
  --csv "${METRICS_DIR}/comprehensive_${TIMESTAMP}.csv" \
  --output "$JSON_FILE" \
  --timestamp "$TIMESTAMP" \
  --comprehensive

echo ""
echo "✅ Comprehensive profiling complete"
echo "📄 Metrics: $JSON_FILE"
echo "📄 Full report: $REPORT_FILE"
echo ""

# Print detailed summary
if [ -f "$JSON_FILE" ]; then
    python3 -c "
import json
with open('$JSON_FILE') as f:
    data = json.load(f)
    latest = data['runs'][-1] if 'runs' in data and data['runs'] else {}
    print('═' * 70)
    print('COMPREHENSIVE PERFORMANCE ANALYSIS')
    print('═' * 70)
    print()
    print('⏱️  TIMING')
    print(f\"   Kernel Duration:     {latest.get('kernel_duration_us', 'N/A')} μs\")
    print(f\"   GPU Time:            {latest.get('gpu_time_us', 'N/A')} μs\")
    print()
    print('🔢 COMPUTE')
    print(f\"   SM Efficiency:       {latest.get('sm_efficiency_pct', 'N/A')}%\")
    print(f\"   Warp Active:         {latest.get('warp_active_pct', 'N/A')}%\")
    print(f\"   Tensor Core Active:  {latest.get('tensor_core_active_pct', 'N/A')}%\")
    print(f\"   Tensor Ops:          {latest.get('tensor_ops_count', 'N/A')}\")
    print(f\"   FP8 Ops:             {latest.get('fp8_ops_count', 'N/A')}\")
    print(f\"   Compute TFLOPS:      {latest.get('compute_tflops', 'N/A')}\")
    print()
    print('💾 MEMORY')
    print(f\"   DRAM Throughput:     {latest.get('dram_throughput_pct', 'N/A')}% of peak\")
    print(f\"   DRAM Bytes:          {latest.get('dram_bytes', 'N/A')}\")
    print(f\"   Global Memory (GB):  {latest.get('global_memory_gb', 'N/A')}\")
    print(f\"   L2 Read Throughput:  {latest.get('l2_read_throughput_pct', 'N/A')}%\")
    print(f\"   L2 Write Throughput: {latest.get('l2_write_throughput_pct', 'N/A')}%\")
    print()
    print('═' * 70)
"
fi

echo ""
echo "💡 View in Nsight UI: nsight-compute $REPORT_FILE"
echo "💡 Compare metrics:   cat $JSON_FILE | jq '.runs[] | {timestamp, sm_efficiency_pct, tensor_core_active_pct}'"

