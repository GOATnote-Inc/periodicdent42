#!/usr/bin/env bash
# Profile BlackwellSparseK vs FlashAttention-3 comparison
# Exports side-by-side metrics for regression tracking

set -e

METRICS_DIR="benchmarks/metrics"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
REPORT_SPARSEK="${METRICS_DIR}/sparsek_${TIMESTAMP}.ncu-rep"
REPORT_FA3="${METRICS_DIR}/fa3_${TIMESTAMP}.ncu-rep"
JSON_FILE="${METRICS_DIR}/fa3_comparison.json"

mkdir -p "$METRICS_DIR"

echo "📊 Profiling BlackwellSparseK vs FlashAttention-3..."
echo ""

# Profile BlackwellSparseK
echo "⚡ [1/2] Profiling BlackwellSparseK..."
ncu \
  --target-processes all \
  --kernel-name regex:"attention|fmha" \
  --launch-skip 10 \
  --launch-count 100 \
  --metrics \
    gpu__time_duration.sum,\
    sm__warps_active.avg.pct_of_peak_sustained_active,\
    dram__throughput.avg.pct_of_peak_sustained_elapsed,\
    sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_active,\
    smsp__inst_executed_pipe_tensor.sum \
  --csv \
  --log-file "${METRICS_DIR}/sparsek_${TIMESTAMP}.csv" \
  --export "$REPORT_SPARSEK" \
  python -c "
import torch
from blackwell_sparsek import attention_forward

B, H, S, D = 1, 96, 4096, 64
Q = torch.randn(B, H, S, D, device='cuda', dtype=torch.float16)
K = torch.randn(B, H, S, D, device='cuda', dtype=torch.float16)
V = torch.randn(B, H, S, D, device='cuda', dtype=torch.float16)

for _ in range(100):
    out = attention_forward(Q, K, V)
    torch.cuda.synchronize()
"

echo ""
echo "⚡ [2/2] Profiling FlashAttention-3..."
ncu \
  --target-processes all \
  --kernel-name regex:"flash_fwd|flash_attn" \
  --launch-skip 10 \
  --launch-count 100 \
  --metrics \
    gpu__time_duration.sum,\
    sm__warps_active.avg.pct_of_peak_sustained_active,\
    dram__throughput.avg.pct_of_peak_sustained_elapsed,\
    sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_active,\
    smsp__inst_executed_pipe_tensor.sum \
  --csv \
  --log-file "${METRICS_DIR}/fa3_${TIMESTAMP}.csv" \
  --export "$REPORT_FA3" \
  python -c "
import torch
try:
    from flash_attn import flash_attn_func
    
    B, H, S, D = 1, 96, 4096, 64
    Q = torch.randn(B, S, H, D, device='cuda', dtype=torch.float16)
    K = torch.randn(B, S, H, D, device='cuda', dtype=torch.float16)
    V = torch.randn(B, S, H, D, device='cuda', dtype=torch.float16)
    
    for _ in range(100):
        out = flash_attn_func(Q, K, V, causal=False)
        torch.cuda.synchronize()
except ImportError:
    print('FlashAttention-3 not installed, skipping...')
"

echo ""
echo "📊 Comparing metrics..."
python3 scripts/compare_fa3_metrics.py \
  --sparsek-csv "${METRICS_DIR}/sparsek_${TIMESTAMP}.csv" \
  --fa3-csv "${METRICS_DIR}/fa3_${TIMESTAMP}.csv" \
  --output "$JSON_FILE" \
  --timestamp "$TIMESTAMP"

echo ""
echo "✅ Comparison profiling complete"
echo "📄 Comparison: $JSON_FILE"
echo ""

# Print comparison table
if [ -f "$JSON_FILE" ]; then
    python3 -c "
import json
with open('$JSON_FILE') as f:
    data = json.load(f)
    comparison = data.get('comparison', {})
    
    print('═' * 80)
    print('BLACKWELLSPARSEK vs FLASHATTENTION-3 COMPARISON')
    print('═' * 80)
    print()
    print(f\"{'Metric':<30} {'SparseK':>15} {'FA3':>15} {'Speedup':>15}\")
    print('─' * 80)
    
    metrics = [
        ('Kernel Duration (μs)', 'kernel_duration_us'),
        ('SM Efficiency (%)', 'sm_efficiency_pct'),
        ('DRAM Throughput (%)', 'dram_throughput_pct'),
        ('Tensor Core Active (%)', 'tensor_core_active_pct'),
        ('Tensor Ops', 'tensor_ops_count'),
    ]
    
    for label, key in metrics:
        sparsek_val = comparison.get('sparsek', {}).get(key, 'N/A')
        fa3_val = comparison.get('fa3', {}).get(key, 'N/A')
        
        if key == 'kernel_duration_us' and isinstance(sparsek_val, (int, float)) and isinstance(fa3_val, (int, float)):
            speedup = f\"{fa3_val / sparsek_val:.2f}x\"
        else:
            speedup = '-'
        
        print(f\"{label:<30} {str(sparsek_val):>15} {str(fa3_val):>15} {speedup:>15}\")
    
    print('═' * 80)
    
    verdict = comparison.get('verdict', 'Unknown')
    print(f\"\\n📊 Verdict: {verdict}\")
"
fi

echo ""
echo "💡 View reports:"
echo "   SparseK: nsight-compute $REPORT_SPARSEK"
echo "   FA3:     nsight-compute $REPORT_FA3"

