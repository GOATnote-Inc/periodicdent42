#!/bin/bash
# NCU profiling for hybrid attention (Phase B.4)

set -e

cd ~/periodicdent42
source ~/venv/bin/activate

echo "=" 70
echo "Phase B.4: NCU Profiling - Hybrid Attention"
echo "="*70
echo ""

# Create evidence directory
mkdir -p evidence

# Profile hybrid attention
echo "Profiling hybrid attention (cuBLAS + softmax + matmul)..."
echo ""

ncu --target-processes all --replay-mode kernel \
  --metrics sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_elapsed,\
sm__throughput.avg.pct_of_peak_sustained_elapsed,\
sm__warps_active.avg.pct_of_peak_sustained_active,\
dram__throughput.avg.pct_of_peak_sustained_elapsed,\
sm__inst_executed.avg.pct_of_peak_sustained_active \
  --csv \
  --export evidence/ncu_hybrid \
  python bench/test_hybrid_attention.py

echo ""
echo "✅ NCU profiling complete"
echo ""
echo "Results saved to:"
echo "  - evidence/ncu_hybrid.ncu-rep (full report)"
echo "  - evidence/ncu_hybrid.csv (metrics)"
echo ""

# Extract key metrics
if [ -f evidence/ncu_hybrid.csv ]; then
    echo "Key Metrics:"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    
    # Tensor Core utilization
    TC=$(grep "sm__pipe_tensor_cycles_active" evidence/ncu_hybrid.csv | tail -1 | cut -d',' -f2 | sed 's/[^0-9.]//g' || echo "N/A")
    echo "  Tensor Core Activity: ${TC}%"
    
    # SM throughput
    SM=$(grep "sm__throughput" evidence/ncu_hybrid.csv | tail -1 | cut -d',' -f2 | sed 's/[^0-9.]//g' || echo "N/A")
    echo "  SM Throughput: ${SM}%"
    
    # Warp occupancy
    WARP=$(grep "sm__warps_active" evidence/ncu_hybrid.csv | tail -1 | cut -d',' -f2 | sed 's/[^0-9.]//g' || echo "N/A")
    echo "  Warp Occupancy: ${WARP}%"
    
    # DRAM throughput
    DRAM=$(grep "dram__throughput" evidence/ncu_hybrid.csv | tail -1 | cut -d',' -f2 | sed 's/[^0-9.]//g' || echo "N/A")
    echo "  DRAM Throughput: ${DRAM}%"
    
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo ""
    
    echo "Analysis:"
    echo "  - TC Activity > 50%: Good Tensor Core utilization ✅"
    echo "  - TC Activity < 50%: More TC optimization needed ⚠️"
    echo "  - SM Throughput > 70%: Compute-bound (good) ✅"
    echo "  - DRAM < 10%: Memory-efficient ✅"
    echo ""
fi

echo "Next: Analyze results and plan Phase C optimizations"

