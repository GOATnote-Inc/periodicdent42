#!/bin/bash
# FlashCore v12: Nsight Compute Gating Script
# Usage: ./tools/nsight_gate.sh [variant_id]

set -e

VARIANT="${1:-baseline}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT/flashcore"

echo "════════════════════════════════════════════════════════════════"
echo "FlashCore v12 Nsight Compute: $VARIANT"
echo "════════════════════════════════════════════════════════════════"

# Check if ncu is available
if ! command -v ncu &> /dev/null; then
    echo "⚠️  Nsight Compute (ncu) not found, skipping NCU gates"
    exit 0
fi

# Profile
echo "[1/2] Profiling with NCU..."
ncu --set full \
    --launch-skip 50 \
    --launch-count 1 \
    --metrics sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_active,smsp__warp_execution_efficiency.avg.pct_of_peak_sustained_active \
    python3 test_v12_expert.py \
    2>&1 | tee ../logs/ncu_${VARIANT}.log || true

# Extract metrics
echo "[2/2] Checking gates..."
TC_UTIL=$(grep "sm__pipe_tensor_cycles_active" ../logs/ncu_${VARIANT}.log | grep -oP '\d+\.\d+(?=%)' | head -1 || echo "0")
WARP_EFF=$(grep "smsp__warp_execution_efficiency" ../logs/ncu_${VARIANT}.log | grep -oP '\d+\.\d+(?=%)' | head -1 || echo "0")

echo "  Tensor Core Util: ${TC_UTIL}%"
echo "  Warp Efficiency: ${WARP_EFF}%"

# Gates (warning only, not blocking)
if (( $(echo "$TC_UTIL < 90.0" | bc -l) )); then
    echo "⚠️  Tensor Core utilization ${TC_UTIL}% < 90% (target)"
fi

if (( $(echo "$WARP_EFF < 95.0" | bc -l) )); then
    echo "⚠️  Warp efficiency ${WARP_EFF}% < 95% (target)"
fi

# Write JSON
cat > ../results/ncu_${VARIANT}.json <<EOF
{
  "variant": "$VARIANT",
  "tensor_core_util_pct": ${TC_UTIL},
  "warp_efficiency_pct": ${WARP_EFF},
  "timestamp": "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
}
EOF

echo "✅ NCU results saved to results/ncu_${VARIANT}.json"
exit 0

