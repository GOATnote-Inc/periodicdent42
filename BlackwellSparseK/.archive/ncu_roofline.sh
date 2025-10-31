#!/bin/bash
# ============================================================================
# Nsight Compute Roofline Analysis for BlackwellSparseK
# ============================================================================
# Profiles kernel with Nsight Compute and generates roofline chart.
# Requires: Nsight Compute 2025.3.0+, SYS_ADMIN capability
# ============================================================================

set -e

# Configuration
KERNEL_NAME="fmha_kernel_impl"
OUTPUT_DIR="results/ncu"
CONFIG="B=1,H=8,S=512,D=64"

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Timestamp
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT_FILE="$OUTPUT_DIR/ncu_${TIMESTAMP}.ncu-rep"

echo "============================================"
echo "BlackwellSparseK Nsight Compute Profile"
echo "============================================"
echo "Kernel: $KERNEL_NAME"
echo "Config: $CONFIG"
echo "Output: $OUTPUT_FILE"
echo "============================================"

# Run Nsight Compute profiling
ncu \
    --set full \
    --kernel-name "$KERNEL_NAME" \
    --launch-skip 10 \
    --launch-count 1 \
    --target-processes all \
    --export "$OUTPUT_FILE" \
    python3 benchmarks/perf.py

echo ""
echo "✅ Profiling complete!"
echo "Report: $OUTPUT_FILE"
echo ""
echo "View with: ncu-ui $OUTPUT_FILE"
echo ""

# Generate text summary
ncu --import "$OUTPUT_FILE" --page details > "${OUTPUT_FILE%.ncu-rep}_summary.txt"
echo "Text summary: ${OUTPUT_FILE%.ncu-rep}_summary.txt"

# Extract key metrics
echo ""
echo "KEY METRICS:"
echo "============================================"
ncu --import "$OUTPUT_FILE" --query "gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed" --csv | tail -n 1
ncu --import "$OUTPUT_FILE" --query "sm__throughput.avg.pct_of_peak_sustained_elapsed" --csv | tail -n 1
ncu --import "$OUTPUT_FILE" --query "gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed" --csv | tail -n 1
echo "============================================"

