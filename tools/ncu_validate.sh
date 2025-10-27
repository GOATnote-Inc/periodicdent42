#!/bin/bash
# NSight Compute Validation Script
# Measures: DRAM bandwidth, SM throughput, roofline metrics
# Target: DRAM ≥85%, SM ≥70% (FA3 baseline)

set -e

BINARY="${1:-./build/bin/test_hopper}"
OUTPUT_DIR="build/ncu_reports"

mkdir -p "$OUTPUT_DIR"

echo "========================================"
echo "NSIGHT COMPUTE VALIDATION"
echo "========================================"
echo ""
echo "Binary: $BINARY"
echo "Output: $OUTPUT_DIR"
echo ""

# Check if ncu is available
if ! command -v ncu &> /dev/null; then
    echo "❌ ERROR: ncu (Nsight Compute) not found"
    echo "Install: https://developer.nvidia.com/nsight-compute"
    exit 1
fi

echo "NCU Version: $(ncu --version | head -1)"
echo ""

#==============================================================================
# 1. SPEED OF LIGHT (SOL) - Quick roofline check
#==============================================================================

echo "[1/4] Speed of Light (SOL) analysis..."
echo "       Target: DRAM ≥85%, SM ≥70%"
echo ""

ncu --section SpeedOfLight \
    --apply-rules no \
    --metrics sm__throughput.avg.pct_of_peak_sustained_elapsed,dram__throughput.avg.pct_of_peak_sustained_elapsed \
    --target-processes all \
    --export "$OUTPUT_DIR/sol" \
    --force-overwrite \
    "$BINARY" 2>&1 | tee "$OUTPUT_DIR/sol.log"

echo ""
echo "✅ SOL metrics saved to: $OUTPUT_DIR/sol.ncu-rep"
echo ""

#==============================================================================
# 2. MEMORY WORKLOAD ANALYSIS
#==============================================================================

echo "[2/4] Memory workload analysis..."
echo "       Checking: bandwidth utilization, L2 hit rate, TMA efficiency"
echo ""

ncu --section MemoryWorkloadAnalysis \
    --section MemoryWorkloadAnalysis_Chart \
    --section MemoryWorkloadAnalysis_Tables \
    --target-processes all \
    --export "$OUTPUT_DIR/memory" \
    --force-overwrite \
    "$BINARY" 2>&1 | tee "$OUTPUT_DIR/memory.log"

echo ""
echo "✅ Memory analysis saved to: $OUTPUT_DIR/memory.ncu-rep"
echo ""

#==============================================================================
# 3. COMPUTE WORKLOAD ANALYSIS (TENSOR CORES)
#==============================================================================

echo "[3/4] Compute workload analysis (Tensor Cores)..."
echo "       Checking: WGMMA utilization, warp stalls, occupancy"
echo ""

ncu --section ComputeWorkloadAnalysis \
    --section Occupancy \
    --section WarpStateStats \
    --target-processes all \
    --export "$OUTPUT_DIR/compute" \
    --force-overwrite \
    "$BINARY" 2>&1 | tee "$OUTPUT_DIR/compute.log"

echo ""
echo "✅ Compute analysis saved to: $OUTPUT_DIR/compute.ncu-rep"
echo ""

#==============================================================================
# 4. FULL PROFILE (detailed metrics)
#==============================================================================

echo "[4/4] Full profile (all sections)..."
echo "       Warning: This may take 5-10 minutes"
echo ""

ncu --set full \
    --target-processes all \
    --export "$OUTPUT_DIR/full_profile" \
    --force-overwrite \
    "$BINARY" 2>&1 | tee "$OUTPUT_DIR/full_profile.log"

echo ""
echo "✅ Full profile saved to: $OUTPUT_DIR/full_profile.ncu-rep"
echo ""

#==============================================================================
# SUMMARY: Extract key metrics
#==============================================================================

echo "========================================"
echo "VALIDATION SUMMARY"
echo "========================================"
echo ""

# Parse SOL metrics from log
echo "Key Metrics (from SOL):"
echo ""
grep -E "(dram__throughput|sm__throughput)" "$OUTPUT_DIR/sol.log" || echo "(parsing failed - check $OUTPUT_DIR/sol.log manually)"

echo ""
echo "========================================"
echo "NEXT STEPS"
echo "========================================"
echo ""
echo "1. Open NSight Compute UI:"
echo "   ncu-ui $OUTPUT_DIR/sol.ncu-rep"
echo ""
echo "2. Check roofline plot:"
echo "   - DRAM bandwidth: Should be ≥85% on memory-bound phases"
echo "   - SM throughput: Should be ≥70% during WGMMA bursts"
echo ""
echo "3. Compare vs FA3 baseline:"
echo "   - FA3 achieves 85-90% DRAM utilization"
echo "   - FA3 achieves 70-80% SM utilization"
echo ""
echo "4. If metrics are low:"
echo "   - Check Memory Analysis for bottlenecks"
echo "   - Check Compute Analysis for warp stalls"
echo "   - Check Occupancy for SM utilization"
echo ""
echo "Reports saved in: $OUTPUT_DIR"
echo ""

