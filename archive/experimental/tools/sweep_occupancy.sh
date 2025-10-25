#!/usr/bin/env bash
#
# Occupancy Sweep: Register Pressure Attack
#
# Goal: Find optimal REGCAP + THREADS config that maximizes:
#   - Eligible warps per scheduler (≥ 2)
#   - Issue slot utilization (≥ 60%)
#   - Achieved occupancy (≥ 20%)
#   - Latency (< 33.19 μs)
#

set -euo pipefail

echo "════════════════════════════════════════════════════════════════════════════════"
echo "OCCUPANCY SWEEP: Register Pressure Attack"
echo "════════════════════════════════════════════════════════════════════════════════"
echo ""
echo "Target: Beat xFormers (33.19 μs) with better occupancy"
echo "Strategy: Sweep REGCAP + THREADS to lift eligible warps"
echo ""

# Create output directory
mkdir -p out/sweep
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
SWEEP_LOG="out/sweep/occupancy_sweep_${TIMESTAMP}.log"

echo "Sweep log: $SWEEP_LOG"
echo ""

# Dimensions
REGCAPS=(64 72 80 88 96)
THREADS=(128 192 256)
TILE_MS=(32 64)

TOTAL=$((${#REGCAPS[@]} * ${#THREADS[@]} * ${#TILE_MS[@]}))
CURRENT=0

echo "Total configurations: $TOTAL"
echo ""

# Initialize results file
cat > "$SWEEP_LOG" <<EOF
# Occupancy Sweep Results
# Date: $(date)
# Target: < 33.19 μs (xFormers baseline)
#
# REGCAP,THREADS,TILE_M,Latency_us,Correct,Max_Diff,Occupancy,Eligible_Warps,Issue_Util
EOF

for REGCAP in "${REGCAPS[@]}"; do
  for TH in "${THREADS[@]}"; do
    for TILE_M in "${TILE_MS[@]}"; do
      CURRENT=$((CURRENT + 1))
      
      echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
      echo "CONFIG $CURRENT/$TOTAL: REGCAP=$REGCAP, THREADS=$TH, TILE_M=$TILE_M"
      echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
      echo ""
      
      # Build with config
      echo "Building..."
      export REGCAP=$REGCAP
      export LB_THREADS=$TH
      export LB_MIN=2
      export TILE_M=$TILE_M
      
      if python bench/build_custom_tuned.py 2>&1 | tee -a "$SWEEP_LOG.build"; then
        echo "  ✅ Build successful"
      else
        echo "  ❌ Build failed, skipping"
        echo "$REGCAP,$TH,$TILE_M,FAIL,FAIL,FAIL,FAIL,FAIL,FAIL" >> "$SWEEP_LOG"
        continue
      fi
      
      # Benchmark
      echo ""
      echo "Benchmarking..."
      if python bench/run_custom_tuned.py --shape S=512,D=64 2>&1 | tee -a "$SWEEP_LOG.bench"; then
        # Extract metrics (will parse from output)
        LATENCY=$(grep "Latency:" "$SWEEP_LOG.bench" | tail -1 | awk '{print $2}')
        CORRECT=$(grep "Correct:" "$SWEEP_LOG.bench" | tail -1 | awk '{print $2}')
        MAX_DIFF=$(grep "Max diff:" "$SWEEP_LOG.bench" | tail -1 | awk '{print $3}')
        
        echo "  ✅ Latency: ${LATENCY:-N/A} μs"
        echo "  ✅ Correct: ${CORRECT:-N/A}"
        echo "  ✅ Max diff: ${MAX_DIFF:-N/A}"
        
        # Placeholder for NCU metrics (would run ncu separately)
        OCC="N/A"
        ELIG="N/A"
        ISSUE="N/A"
        
        echo "$REGCAP,$TH,$TILE_M,${LATENCY:-N/A},${CORRECT:-N/A},${MAX_DIFF:-N/A},$OCC,$ELIG,$ISSUE" >> "$SWEEP_LOG"
      else
        echo "  ❌ Benchmark failed"
        echo "$REGCAP,$TH,$TILE_M,FAIL,FAIL,FAIL,FAIL,FAIL,FAIL" >> "$SWEEP_LOG"
      fi
      
      echo ""
    done
  done
done

echo "════════════════════════════════════════════════════════════════════════════════"
echo "SWEEP COMPLETE"
echo "════════════════════════════════════════════════════════════════════════════════"
echo ""
echo "Results: $SWEEP_LOG"
echo ""
echo "Finding best configuration..."

# Find best (lowest latency, correct)
if grep -v "^#" "$SWEEP_LOG" | grep -v "FAIL" | sort -t',' -k4 -n | head -5 > "$SWEEP_LOG.best"; then
  echo "Top 5 configurations:"
  cat "$SWEEP_LOG.best"
else
  echo "No successful configurations found!"
fi

echo ""
echo "Next steps:"
echo "  1. Review $SWEEP_LOG.best"
echo "  2. Run NCU on best config"
echo "  3. Verify occupancy metrics"
echo ""

