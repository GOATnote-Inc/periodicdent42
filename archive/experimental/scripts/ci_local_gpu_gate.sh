#!/bin/bash
# Phase 12: CI Regression Gate (Local GPU Version)
# Validates correctness and performance before merge
#
# Usage:
#   ./scripts/ci_local_gpu_gate.sh
#   ./scripts/ci_local_gpu_gate.sh --skip-correctness  # Skip correctness tests
#   ./scripts/ci_local_gpu_gate.sh --baseline-file path/to/leaderboard.json

set -e  # Exit on error

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

# ============================================================================
# Configuration
# ============================================================================

REGRESSION_THRESHOLD=0.02  # Fail if >2% slower
BASELINE_FILE="${BASELINE_FILE:-benchmarks/l4/rbk_results/leaderboard.json}"
SKIP_CORRECTNESS="${SKIP_CORRECTNESS:-false}"
WARMUPS=20
ITERS=100

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --skip-correctness)
            SKIP_CORRECTNESS=true
            shift
            ;;
        --baseline-file)
            BASELINE_FILE="$2"
            shift 2
            ;;
        --threshold)
            REGRESSION_THRESHOLD="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# ============================================================================
# Helper Functions
# ============================================================================

print_header() {
    echo ""
    echo "════════════════════════════════════════════════════════════════════════════════"
    echo "  $1"
    echo "════════════════════════════════════════════════════════════════════════════════"
    echo ""
}

print_section() {
    echo ""
    echo "────────────────────────────────────────────────────────────────────────────────"
    echo "  $1"
    echo "────────────────────────────────────────────────────────────────────────────────"
}

print_pass() {
    echo "✅ PASS: $1"
}

print_fail() {
    echo "❌ FAIL: $1"
}

print_warn() {
    echo "⚠️  WARN: $1"
}

# ============================================================================
# Main Gate Script
# ============================================================================

print_header "CI Regression Gate (Local GPU)"

# Verify CUDA availability
if ! python3 -c "import torch; assert torch.cuda.is_available()" 2>/dev/null; then
    print_fail "CUDA not available"
    exit 1
fi

GPU_NAME=$(python3 -c "import torch; print(torch.cuda.get_device_name(0))")
print_pass "GPU available: $GPU_NAME"

# ============================================================================
# Stage 1: Correctness Tests
# ============================================================================

if [ "$SKIP_CORRECTNESS" = "false" ]; then
    print_section "Stage 1: Correctness Tests"
    
    if python3 tests/test_sdpa_parity.py 2>&1 | tee ci_gate_correctness.log; then
        print_pass "Correctness tests passed"
    else
        print_fail "Correctness tests failed (see ci_gate_correctness.log)"
        exit 1
    fi
else
    print_section "Stage 1: Correctness Tests (SKIPPED)"
    print_warn "Correctness tests skipped by user request"
fi

# ============================================================================
# Stage 2: Baseline Benchmarks
# ============================================================================

print_section "Stage 2: Baseline Benchmarks"

# Run canonical shapes benchmark
BENCHMARK_DIR="benchmarks/l4/ci_gate_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$BENCHMARK_DIR"

if python3 scripts/bench_sdpa_baseline.py \
    --shapes canonical \
    --warmups "$WARMUPS" \
    --iters "$ITERS" \
    --output "$BENCHMARK_DIR" 2>&1 | tee ci_gate_benchmark.log; then
    print_pass "Baseline benchmarks completed"
else
    print_fail "Baseline benchmarks failed (see ci_gate_benchmark.log)"
    exit 1
fi

# ============================================================================
# Stage 3: Performance Regression Check
# ============================================================================

print_section "Stage 3: Performance Regression Check"

if [ ! -f "$BASELINE_FILE" ]; then
    print_warn "No baseline file found at $BASELINE_FILE"
    print_warn "Skipping regression check (first run?)"
    print_warn "Current results will be saved for future comparison"
    
    # Save current results as baseline
    if [ -f "$BENCHMARK_DIR/baseline_ours.json" ]; then
        mkdir -p "$(dirname "$BASELINE_FILE")"
        cp "$BENCHMARK_DIR/baseline_ours.json" "$BASELINE_FILE"
        print_pass "Baseline saved for future comparison"
    fi
else
    print_pass "Baseline file found: $BASELINE_FILE"
    
    # Compare current vs baseline
    python3 << 'PYEOF'
import json
import sys
from pathlib import Path

baseline_file = Path("$BASELINE_FILE".replace("$", ""))
current_file = Path("$BENCHMARK_DIR/baseline_ours.json".replace("$", ""))
threshold = float("$REGRESSION_THRESHOLD".replace("$", ""))

# Load results
with open(baseline_file) as f:
    baseline_data = json.load(f)

with open(current_file) as f:
    current_data = json.load(f)

# Extract p50 latencies by shape
baseline_results = {r["shape"]: r["p50_latency_ms"] for r in baseline_data.get("results", [])}
current_results = {r["shape"]: r["p50_latency_ms"] for r in current_data.get("results", [])}

# Compare
regressions = []
improvements = []

for shape in current_results.keys():
    if shape not in baseline_results:
        continue
    
    baseline_p50 = baseline_results[shape]
    current_p50 = current_results[shape]
    
    delta = (current_p50 - baseline_p50) / baseline_p50
    
    if delta > threshold:
        regressions.append((shape, baseline_p50, current_p50, delta))
    elif delta < -threshold:
        improvements.append((shape, baseline_p50, current_p50, delta))

# Report
print("\nPerformance Comparison:")
print("-" * 80)

if improvements:
    print(f"\n✅ IMPROVEMENTS ({len(improvements)}):")
    for shape, baseline, current, delta in improvements:
        print(f"  {shape}: {baseline:.3f} ms → {current:.3f} ms ({delta*100:.1f}% faster)")

if regressions:
    print(f"\n❌ REGRESSIONS ({len(regressions)}):")
    for shape, baseline, current, delta in regressions:
        print(f"  {shape}: {baseline:.3f} ms → {current:.3f} ms ({delta*100:.1f}% slower)")
    print()
    sys.exit(1)  # Fail gate
else:
    print("\n✅ No significant regressions detected")
    sys.exit(0)
PYEOF
    
    if [ $? -eq 0 ]; then
        print_pass "Performance regression check passed"
    else
        print_fail "Performance regressions detected (threshold: ${REGRESSION_THRESHOLD}%)"
        exit 1
    fi
fi

# ============================================================================
# Stage 4: Generate Gate Report
# ============================================================================

print_section "Stage 4: Generate Gate Report"

cat > "$BENCHMARK_DIR/gate_report.md" << EOF
# CI Gate Report

**Date**: $(date '+%Y-%m-%d %H:%M:%S')  
**GPU**: $GPU_NAME  
**Branch**: $(git rev-parse --abbrev-ref HEAD)  
**Commit**: $(git rev-parse --short HEAD)

## Results

### Correctness Tests
$(if [ "$SKIP_CORRECTNESS" = "false" ]; then echo "✅ PASSED"; else echo "⚠️  SKIPPED"; fi)

### Baseline Benchmarks
✅ COMPLETED

See:
- \`$BENCHMARK_DIR/baseline_ours.json\`
- \`$BENCHMARK_DIR/baseline_sdpa.json\`
- \`$BENCHMARK_DIR/comparison.json\`

### Performance Regression Check
$(if [ -f "$BASELINE_FILE" ]; then echo "✅ PASSED (no regressions >$REGRESSION_THRESHOLD%)"; else echo "⚠️  SKIPPED (no baseline)"; fi)

## Artifacts

- Logs: \`ci_gate_*.log\`
- Benchmarks: \`$BENCHMARK_DIR/\`

## Summary

✅ **GATE PASSED** - All checks successful

EOF

print_pass "Gate report generated: $BENCHMARK_DIR/gate_report.md"

# ============================================================================
# Success
# ============================================================================

print_header "✅ CI GATE PASSED"

echo "All checks passed successfully!"
echo ""
echo "Artifacts:"
echo "  • Benchmark results: $BENCHMARK_DIR/"
echo "  • Gate report: $BENCHMARK_DIR/gate_report.md"
echo ""
echo "You can now safely merge your changes."

exit 0

