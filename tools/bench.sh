#!/bin/bash
# FlashCore v12: Build + Benchmark Script
# Usage: ./tools/bench.sh [variant_id]

set -e

VARIANT="${1:-baseline}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT/flashcore"

echo "════════════════════════════════════════════════════════════════"
echo "FlashCore v12 Benchmark: $VARIANT"
echo "════════════════════════════════════════════════════════════════"

# Build
echo "[1/3] Building..."
python3 build_wmma.py 2>&1 | tee ../logs/build_${VARIANT}.log
BUILD_EXIT=${PIPESTATUS[0]}

if [ $BUILD_EXIT -ne 0 ]; then
    echo "❌ BUILD FAILED"
    exit 1
fi

# Extract PTXAS metrics
echo "[2/3] Checking PTXAS..."
REGS=$(grep "registers" ../logs/build_${VARIANT}.log | tail -1 | grep -oP '\d+(?= registers)' || echo "0")
SMEM=$(grep "bytes smem" ../logs/build_${VARIANT}.log | tail -1 | grep -oP '\d+(?= bytes smem)' || echo "0")
SPILL=$(grep "bytes spill" ../logs/build_${VARIANT}.log | tail -1 | grep -oP '\d+(?= bytes spill)' || echo "0")
STACK=$(grep "bytes stack frame" ../logs/build_${VARIANT}.log | tail -1 | grep -oP '\d+(?= bytes stack frame)' || echo "0")

echo "  Registers: $REGS/thread"
echo "  SMEM: $SMEM bytes/block"
echo "  Spill: $SPILL bytes"
echo "  Stack: $STACK bytes"

# Gate checks
if [ "$REGS" -gt 64 ]; then
    echo "❌ GATE FAIL: Registers $REGS > 64"
    exit 1
fi

if [ "$SPILL" -gt 0 ]; then
    echo "❌ GATE FAIL: Spills $SPILL > 0"
    exit 1
fi

if [ "$STACK" -gt 0 ]; then
    echo "❌ GATE FAIL: Stack $STACK > 0"
    exit 1
fi

echo "✅ PTXAS gates passed"

# Benchmark
echo "[3/3] Benchmarking..."
python3 test_v12_expert.py 2>&1 | tee ../logs/bench_${VARIANT}.log
BENCH_EXIT=${PIPESTATUS[0]}

if [ $BENCH_EXIT -ne 0 ]; then
    echo "❌ BENCHMARK FAILED"
    exit 1
fi

# Extract metrics
LATENCY=$(grep "Final Latency:" ../logs/bench_${VARIANT}.log | grep -oP '\d+\.\d+(?= µs)' || echo "0")
MAX_ERR=$(grep "Max error:" ../logs/bench_${VARIANT}.log | grep -oP '\d+\.\d+' | head -1 || echo "1.0")

echo ""
echo "════════════════════════════════════════════════════════════════"
echo "Results: $VARIANT"
echo "════════════════════════════════════════════════════════════════"
echo "Latency: $LATENCY µs"
echo "Max Error: $MAX_ERR"
echo "Registers: $REGS/thread"
echo "SMEM: $SMEM bytes"
echo "════════════════════════════════════════════════════════════════"

# Write JSON
cat > ../results/bench_${VARIANT}.json <<EOF
{
  "variant": "$VARIANT",
  "latency_us": $LATENCY,
  "max_error": $MAX_ERR,
  "ptxas": {
    "registers": $REGS,
    "smem_bytes": $SMEM,
    "spill_bytes": $SPILL,
    "stack_bytes": $STACK
  },
  "timestamp": "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
}
EOF

echo "✅ Results saved to results/bench_${VARIANT}.json"
exit 0

