#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"

echo "==== Stage 0: Capture Environment ===="
python3 "$ROOT/scripts/capture_env.py"

echo ""
echo "==== Stage 1: Benchmark Baselines & Candidates ===="
bash "$ROOT/scripts/bench.sh"

echo ""
echo "==== Stage 2: EvoEngineer-Full Autotune ===="
python3 "$ROOT/scripts/evo_tune.py" --shape "${SHAPE:-2,8,512,64}" --budget "${BUDGET:-128}" --elite_k 6

echo ""
echo "==== Stage 3: NCU Profiling ===="
bash "$ROOT/scripts/profile.sh"

echo ""
echo "==== Stage 4: Generate Report ===="
python3 "$ROOT/scripts/summarize.py"

echo ""
echo "✅ Repro complete. See reports/summary.md"
