#!/usr/bin/env bash
set -euxo pipefail

# Script to build and benchmark scalar baseline (USE_WMMA=0)
# Run on GPU instance with CUDA available

cd ~/periodicdent42 || cd "$(dirname "$0")/.."

export CUDA_HOME=${CUDA_HOME:-/usr/local/cuda}
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="$HOME/periodicdent42:$HOME/periodicdent42/cudadent42/bench:${PYTHONPATH:-}"

echo "═══════════════════════════════════════════════════════════════════"
echo "  Scalar Baseline Benchmark (USE_WMMA=0)"
echo "═══════════════════════════════════════════════════════════════════"
echo ""

# Step 1: Build with WMMA disabled
echo "[1/5] Building scalar kernel (USE_WMMA=0)..."
export USE_WMMA=0
rm -rf ~/.cache/torch_extensions/* /tmp/torch_extensions/* || true

python3 - <<'PY'
from build_v3_release import build_v3_release
import os
print(f"USE_WMMA={os.environ.get('USE_WMMA', '1')}")
build_v3_release(False)
print("✅ scalar baseline build OK (USE_WMMA=0)")
PY

echo ""
echo "[2/5] Running benchmark (global stream)..."
python3 scripts/bench_s512_tc_vs_sdpa.py 2>&1 | tee cudadent42/artifacts/bench/bench_scalar_global.log || true

echo ""
echo "[3/5] Running benchmark (per-iteration streams)..."
python3 scripts/bench_s512_tc_vs_sdpa.py --streams 2>&1 | tee cudadent42/artifacts/bench/bench_scalar_streams.log || true

echo ""
echo "[4/5] Generating summary..."
python3 scripts/summarize_s512_bench.py || true

echo ""
echo "[5/5] Results..."
cat cudadent42/artifacts/bench/S512_BENCH_SUMMARY.md 2>/dev/null || echo "No summary generated"

echo ""
echo "═══════════════════════════════════════════════════════════════════"
echo "  Scalar Baseline Complete"
echo "═══════════════════════════════════════════════════════════════════"
echo "Artifacts:"
ls -lh cudadent42/artifacts/bench/*.{json,log,md} 2>/dev/null | head -20

