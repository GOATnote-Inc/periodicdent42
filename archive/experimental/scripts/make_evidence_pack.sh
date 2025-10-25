#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")"/.. && pwd)"
OUT="$ROOT/evidence_pack_$(date +%Y%m%d_%H%M%S).zip"
cd "$ROOT"
echo "[1/4] Verifying critical files..."
test -f EVIDENCE_NAV.md
test -f HIRING_DECISION_RESPONSE.md || true
test -f cudadent42/artifacts/stats/ptxas.txt
test -f cudadent42/artifacts/stats/wmma_proof.txt

echo "[2/4] Collecting artifacts..."
mkdir -p /tmp/_ev
cp -a EVIDENCE_NAV.md /tmp/_ev/
cp -a HIRING_DECISION_RESPONSE.md /tmp/_ev/ 2>/dev/null || true
cp -a cudadent42/artifacts /tmp/_ev/
cp -a cudadent42/bench/kernels/fa_s512_v3.cu /tmp/_ev/
cp -a cudadent42/bench/build_v3_release.py /tmp/_ev/
cp -a scripts/ci/*.sh /tmp/_ev/ 2>/dev/null || true
cp -a scripts/bench_s512_tc_vs_sdpa.py /tmp/_ev/ 2>/dev/null || true

echo "[3/4] Zipping..."
cd /tmp/_ev
zip -r "$OUT" . >/dev/null
cd -
echo "[4/4] Evidence pack -> $OUT"
echo "$OUT"

