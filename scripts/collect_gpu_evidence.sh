#!/usr/bin/env bash
set -euo pipefail
export PATH="/usr/local/cuda/bin:$PATH"
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")"/.. && pwd)"
ART="$ROOT/cudadent42/artifacts"
mkdir -p "$ART" "$ART/sanitizers" "$ART/stats" "$ART/bench"
cd "$ROOT"

echo "[1/6] Build (debug, WMMA+asserts)"
rm -rf ~/.cache/torch_extensions/* /tmp/torch_extensions/* || true
python3 - <<'PY' || true
from build_v3_release import build_v3_release
m = build_v3_release(debug=True)
print("✅ build(debug) OK")
PY

echo "[2/6] Sanitizers"
if command -v compute-sanitizer >/dev/null 2>&1; then
  scripts/ci/compute_sanitizer_gate.sh || true
else
  echo "compute-sanitizer not available" > "$ART/sanitizers/SANITIZER_STATUS.txt"
fi

echo "[3/6] PTXAS snapshot"
if command -v nvcc >/dev/null 2>&1; then
  scripts/ci/ptxas_snapshot.sh || true
else
  echo "nvcc not in PATH" > "$ART/stats/ptxas.txt"
fi

echo "[4/6] SASS proof (mma.sync/HMMA)"
SO=$(ls -1 ~/.cache/torch_extensions/*/*.so 2>/dev/null | head -1 || true)
if [ -n "${SO:-}" ] && command -v cuobjdump >/dev/null 2>&1; then
  cuobjdump --dump-sass "$SO" | grep -mi1 "mma.sync\|HMMA" > "$ART/stats/wmma_proof.txt" || echo "No WMMA instructions found" > "$ART/stats/wmma_proof.txt"
else
  echo "cuobjdump or .so missing" > "$ART/stats/wmma_proof.txt"
fi

echo "[5/6] Parity + bench"
pytest -q tests/test_sdpa_parity.py 2>&1 | tee "$ART/sanitizers/parity.log" || true
pytest -q tests/test_tc_sdpa_parity.py 2>&1 | tee "$ART/sanitizers/tc_parity.log" || true
python3 scripts/bench_s512_tc_vs_sdpa.py 2>&1 | tee "$ART/bench/bench.log" || true

echo "[6/6] Done → artifacts under $ART"
ls -lh "$ART"/**/*.{txt,log,json} 2>/dev/null || true
