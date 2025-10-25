#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")"/../.. && pwd)"
OUT="$ROOT/cudadent42/artifacts/stats/ptxas.txt"
mkdir -p "$(dirname "$OUT")"
nvcc -arch=sm_89 -std=c++17 -O3 --ptxas-options=-v \
  -I "$ROOT/third_party/cutlass/include" \
  -I "$ROOT/third_party/cutlass/tools/util/include" \
  -c "$ROOT/cudadent42/bench/kernels/fa_s512_v3.cu" -o /dev/null \
  |& tee "$OUT"
echo "✅ ptxas snapshot -> $OUT"

