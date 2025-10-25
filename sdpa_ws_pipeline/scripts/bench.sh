#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
mkdir -p "$ROOT/artifacts/bench"

# Mission-shaped dims (adjust if needed)
SHAPE="${SHAPE:-2,8,512,64}"
WARMUP="${WARMUP:-10}"
ITERS="${ITERS:-100}"

python3 "$ROOT/scripts/kbench.py" --shape "$SHAPE" --warmup "$WARMUP" --iters "$ITERS" \
  --variants candidate_triton_flashlike candidate_cuda_stub candidate_triton_ws
