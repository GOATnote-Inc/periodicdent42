#!/usr/bin/env bash
set -euo pipefail

# Hardened benchmark driver for SDPA candidates.
# Produces p50/p90 + 95% CI JSON under artifacts/bench/

BACKENDS=${BACKENDS:-"baseline_a baseline_b candidate_triton_ws candidate_triton_flashlike candidate_cuda_stub"}
SHAPES=${SHAPES:-"small mission long"}
ITERS=${ITERS:-100}
WARMUP=${WARMUP:-20}

mkdir -p artifacts/bench

echo "Running kbench..."
for backend in $BACKENDS; do
  for shape in $SHAPES; do
    echo "==> backend=$backend shape=$shape iters=$ITERS warmup=$WARMUP"
    python3 scripts/kbench.py --backend "$backend" --shape "$shape" --iters "$ITERS" --warmup "$WARMUP" --out "artifacts/bench/${backend}_${shape}.json" || true
  done
done

echo "DONE. See artifacts/bench/*.json"
