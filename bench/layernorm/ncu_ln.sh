#!/usr/bin/env bash
set -euo pipefail
export PATH="/usr/local/cuda/bin:$PATH"
: "${THREADS:=256}" "${ROWS_PER_CTA:=1}" "${VEC_WIDTH:=4}" "${USE_WARP:=1}"
ncu --target-processes all --replay-mode kernel \
  --metrics sm__warps_active.avg.pct_of_peak_sustained_active,\
sm__pipe_tensor_active.avg.pct_of_peak_sustained_active,\
dram__throughput.avg.pct_of_peak_sustained_elapsed \
  --csv -o evidence/ncu_layernorm \
  env THREADS=$THREADS ROWS_PER_CTA=$ROWS_PER_CTA VEC_WIDTH=$VEC_WIDTH USE_WARP=$USE_WARP \
  python bench/layernorm/bench_ln.py

