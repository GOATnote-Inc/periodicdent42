#!/usr/bin/env bash
set -euo pipefail
nvcc -O3 -use_fast_math -std=c++17 -Xptxas -v \
  -gencode=arch=compute_89,code=sm_89 \
  ${CUSTOM_TILE_FLAGS:-} \
  bench/micro/bench_many.cu -o bench/micro/bench_many
echo "Built bench/micro/bench_many"

