#!/bin/bash
# bench/micro/build_micro.sh
set -e

export PATH="/usr/local/cuda/bin:$PATH"

nvcc -O3 -use_fast_math -std=c++17 \
  -gencode=arch=compute_89,code=sm_89 \
  bench/micro/bench_many.cu -o bench/micro/bench_many

echo "✅ Microbench compiled"
