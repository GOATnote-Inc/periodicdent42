#!/bin/bash
# Test one Evo variant
set -e

export PATH="/usr/local/cuda/bin:$PATH"
source ~/venv/bin/activate

BLOCK_M=${1:-32}
NUM_WARPS=${2:-4}
VEC_WIDTH=${3:-4}

cd ~/periodicdent42

echo "Testing: BLOCK_M=$BLOCK_M NUM_WARPS=$NUM_WARPS VEC_WIDTH=$VEC_WIDTH"

export BLOCK_M NUM_WARPS VEC_WIDTH
export SYNC_POLICY=2
export REDUCE=warp

python3 bench/build_phase3_variant.py >/dev/null 2>&1

python3 -c "
import torch
import time
import fa_phase3

B, H, S, D = 1, 8, 512, 64
q = torch.randn(B, H, S, D, dtype=torch.float16, device='cuda:0')
k = torch.randn(B, H, S, D, dtype=torch.float16, device='cuda:0')
v = torch.randn(B, H, S, D, dtype=torch.float16, device='cuda:0')
scale = 1.0 / (D ** 0.5)

for _ in range(5):
    o = fa_phase3.forward(q, k, v, scale)

torch.cuda.synchronize()
start = time.perf_counter()
for _ in range(50):
    o = fa_phase3.forward(q, k, v, scale)
torch.cuda.synchronize()
end = time.perf_counter()

time_us = (end - start) * 1e6 / 50
print(f'{time_us:.2f}')
"

