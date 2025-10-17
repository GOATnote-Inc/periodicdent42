#!/bin/bash
# Test M=64 with timeout to diagnose hang

export PATH="/usr/local/cuda/bin:$PATH"
source ~/venv/bin/activate

cd ~/periodicdent42

export BLOCK_M=64
export NUM_WARPS=4
export VEC_WIDTH=4
export SYNC_POLICY=2
export REDUCE=warp

echo "Building M=64..."
python3 bench/build_phase3_variant.py >/dev/null 2>&1

echo "Testing with 5s timeout..."
timeout 5 python3 -c "
import sys
sys.path.insert(0, '/home/kiteboard/.cache/torch_extensions/py310_cu121/fa_phase3')
import torch
import time
import fa_phase3

B, H, S, D = 1, 8, 512, 64
q = torch.randn(B, H, S, D, dtype=torch.float16, device='cuda:0')
k = torch.randn(B, H, S, D, dtype=torch.float16, device='cuda:0')
v = torch.randn(B, H, S, D, dtype=torch.float16, device='cuda:0')
scale = 1.0 / (D ** 0.5)

print('Launching kernel...')
torch.cuda.synchronize()
start = time.time()
o = fa_phase3.forward(q, k, v, scale)
torch.cuda.synchronize()
elapsed = time.time() - start
print(f'SUCCESS: {elapsed*1e6:.2f} μs')
" 2>&1

EXIT_CODE=$?
if [ $EXIT_CODE -eq 124 ]; then
    echo "❌ TIMEOUT (5s) - kernel hung"
elif [ $EXIT_CODE -eq 0 ]; then
    echo "✅ Completed successfully"
else
    echo "❌ Error (exit code $EXIT_CODE)"
fi

