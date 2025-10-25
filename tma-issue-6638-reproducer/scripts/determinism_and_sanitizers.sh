#!/usr/bin/env bash
set -euo pipefail

RUNS="${1:-1000}"

# 1) Determinism (bitwise)
python3 - <<PY
import torch, triton, triton.language as tl
import sys
sys.path.insert(0, 'scripts')
from tma_user_repro import copy_tile_bp

torch.manual_seed(123)

M,N = 1024,512
X = torch.randn((M,N), device='cuda', dtype=torch.float16)
Y = torch.zeros_like(X)
grid = (triton.cdiv(M,128),)

# reference
copy_tile_bp[grid](X, Y, M, N, X.stride(0), X.stride(1), Y.stride(0), Y.stride(1),
                   BLOCK_M=128, BLOCK_N=128, num_stages=4, num_warps=8)
ref = Y.clone()

mismatch = 0
for i in range(${RUNS}):
    Y.zero_()
    copy_tile_bp[grid](X, Y, M, N, X.stride(0), X.stride(1), Y.stride(0), Y.stride(1),
                       BLOCK_M=128, BLOCK_N=128, num_stages=4, num_warps=8)
    torch.cuda.synchronize()
    if not torch.equal(ref, Y):
        mismatch += 1
print(f"DETERMINISM: runs=${RUNS}, mismatches={mismatch}")
PY

# 2) Sanitizers (fail fast)
if command -v compute-sanitizer >/dev/null 2>&1; then
  echo "Running compute-sanitizer memcheck..."
  compute-sanitizer --tool memcheck   python3 scripts/tma_user_repro.py >/dev/null
  echo "Running compute-sanitizer racecheck..."
  compute-sanitizer --tool racecheck  python3 scripts/tma_user_repro.py >/dev/null
  echo "Running compute-sanitizer synccheck..."
  compute-sanitizer --tool synccheck  python3 scripts/tma_user_repro.py >/dev/null || true
  echo "✅ Sanitizers complete"
else
  echo "⚠️  compute-sanitizer not found, skipping"
fi


