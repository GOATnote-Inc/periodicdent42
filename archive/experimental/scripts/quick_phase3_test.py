#!/usr/bin/env python3
import torch
import sys
from pathlib import Path

repo_root = Path(__file__).parent.parent
sys.path.insert(0, str(repo_root / "bench"))

from build_phase3_variant import build_phase3_variant

print("Building Phase 3...")
build_phase3_variant()

import fa_phase3

# Test config
B, H, S, D = 1, 8, 512, 64
q = torch.randn(B, H, S, D, dtype=torch.float16, device='cuda:0')
k = torch.randn(B, H, S, D, dtype=torch.float16, device='cuda:0')
v = torch.randn(B, H, S, D, dtype=torch.float16, device='cuda:0')
scale = 1.0 / (D ** 0.5)

# Warmup
for _ in range(10):
    o = fa_phase3.forward(q, k, v, scale)

# Benchmark
torch.cuda.synchronize()
import time
start = time.perf_counter()
for _ in range(100):
    o = fa_phase3.forward(q, k, v, scale)
torch.cuda.synchronize()
end = time.perf_counter()

time_us = (end - start) * 1e6 / 100
print(f"Phase3: {time_us:.2f} μs")

