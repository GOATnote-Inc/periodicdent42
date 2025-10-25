#!/usr/bin/env python3
"""
NCU Profile: SDPA Kernel Only

Strategy:
1. Pre-generate Q, K, V tensors (OUTSIDE profiling scope)
2. Profile ONLY the SDPA call
3. This isolates attention kernels from tensor generation
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.absolute()))

import torch
from baselines import registry

# Configuration
B, H, S, D = 1, 8, 512, 64

print("=" * 80)
print("NCU PROFILE: SDPA Kernel Only (pytorch_sdpa_efficient)")
print("=" * 80)
print()
print(f"Shape: B={B}, H={H}, S={S}, D={D}")
print()

# Get champion
champion = registry.get("pytorch_sdpa_efficient")

print("Step 1: Pre-generate tensors (outside profiling scope)...")
torch.manual_seed(42)
q = torch.randn(B, H, S, D, device='cuda', dtype=torch.float16)
k = torch.randn(B, H, S, D, device='cuda', dtype=torch.float16)
v = torch.randn(B, H, S, D, device='cuda', dtype=torch.float16)
torch.cuda.synchronize()
print("  ✅ Tensors ready")
print()

print("Step 2: Warmup (5 iterations)...")
for _ in range(5):
    _ = champion.fn(q, k, v, causal=False, dropout_p=0.0)
torch.cuda.synchronize()
print("  ✅ Warmup complete")
print()

print("Step 3: Profile SDPA call (this is what NCU captures)...")
print("=" * 80)
torch.cuda.synchronize()

# THIS IS THE ONLY KERNEL CALL THAT WILL BE PROFILED
out = champion.fn(q, k, v, causal=False, dropout_p=0.0)

torch.cuda.synchronize()
print("=" * 80)
print("  ✅ SDPA call complete")
print()

print("Output shape:", out.shape)
print()
print("NCU should have captured only the SDPA attention kernels!")

