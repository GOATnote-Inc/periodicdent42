#!/usr/bin/env python3
"""
Quick Nsight Systems (nsys) profiling for V3 kernel
Generates timeline and identifies performance bottlenecks
"""

import torch
import os
import sys

# Add repo root to path
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, REPO_ROOT)

from cudadent42.bench.fa_s512_v3 import flash_attention_s512_v3_forward

print("="*80)
print("Nsight Systems Quick Profile - V3 Kernel")
print("="*80)

# Test shape (small for quick profiling)
B, H, S, D = 1, 8, 512, 64
causal = False

print(f"Shape: B={B}, H={H}, S={S}, D={D}, causal={causal}")
print(f"GPU: {torch.cuda.get_device_name(0)}")

# Create inputs
Q = torch.randn(B, H, S, D, device='cuda', dtype=torch.float16)
K = torch.randn(B, H, S, D, device='cuda', dtype=torch.float16)
V = torch.randn(B, H, S, D, device='cuda', dtype=torch.float16)

print("\nRunning warmup...")
# Warmup
for _ in range(3):
    output = flash_attention_s512_v3_forward(Q, K, V, is_causal=causal, config_id=1)
torch.cuda.synchronize()

print("Starting profiled run...")
# Profiled runs
torch.cuda.nvtx.range_push("V3_kernel_profile")
for i in range(10):
    torch.cuda.nvtx.range_push(f"iteration_{i}")
    output = flash_attention_s512_v3_forward(Q, K, V, is_causal=causal, config_id=1)
    torch.cuda.nvtx.range_pop()
torch.cuda.nvtx.range_pop()

torch.cuda.synchronize()
print("✓ Profile complete")
print(f"Output shape: {output.shape}")
print(f"Output range: [{output.min().item():.4f}, {output.max().item():.4f}]")
print("="*80)

