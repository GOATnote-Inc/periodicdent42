#!/usr/bin/env python3
"""
Focused Nsight Compute profiling script for V3 kernel
Runs a single shape to be profiled by ncu
"""

import os
import sys
import torch

# Set Ninja env vars
os.environ['USE_NINJA'] = '1'
os.environ['MAX_JOBS'] = '4'

# Add repo root to path
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, REPO_ROOT)

from cudadent42.bench.fa_s512_v3 import flash_attention_s512_v3_forward

# Get shape from args or use default
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--B', type=int, default=1)
parser.add_argument('--H', type=int, default=8)
parser.add_argument('--S', type=int, default=512)
parser.add_argument('--D', type=int, default=64)
parser.add_argument('--causal', action='store_true')
args = parser.parse_args()

B, H, S, D = args.B, args.H, args.S, args.D
causal = args.causal

print(f"Profiling V3 kernel: B={B}, H={H}, S={S}, D={D}, causal={causal}")

# Create inputs
Q = torch.randn(B, H, S, D, device='cuda', dtype=torch.float16)
K = torch.randn(B, H, S, D, device='cuda', dtype=torch.float16)
V = torch.randn(B, H, S, D, device='cuda', dtype=torch.float16)

# Warmup (ensure kernel is compiled)
print("Warmup...")
for _ in range(5):
    output = flash_attention_s512_v3_forward(Q, K, V, is_causal=causal, config_id=1)
torch.cuda.synchronize()

print("Profiled run...")
# Single profiled run (ncu will capture this)
output = flash_attention_s512_v3_forward(Q, K, V, is_causal=causal, config_id=1)
torch.cuda.synchronize()

print(f"✓ Output shape: {output.shape}")
print(f"✓ Output range: [{output.min().item():.4f}, {output.max().item():.4f}]")

