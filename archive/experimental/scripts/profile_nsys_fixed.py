#!/usr/bin/env python3
"""
Fixed Nsight Systems profiling with proper environment setup
"""

import torch
import os
import sys

# Set environment variables for Ninja BEFORE importing our kernel
os.environ['USE_NINJA'] = '1'
os.environ['MAX_JOBS'] = str(os.cpu_count() or 4)

# Add repo root to path
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, REPO_ROOT)

from cudadent42.bench.fa_s512_v3 import flash_attention_s512_v3_forward

print("="*80)
print("Nsight Systems Profile - V3 Kernel (Fixed)")
print("="*80)
print(f"USE_NINJA: {os.environ.get('USE_NINJA')}")
print(f"MAX_JOBS: {os.environ.get('MAX_JOBS')}")

# Test shape
B, H, S, D = 1, 8, 512, 64
causal = False

print(f"\nShape: B={B}, H={H}, S={S}, D={D}, causal={causal}")
print(f"GPU: {torch.cuda.get_device_name(0)}")

# Create inputs
Q = torch.randn(B, H, S, D, device='cuda', dtype=torch.float16)
K = torch.randn(B, H, S, D, device='cuda', dtype=torch.float16)
V = torch.randn(B, H, S, D, device='cuda', dtype=torch.float16)

print("\nRunning warmup (will compile kernel)...")
# Warmup - this will trigger JIT compilation
for i in range(3):
    output = flash_attention_s512_v3_forward(Q, K, V, is_causal=causal, config_id=1)
    print(f"  Warmup {i+1}/3 complete")
torch.cuda.synchronize()

print("\n✓ Kernel compiled and warmed up")
print("Starting profiled runs...")

# Profiled runs with NVTX markers
torch.cuda.nvtx.range_push("V3_KERNEL_PROFILE")
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

