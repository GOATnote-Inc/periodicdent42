#!/usr/bin/env python3
"""
Manual V3 kernel performance analysis using PyTorch profiler
Identifies kernel launch overhead, actual CUDA time, and memory patterns
"""

import os
import sys
import torch
from torch.profiler import profile, ProfilerActivity, record_function

# Set Ninja env vars
os.environ['USE_NINJA'] = '1'
os.environ['MAX_JOBS'] = '4'

# Add repo root to path
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, REPO_ROOT)

from cudadent42.bench.fa_s512_v3 import flash_attention_s512_v3_forward
import torch.nn.functional as F

print("="*80)
print("V3 Kernel Performance Analysis (PyTorch Profiler)")
print("="*80)

# Test shapes
shapes = [
    (1, 8, 512, 64, False, "v3_small"),
    (4, 16, 512, 64, True, "v3_medium_causal"),
    (8, 16, 512, 64, False, "v3_large"),
]

for B, H, S, D, causal, name in shapes:
    print(f"\n{'='*80}")
    print(f"Shape: {name} (B={B}, H={H}, S={S}, D={D}, causal={causal})")
    print(f"{'='*80}")
    
    Q = torch.randn(B, H, S, D, device='cuda', dtype=torch.float16)
    K = torch.randn(B, H, S, D, device='cuda', dtype=torch.float16)
    V = torch.randn(B, H, S, D, device='cuda', dtype=torch.float16)
    
    # Warmup
    for _ in range(5):
        output = flash_attention_s512_v3_forward(Q, K, V, is_causal=causal, config_id=1)
    torch.cuda.synchronize()
    
    # Profile V3 kernel
    with profile(activities=[ProfilerActivity.CUDA], record_shapes=True) as prof:
        with record_function("V3_kernel"):
            for _ in range(10):
                output = flash_attention_s512_v3_forward(Q, K, V, is_causal=causal, config_id=1)
    
    print("\n--- V3 Kernel Profile ---")
    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))
    
    # Profile SDPA for comparison
    with profile(activities=[ProfilerActivity.CUDA], record_shapes=True) as prof_sdpa:
        with record_function("SDPA"):
            for _ in range(10):
                sdpa_out = F.scaled_dot_product_attention(Q, K, V, is_causal=causal)
    
    print("\n--- SDPA Profile (Reference) ---")
    print(prof_sdpa.key_averages().table(sort_by="cuda_time_total", row_limit=10))
    
    # Timing comparison
    import time
    torch.cuda.synchronize()
    
    # V3 timing
    start = time.perf_counter()
    for _ in range(50):
        output = flash_attention_s512_v3_forward(Q, K, V, is_causal=causal, config_id=1)
    torch.cuda.synchronize()
    v3_time = (time.perf_counter() - start) / 50 * 1000  # ms
    
    # SDPA timing
    start = time.perf_counter()
    for _ in range(50):
        sdpa_out = F.scaled_dot_product_attention(Q, K, V, is_causal=causal)
    torch.cuda.synchronize()
    sdpa_time = (time.perf_counter() - start) / 50 * 1000  # ms
    
    slowdown = v3_time / sdpa_time
    print(f"\n--- Timing Comparison ---")
    print(f"V3:   {v3_time:.3f} ms")
    print(f"SDPA: {sdpa_time:.3f} ms")
    print(f"Slowdown: {slowdown:.1f}×")

print("\n" + "="*80)
print("✅ Analysis Complete")
print("="*80)

