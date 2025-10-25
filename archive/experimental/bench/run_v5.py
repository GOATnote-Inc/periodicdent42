#!/usr/bin/env python3
"""Minimal V5 kernel runner for correctness + latency."""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import time
from bench.build_v5 import build_v5

def run_v5(M=64, N=64, K=32, STAGES=2, NUM_WARPS=8, iters=100):
    """Run V5 kernel and measure performance."""
    
    # Build
    mod = build_v5(M=M, N=N, K=K, STAGES=STAGES, NUM_WARPS=NUM_WARPS)
    
    # Setup
    B, H, S, D = 1, 8, 512, 64
    scale = 1.0 / (D ** 0.5)
    
    torch.manual_seed(42)
    q = torch.randn(B, H, S, D, dtype=torch.float16, device="cuda")
    k = torch.randn(B, H, S, D, dtype=torch.float16, device="cuda")
    v = torch.randn(B, H, S, D, dtype=torch.float16, device="cuda")
    
    # Warmup
    for _ in range(10):
        o = mod.forward(q, k, v, scale)
    
    # Benchmark
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        o = mod.forward(q, k, v, scale)
    torch.cuda.synchronize()
    t1 = time.perf_counter()
    
    avg_us = (t1 - t0) * 1e6 / iters
    
    # Correctness
    o_ref = torch.nn.functional.scaled_dot_product_attention(
        q, k, v, attn_mask=None, dropout_p=0.0, is_causal=False, scale=scale
    )
    
    max_diff = (o.float() - o_ref.float()).abs().max().item()
    passed = torch.allclose(o, o_ref, atol=1e-3, rtol=1e-3)
    
    return {
        "time_us": avg_us,
        "max_diff": max_diff,
        "correct": passed
    }

if __name__ == "__main__":
    result = run_v5()
    print(f"{result['time_us']:.2f}")
    if not result['correct']:
        print(f"❌ FAIL: max_diff={result['max_diff']:.6f}", file=sys.stderr)
        sys.exit(1)

