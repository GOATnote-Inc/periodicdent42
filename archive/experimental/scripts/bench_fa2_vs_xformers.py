#!/usr/bin/env python3
"""
Benchmark: FlashAttention-2 vs xFormers SDPA (champion)

Compare FA-2 direct vs pytorch_sdpa_efficient (xFormers backend)
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.absolute()))

import torch
import time

def benchmark(name, fn, q, k, v, warmup=10, iters=100):
    """Benchmark a function."""
    # Warmup
    for _ in range(warmup):
        _ = fn(q, k, v)
    torch.cuda.synchronize()
    
    # Benchmark
    torch.cuda.synchronize()
    start = time.perf_counter()
    
    for _ in range(iters):
        out = fn(q, k, v)
    
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    
    latency_us = (elapsed / iters) * 1e6
    
    return out, latency_us

def main():
    # Test shape
    B, H, S, D = 1, 8, 512, 64
    
    print("=" * 80)
    print("BENCHMARK: FlashAttention-2 vs xFormers SDPA")
    print("=" * 80)
    print()
    print(f"Shape: B={B}, H={H}, S={S}, D={D}")
    print()
    
    # Generate data
    torch.manual_seed(42)
    q = torch.randn(B, H, S, D, device='cuda', dtype=torch.float16)
    k = torch.randn(B, H, S, D, device='cuda', dtype=torch.float16)
    v = torch.randn(B, H, S, D, device='cuda', dtype=torch.float16)
    
    # Import FA-2
    try:
        from flash_attn import flash_attn_func
        fa2_available = True
    except ImportError:
        print("❌ FlashAttention-2 not available")
        fa2_available = False
    
    # Benchmark FA-2
    if fa2_available:
        print("Benchmarking FlashAttention-2 (direct)...")
        def fa2_fn(q, k, v):
            # FA-2 expects (B, S, H, D)
            q_fa = q.transpose(1, 2)
            k_fa = k.transpose(1, 2)
            v_fa = v.transpose(1, 2)
            out_fa = flash_attn_func(q_fa, k_fa, v_fa, causal=False)
            return out_fa.transpose(1, 2)
        
        out_fa2, latency_fa2 = benchmark("FA-2", fa2_fn, q, k, v)
        print(f"  Latency: {latency_fa2:.2f} μs")
        print()
    
    # Benchmark xFormers (via PyTorch SDPA)
    print("Benchmarking PyTorch SDPA (xFormers backend)...")
    def sdpa_fn(q, k, v):
        return torch.nn.functional.scaled_dot_product_attention(
            q, k, v, attn_mask=None, dropout_p=0.0, is_causal=False
        )
    
    out_sdpa, latency_sdpa = benchmark("SDPA", sdpa_fn, q, k, v)
    print(f"  Latency: {latency_sdpa:.2f} μs")
    print()
    
    # Correctness
    if fa2_available:
        max_diff = (out_fa2 - out_sdpa).abs().max().item()
        print(f"Correctness: max_diff = {max_diff:.6f}")
        correct = max_diff <= 2e-3
        print(f"  {'✅' if correct else '❌'} {correct}")
        print()
    
    # Summary
    print("=" * 80)
    print("RESULTS")
    print("=" * 80)
    if fa2_available:
        print(f"FlashAttention-2:  {latency_fa2:.2f} μs")
    print(f"xFormers SDPA:     {latency_sdpa:.2f} μs")
    print()
    
    if fa2_available:
        speedup = latency_sdpa / latency_fa2
        if speedup > 1.0:
            print(f"✅ FA-2 is {speedup:.2f}× faster")
        else:
            print(f"⚠️ xFormers is {1/speedup:.2f}× faster")
    print()
    
    return 0

if __name__ == '__main__':
    sys.exit(main())

