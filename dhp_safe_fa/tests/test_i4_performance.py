#!/usr/bin/env python3
"""
Test I4 Performance (TDD)
=========================

Benchmark I4 against PyTorch SDPA baseline.
"""

import torch
import torch.nn.functional as F
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

def bench(fn, *args, warmup=10, runs=100):
    """Burn methodology benchmarking"""
    for _ in range(warmup):
        fn(*args)
    torch.cuda.synchronize()
    
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    
    start.record()
    for _ in range(runs):
        fn(*args)
    end.record()
    
    torch.cuda.synchronize()
    return start.elapsed_time(end) / runs

def test_i4_performance():
    """Benchmark I4 vs PyTorch SDPA"""
    
    print("="*80)
    print("TEST: I4 Performance (Burn Methodology)")
    print("="*80)
    print()
    
    try:
        import dhp_i4_kernel
    except ImportError:
        print("⚠️  SKIP: I4 kernel not compiled")
        return None
    
    # Config
    B, H, S, D = 4, 16, 1024, 64
    S_max = 1024
    
    print(f"Configuration: B={B}, H={H}, S={S}, D={D}")
    print()
    
    # Generate inputs
    torch.manual_seed(42)
    Q = torch.randn(B, H, S, D, dtype=torch.float16, device='cuda')
    K = torch.randn(B, H, S, D, dtype=torch.float16, device='cuda')
    V = torch.randn(B, H, S, D, dtype=torch.float16, device='cuda')
    
    # Benchmark PyTorch SDPA
    print("Benchmarking PyTorch SDPA...")
    ms_sdpa = bench(F.scaled_dot_product_attention, Q, K, V)
    us_per_head_sdpa = (ms_sdpa * 1000.0) / H
    
    print(f"  PyTorch SDPA: {ms_sdpa:.3f} ms ({us_per_head_sdpa:.2f} μs/head)")
    print()
    
    # Prepare I4 inputs
    scale = 1.0 / (D ** 0.5)
    scores = torch.matmul(Q, K.transpose(-2, -1)) * scale
    scores_flat = scores.reshape(B*H, S, S)
    V_flat = V.reshape(B*H, S, D)
    
    # Benchmark I4
    print("Benchmarking I4 kernel...")
    ms_i4 = bench(dhp_i4_kernel.forward, scores_flat, V_flat, S_max, S)
    us_per_head_i4 = (ms_i4 * 1000.0) / H
    
    print(f"  I4 kernel:    {ms_i4:.3f} ms ({us_per_head_i4:.2f} μs/head)")
    print()
    
    # Analysis
    speedup = ms_sdpa / ms_i4
    pct_of_baseline = (1.0 / speedup) * 100.0
    
    print("Results:")
    print(f"  Speedup:         {speedup:.2f}×")
    print(f"  % of SDPA:       {pct_of_baseline:.1f}%")
    print()
    
    # Targets from expert review
    target_60pct = 60.0
    target_70pct = 70.0
    
    print("Target Analysis:")
    if pct_of_baseline >= target_70pct:
        print(f"  ✅ EXCELLENT: {pct_of_baseline:.1f}% ≥ {target_70pct}% target")
        return True
    elif pct_of_baseline >= target_60pct:
        print(f"  ✅ GOOD: {pct_of_baseline:.1f}% ≥ {target_60pct}% target")
        return True
    else:
        print(f"  ⚠️  BELOW TARGET: {pct_of_baseline:.1f}% < {target_60pct}%")
        print(f"     Need optimization")
        return False

if __name__ == '__main__':
    result = test_i4_performance()
    sys.exit(0 if result else 1)

