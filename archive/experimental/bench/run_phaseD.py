#!/usr/bin/env python3
"""
Benchmark Phase D tuned kernel.

Usage:
  python bench/run_phaseD.py --shape S=512,D=64
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.absolute()))

import torch
import time

def benchmark_phaseD(B=1, H=8, S=512, D=64, warmup=10, iters=100):
    """Benchmark Phase D tuned kernel."""
    
    print("=" * 80)
    print("BENCHMARK: Phase D Tuned Kernel")
    print("=" * 80)
    print()
    print(f"Shape: B={B}, H={H}, S={S}, D={D}")
    print()
    
    # Generate test data
    torch.manual_seed(42)
    q = torch.randn(B, H, S, D, device='cuda', dtype=torch.float16)
    k = torch.randn(B, H, S, D, device='cuda', dtype=torch.float16)
    v = torch.randn(B, H, S, D, device='cuda', dtype=torch.float16)
    
    # Build/load module
    try:
        from build_phaseD import build_phaseD
        fa_phaseD_tuned = build_phaseD()
    except Exception as e:
        print(f"❌ Module build/load failed: {e}")
        sys.exit(1)
    
    # Warmup
    print(f"Warmup ({warmup} iterations)...")
    for _ in range(warmup):
        _ = fa_phaseD_tuned.forward(q, k, v)
    torch.cuda.synchronize()
    print("  ✅ Done")
    print()
    
    # Benchmark
    print(f"Benchmarking ({iters} iterations)...")
    torch.cuda.synchronize()
    start = time.perf_counter()
    
    for _ in range(iters):
        out = fa_phaseD_tuned.forward(q, k, v)
    
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    
    latency_us = (elapsed / iters) * 1e6
    print(f"  ✅ Done")
    print()
    
    # Correctness check
    print("Checking correctness...")
    ref = torch.nn.functional.scaled_dot_product_attention(
        q, k, v, attn_mask=None, dropout_p=0.0, is_causal=False
    )
    
    max_diff = (out - ref).abs().max().item()
    correct = max_diff <= 2e-3
    
    print(f"  Max diff: {max_diff:.6f}")
    print(f"  Correct: {'✅' if correct else '❌'}")
    print()
    
    # Results
    print("=" * 80)
    print("RESULTS")
    print("=" * 80)
    print(f"Latency:    {latency_us:.2f} μs")
    print(f"Correct:    {correct}")
    print(f"Max diff:   {max_diff:.6f}")
    print()
    
    # Compare vs xFormers champion
    print("Comparison vs xFormers Champion (24.22 μs):")
    speedup = 24.22 / latency_us
    if speedup > 1.0:
        print(f"  ✅ {speedup:.2f}× faster than xFormers!")
    elif speedup > 0.95:
        print(f"  ✅ {speedup:.2f}× (within 5% of xFormers - excellent!)")
    else:
        print(f"  ⚠️ {1/speedup:.2f}× slower than xFormers")
    print()
    
    return {
        'latency_us': latency_us,
        'correct': correct,
        'max_diff': max_diff,
        'speedup_vs_xformers': speedup,
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--shape', default='S=512,D=64')
    parser.add_argument('--warmup', type=int, default=10)
    parser.add_argument('--iters', type=int, default=100)
    args = parser.parse_args()
    
    # Parse shape
    shape = {}
    for part in args.shape.split(','):
        k, v = part.split('=')
        shape[k] = int(v)
    
    S = shape.get('S', 512)
    D = shape.get('D', 64)
    
    results = benchmark_phaseD(
        B=1, H=8, S=S, D=D,
        warmup=args.warmup,
        iters=args.iters,
    )
    
    return 0 if results['correct'] else 1

if __name__ == '__main__':
    sys.exit(main())

