#!/usr/bin/env python3
"""
Baseline Attention Benchmark

Tests all registered baselines for:
- Correctness (vs MATH backend)
- Performance (latency in ms)

Usage:
    python scripts/bench_baselines.py
    python scripts/bench_baselines.py --shape 2 16 1024 64
    python scripts/bench_baselines.py --iters 100
"""

import argparse
import sys
import time
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.absolute()))

import torch
import math

from baselines import registry

def rand_inputs(B=1, H=8, S=512, D=64, dtype=torch.float16, device='cuda'):
    """Generate random Q, K, V tensors"""
    torch.manual_seed(42)  # Reproducible
    q = torch.randn(B, H, S, D, device=device, dtype=dtype)
    k = torch.randn(B, H, S, D, device=device, dtype=dtype)
    v = torch.randn(B, H, S, D, device=device, dtype=dtype)
    return q, k, v

def run_one(fn, q, k, v, iters=50, warmup=10):
    """
    Benchmark a single baseline
    
    Returns:
        ms: Average latency in milliseconds
        err: Max absolute error vs reference
        success: Whether the test succeeded
    """
    try:
        # Warmup
        for _ in range(warmup):
            _ = fn(q, k, v, causal=False, dropout_p=0.0)
        torch.cuda.synchronize()
        
        # Benchmark
        start = time.perf_counter()
        for _ in range(iters):
            _ = fn(q, k, v, causal=False, dropout_p=0.0)
        torch.cuda.synchronize()
        end = time.perf_counter()
        
        ms = (end - start) * 1000 / iters
        
        # Correctness check (vs MATH backend)
        try:
            ref = registry.get("pytorch_sdpa_math").fn(q, k, v, causal=False, dropout_p=0.0)
            out = fn(q, k, v, causal=False, dropout_p=0.0)
            err = (out.float() - ref.float()).abs().max().item()
        except Exception as e:
            # If MATH backend fails, skip correctness check
            err = -1.0
            print(f"    ⚠️  Correctness check skipped: {e}")
        
        return ms, err, True
    
    except Exception as e:
        print(f"    ❌ FAILED: {e}")
        return -1.0, -1.0, False

def main():
    parser = argparse.ArgumentParser(description="Benchmark attention baselines")
    parser.add_argument('--shape', nargs=4, type=int, default=[1, 8, 512, 64],
                        metavar=('B', 'H', 'S', 'D'),
                        help='Input shape (default: 1 8 512 64)')
    parser.add_argument('--iters', type=int, default=50,
                        help='Benchmark iterations (default: 50)')
    parser.add_argument('--warmup', type=int, default=10,
                        help='Warmup iterations (default: 10)')
    parser.add_argument('--dtype', choices=['float16', 'bfloat16'], default='float16',
                        help='Data type (default: float16)')
    args = parser.parse_args()
    
    B, H, S, D = args.shape
    dtype = torch.float16 if args.dtype == 'float16' else torch.bfloat16
    
    print("=" * 80)
    print("BASELINE ATTENTION BENCHMARK (L4 / sm_89)")
    print("=" * 80)
    print()
    print(f"Configuration:")
    print(f"  Shape: B={B}, H={H}, S={S}, D={D}")
    print(f"  Dtype: {args.dtype}")
    print(f"  Iterations: {args.iters} (warmup: {args.warmup})")
    print()
    
    # Verify CUDA
    if not torch.cuda.is_available():
        print("❌ CUDA not available!")
        return 1
    
    cc = torch.cuda.get_device_capability()
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Compute Capability: sm_{cc[0]}{cc[1]}")
    if cc[0] * 10 + cc[1] != 89:
        print(f"⚠️  Expected sm_89 (Ada/L4), got sm_{cc[0]}{cc[1]}")
    print()
    
    # Generate inputs
    print("Generating test data...")
    q, k, v = rand_inputs(B, H, S, D, dtype=dtype)
    print(f"  Q: {q.shape}, {q.dtype}")
    print(f"  K: {k.shape}, {k.dtype}")
    print(f"  V: {v.shape}, {v.dtype}")
    print()
    
    # Test all baselines
    print("=" * 80)
    print("RESULTS")
    print("=" * 80)
    print()
    print(f"{'Baseline':<30} {'Latency':<12} {'Max Error':<12} {'Status'}")
    print("-" * 80)
    
    results = []
    
    # Priority order: try these first
    priority_names = [
        "pytorch_sdpa_flash",
        "pytorch_sdpa_cudnn", 
        "flashattn2",
        "pytorch_sdpa_efficient",
        "pytorch_sdpa_math",
    ]
    
    for name in priority_names:
        if name not in registry.REGISTRY:
            print(f"{name:<30} {'N/A':<12} {'N/A':<12} NOT REGISTERED")
            continue
        
        try:
            baseline = registry.get(name)
            ms, err, success = run_one(baseline.fn, q, k, v, iters=args.iters, warmup=args.warmup)
            
            if success:
                latency_str = f"{ms:.3f} ms" if ms > 0 else "N/A"
                err_str = f"{err:.2e}" if err >= 0 else "N/A"
                status = "✅ OK"
                results.append((ms, err, name))
            else:
                latency_str = "FAILED"
                err_str = "N/A"
                status = "❌ ERROR"
            
            print(f"{name:<30} {latency_str:<12} {err_str:<12} {status}")
        
        except Exception as e:
            print(f"{name:<30} {'FAILED':<12} {'N/A':<12} ❌ {e}")
    
    print()
    
    # Summary
    if results:
        results.sort(key=lambda x: x[0])  # Sort by latency
        print("=" * 80)
        print("CHAMPION")
        print("=" * 80)
        print()
        print(f"🏆 Fastest: {results[0][2]}")
        print(f"   Latency: {results[0][0]:.3f} ms ({results[0][0]*1000:.2f} μs)")
        print(f"   Max Error: {results[0][1]:.2e}")
        print()
        
        # Top 3
        print("Top 3:")
        for i, (ms, err, name) in enumerate(results[:3], 1):
            print(f"  {i}. {name:<30} {ms:.3f} ms (max_err={err:.2e})")
        print()
        
        return 0
    else:
        print("=" * 80)
        print("❌ NO SUCCESSFUL BASELINES")
        print("=" * 80)
        print()
        print("All baselines failed. Check:")
        print("  - CUDA installation")
        print("  - PyTorch version (>= 2.0)")
        print("  - flash-attn installation")
        print()
        return 1

if __name__ == "__main__":
    sys.exit(main())

