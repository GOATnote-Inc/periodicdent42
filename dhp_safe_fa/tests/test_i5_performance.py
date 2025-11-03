#!/usr/bin/env python3
"""
I4 vs I5 Performance Comparison
================================

Compares:
- PyTorch SDPA (baseline)
- I4 kernel (non-coalesced memory)
- I5 kernel (warp-cooperative loading)

Expected results:
- PyTorch SDPA: 3.62 μs/head
- I4 kernel: 158 μs/head (43× slower)
- I5 kernel: 5-6 μs/head (1.4-1.6× slower) ✅ TARGET
"""

import torch
import torch.nn.functional as F
import sys
import os

# Add parent to path
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

def benchmark_kernel(name, kernel_fn, warmup=10, num_runs=100):
    """Benchmark a kernel using CUDA events"""
    # Warmup
    for _ in range(warmup):
        _ = kernel_fn()
    torch.cuda.synchronize()
    
    # Measure
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    
    start_event.record()
    for _ in range(num_runs):
        _ = kernel_fn()
    end_event.record()
    torch.cuda.synchronize()
    
    elapsed_ms = start_event.elapsed_time(end_event) / num_runs
    return elapsed_ms

def main():
    print("="*80)
    print("I4 vs I5 Performance Comparison")
    print("="*80)
    print()
    
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
    
    # Compute scores for I4/I5
    scale = 1.0 / (D ** 0.5)
    scores = torch.matmul(Q, K.transpose(-2, -1)) * scale
    scores_flat = scores.reshape(B*H, S, S)
    V_flat = V.reshape(B*H, S, D)
    
    # ========================================================================
    # Benchmark 1: PyTorch SDPA (baseline)
    # ========================================================================
    print("Benchmarking PyTorch SDPA...")
    def pytorch_sdpa():
        return F.scaled_dot_product_attention(Q, K, V, is_causal=True)
    
    pytorch_ms = benchmark_kernel("PyTorch SDPA", pytorch_sdpa)
    pytorch_us_per_head = (pytorch_ms * 1000) / H
    print(f"  PyTorch SDPA: {pytorch_ms:.3f} ms ({pytorch_us_per_head:.2f} μs/head)")
    print()
    
    # ========================================================================
    # Benchmark 2: I4 Kernel
    # ========================================================================
    try:
        import dhp_i4_kernel
        
        print("Benchmarking I4 kernel...")
        def i4_kernel():
            return dhp_i4_kernel.forward(scores_flat, V_flat, S_max, S)
        
        i4_ms = benchmark_kernel("I4 kernel", i4_kernel)
        i4_us_per_head = (i4_ms * 1000) / H
        print(f"  I4 kernel:    {i4_ms:.3f} ms ({i4_us_per_head:.2f} μs/head)")
        
        i4_speedup = pytorch_ms / i4_ms
        i4_pct = (i4_ms / pytorch_ms) * 100
        print(f"  Speedup:      {i4_speedup:.2f}×")
        print(f"  % of SDPA:    {i4_pct:.1f}%")
        print()
        
    except ImportError:
        print("⚠️  I4 kernel not compiled")
        i4_ms = None
        i4_us_per_head = None
        print()
    
    # ========================================================================
    # Benchmark 3: I5 Kernel (warp-cooperative)
    # ========================================================================
    try:
        import dhp_i5_kernel
        
        print("Benchmarking I5 kernel (warp-cooperative)...")
        def i5_kernel():
            return dhp_i5_kernel.forward(scores_flat, V_flat, S_max, S)
        
        i5_ms = benchmark_kernel("I5 kernel", i5_kernel)
        i5_us_per_head = (i5_ms * 1000) / H
        print(f"  I5 kernel:    {i5_ms:.3f} ms ({i5_us_per_head:.2f} μs/head)")
        
        i5_speedup = pytorch_ms / i5_ms
        i5_pct = (i5_ms / pytorch_ms) * 100
        print(f"  Speedup:      {i5_speedup:.2f}×")
        print(f"  % of SDPA:    {i5_pct:.1f}%")
        print()
        
        # Correctness check
        print("Correctness check...")
        out_ref = pytorch_sdpa()
        out_i5 = i5_kernel().reshape(B, H, S, D)
        
        max_diff = torch.abs(out_i5 - out_ref).max().item()
        mean_diff = torch.abs(out_i5 - out_ref).mean().item()
        print(f"  Max diff:  {max_diff:.6f}")
        print(f"  Mean diff: {mean_diff:.6f}")
        
        if max_diff < 2e-3:
            print("  ✅ I5 correctness PASSED")
        else:
            print(f"  ❌ I5 correctness FAILED (max_diff={max_diff:.6f})")
        print()
        
        # Compare I4 vs I5
        if i4_ms is not None:
            i5_vs_i4 = i4_ms / i5_ms
            print(f"I5 vs I4:")
            print(f"  Speedup: {i5_vs_i4:.1f}×")
            print(f"  I4: {i4_us_per_head:.2f} μs/head")
            print(f"  I5: {i5_us_per_head:.2f} μs/head")
            print()
        
    except ImportError:
        print("⚠️  I5 kernel not compiled")
        i5_ms = None
        i5_us_per_head = None
        print()
    
    # ========================================================================
    # Summary
    # ========================================================================
    print("="*80)
    print("SUMMARY")
    print("="*80)
    print()
    print(f"PyTorch SDPA:  {pytorch_us_per_head:.2f} μs/head (baseline)")
    
    if i4_us_per_head is not None:
        print(f"I4 kernel:     {i4_us_per_head:.2f} μs/head ({(i4_us_per_head/pytorch_us_per_head):.1f}× slower)")
    
    if i5_us_per_head is not None:
        print(f"I5 kernel:     {i5_us_per_head:.2f} μs/head ({(i5_us_per_head/pytorch_us_per_head):.1f}× slower)")
        print()
        
        # Target assessment
        TARGET_MIN = 5.0  # μs/head
        TARGET_MAX = 6.0  # μs/head
        
        if i5_us_per_head >= TARGET_MIN and i5_us_per_head <= TARGET_MAX:
            print(f"✅ I5 ACHIEVED TARGET: {i5_us_per_head:.2f} μs/head ({TARGET_MIN}-{TARGET_MAX} μs/head)")
        elif i5_us_per_head < TARGET_MIN:
            print(f"🎉 I5 EXCEEDED TARGET: {i5_us_per_head:.2f} μs/head (better than {TARGET_MIN} μs/head)")
        else:
            print(f"⚠️  I5 BELOW TARGET: {i5_us_per_head:.2f} μs/head (target: {TARGET_MIN}-{TARGET_MAX} μs/head)")

if __name__ == '__main__':
    main()

