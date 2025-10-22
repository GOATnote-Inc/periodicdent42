#!/usr/bin/env python3
"""Test simplified FA-3 kernel - Focus on correctness first"""

import torch
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

print("=" * 70)
print("FA-3 Simple Kernel Test (Debug Version)")
print("=" * 70)
print()

# Build
print("[1/4] Building...")
try:
    import build_fa3_simple
    import flashcore_fa3_simple
    print("✅ Build OK")
except Exception as e:
    print(f"❌ Build failed: {e}")
    sys.exit(1)

print()

# Test config
B, H, S, D = 1, 8, 512, 64
print(f"[2/4] Config: B={B}, H={H}, S={S}, D={D}")
print()

# Test data
Q = torch.randn(B, H, S, D, dtype=torch.float16, device='cuda')
K = torch.randn(B, H, S, D, dtype=torch.float16, device='cuda')
V = torch.randn(B, H, S, D, dtype=torch.float16, device='cuda')

# Correctness
print("[3/4] Correctness...")
try:
    O_fa3 = flashcore_fa3_simple.forward(Q, K, V)
    O_ref = torch.nn.functional.scaled_dot_product_attention(Q, K, V)
    
    max_err = (O_fa3 - O_ref).abs().max().item()
    mean_err = (O_fa3 - O_ref).abs().mean().item()
    rel_err = ((O_fa3 - O_ref).abs() / (O_ref.abs() + 1e-8)).mean().item()
    
    print(f"  Max error:  {max_err:.6f}")
    print(f"  Mean error: {mean_err:.6f}")
    print(f"  Rel error:  {rel_err:.6f}")
    
    if max_err < 0.1:
        print("  ✅ PASS")
        correct = True
    else:
        print(f"  ❌ FAIL (max_err {max_err:.4f} >= 0.1)")
        correct = False
        
        # Debug: print first few values
        print("\n  Debug (first 5 output values):")
        print(f"  FA-3:  {O_fa3[0,0,0,:5].tolist()}")
        print(f"  PyTorch: {O_ref[0,0,0,:5].tolist()}")
        
except Exception as e:
    print(f"  ❌ Exception: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print()

# Performance
print("[4/4] Performance...")
try:
    # Warmup
    for _ in range(50):
        O = flashcore_fa3_simple.forward(Q, K, V)
    torch.cuda.synchronize()
    
    # Benchmark
    times = []
    for _ in range(500):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        O = flashcore_fa3_simple.forward(Q, K, V)
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end) * 1000)
    
    p50 = statistics.median(times)
    p90 = statistics.quantiles(times, n=10)[8]
    
    print(f"  p50: {p50:.2f} μs")
    print(f"  p90: {p90:.2f} μs")
    
except Exception as e:
    print(f"  ❌ Benchmark failed: {e}")
    p50 = float('inf')

print()
print("=" * 70)
print("RESULTS")
print("=" * 70)

pytorch_us = 44.10
target_us = 40.0

print(f"\nCorrectness: {'✅ PASS' if correct else '❌ FAIL'}")
print(f"  Max error: {max_err:.6f}")
print()

print(f"Performance:")
print(f"  FA-3 Simple: {p50:.2f} μs")
print(f"  PyTorch SDPA: {pytorch_us:.2f} μs")
print(f"  Target: <{target_us} μs")
print()

if correct and p50 < target_us:
    print("🎉 SUCCESS! Beats target!")
    print(f"   {p50:.2f} < {target_us} μs")
elif correct and p50 < pytorch_us:
    print("✅ BEATS PyTorch!")
    margin = ((pytorch_us - p50) / pytorch_us) * 100
    print(f"   {p50:.2f} < {pytorch_us:.2f} μs ({margin:.1f}% faster)")
elif correct:
    print("✅ Correct, but slower than PyTorch")
    print(f"   {p50:.2f} > {pytorch_us:.2f} μs")
    print("\n   Next: Add optimizations")
    print("   - Double-buffering")
    print("   - WMMA for Tensor Cores")
    print("   - Tune tile sizes")
else:
    print("❌ Correctness failed - fix algorithm first")

print()

