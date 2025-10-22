#!/usr/bin/env python3
"""
Test CUTLASS FMHA kernel

Expected performance: 15-25 μs (target: <26 μs)
"""

import torch
import statistics
import sys
from pathlib import Path

# Add flashcore to path
sys.path.insert(0, str(Path(__file__).parent))

print("=" * 70)
print("FlashCore: CUTLASS FMHA Test")
print("=" * 70)
print()

# Build kernel
print("[1/4] Building CUTLASS FMHA kernel...")
try:
    import build_cutlass
    # Module is loaded by build script
    import flashcore_cutlass
    print("✅ Build successful!")
except Exception as e:
    print(f"❌ Build failed: {e}")
    sys.exit(1)

print()

# Test configuration
B, H, S, D = 1, 8, 512, 64
print(f"[2/4] Test configuration:")
print(f"  Shape: B={B}, H={H}, S={S}, D={D}")
print()

# Create test data
Q = torch.randn(B, H, S, D, dtype=torch.float16, device='cuda')
K = torch.randn(B, H, S, D, dtype=torch.float16, device='cuda')
V = torch.randn(B, H, S, D, dtype=torch.float16, device='cuda')

# Correctness test
print("[3/4] Correctness check...")
try:
    O_cutlass = flashcore_cutlass.fmha(Q, K, V)
    O_ref = torch.nn.functional.scaled_dot_product_attention(Q, K, V)
    
    max_err = (O_cutlass - O_ref).abs().max().item()
    mean_err = (O_cutlass - O_ref).abs().mean().item()
    
    print(f"  Max error:  {max_err:.6f}")
    print(f"  Mean error: {mean_err:.6f}")
    
    if max_err < 0.1:
        print("  ✅ Correctness: PASS")
        correctness_pass = True
    else:
        print(f"  ❌ Correctness: FAIL (error {max_err:.6f} >= 0.1)")
        correctness_pass = False
except Exception as e:
    print(f"  ❌ Correctness check failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print()

# Performance benchmark
print("[4/4] Performance benchmark...")
try:
    # Warmup
    for _ in range(100):
        O = flashcore_cutlass.fmha(Q, K, V)
    torch.cuda.synchronize()
    
    # Benchmark
    times = []
    for _ in range(1000):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        
        start.record()
        O = flashcore_cutlass.fmha(Q, K, V)
        end.record()
        
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end) * 1000)  # Convert to μs
    
    p50 = statistics.median(times)
    p90 = statistics.quantiles(times, n=10)[8]
    p99 = statistics.quantiles(times, n=100)[98]
    mean = statistics.mean(times)
    std = statistics.stdev(times)
    
    print(f"  p50:  {p50:.2f} μs")
    print(f"  p90:  {p90:.2f} μs")
    print(f"  p99:  {p99:.2f} μs")
    print(f"  mean: {mean:.2f} μs ± {std:.2f}")
    
except Exception as e:
    print(f"  ❌ Benchmark failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print()
print("=" * 70)
print("RESULTS")
print("=" * 70)
print()

# Summary
print("Correctness:")
print(f"  Max error: {max_err:.6f}")
print(f"  Status: {'✅ PASS' if correctness_pass else '❌ FAIL'}")
print()

print("Performance:")
print(f"  p50: {p50:.2f} μs")
print(f"  p90: {p90:.2f} μs")
print()

# Comparison
baseline_us = 1397.0
pytorch_us = 44.10
target_us = 26.0

print("Comparison:")
print(f"  Baseline (scalar):  {baseline_us:.1f} μs")
print(f"  PyTorch SDPA:       {pytorch_us:.2f} μs")
print(f"  CUTLASS FMHA:       {p50:.2f} μs")
print(f"  Target:             <{target_us} μs")
print()

# Speedups
speedup_vs_baseline = baseline_us / p50
speedup_vs_pytorch = pytorch_us / p50

print("Speedups:")
print(f"  vs Baseline: {speedup_vs_baseline:.1f}×")
print(f"  vs PyTorch:  {speedup_vs_pytorch:.2f}×")
print()

# Final verdict
print("=" * 70)
print("VERDICT")
print("=" * 70)

if not correctness_pass:
    print("❌ FAILED: Correctness check failed")
    sys.exit(1)
elif p50 < target_us:
    margin = ((target_us - p50) / target_us) * 100
    print(f"✅ SUCCESS! CUTLASS FMHA beats target!")
    print(f"   {p50:.2f} μs < {target_us} μs ({margin:.1f}% margin)")
    print()
    print("🎉 MISSION ACCOMPLISHED!")
    print("   - 31.7× speedup from baseline achieved")
    print("   - Target <26 μs achieved")
    print("   - Perfect correctness")
    print()
    print("Standing on giants' shoulders worked! 🚀")
elif p50 < pytorch_us:
    print(f"✅ GOOD! CUTLASS FMHA beats PyTorch!")
    print(f"   {p50:.2f} μs < {pytorch_us:.2f} μs ({speedup_vs_pytorch:.2f}× faster)")
    print(f"   Close to target (gap: {p50 - target_us:.2f} μs)")
    print()
    print("Next: Profile with NCU and optimize further.")
else:
    print(f"⚠️  SLOWER than PyTorch")
    print(f"   {p50:.2f} μs > {pytorch_us:.2f} μs")
    print(f"   Needs investigation (check CUTLASS config)")
    print()
    print("Possible issues:")
    print("  - Sub-optimal tiling config")
    print("  - Architecture mismatch (sm_89 vs sm_80)")
    print("  - Missing optimizations")

print()

