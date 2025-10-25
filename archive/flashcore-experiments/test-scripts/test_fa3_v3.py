#!/usr/bin/env python3
"""
Test FA-3 v3 kernel (inverted loops + optimized)
Goal: Beat PyTorch SDPA (44-45 μs) → Target <40 μs
"""

import torch
import argparse
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

parser = argparse.ArgumentParser()
parser.add_argument("--B", type=int, default=1)
parser.add_argument("--H", type=int, default=8)
parser.add_argument("--S", type=int, default=512)
parser.add_argument("--D", type=int, default=64)
parser.add_argument("--causal", action="store_true")
args = parser.parse_args()

B, H, S, D = args.B, args.H, args.S, args.D
is_causal = args.causal

print("=" * 70)
print("FA-3 v3 Kernel Test (Inverted Loops + Optimized)")
print("=" * 70)
print()

# Build
print("[1/4] Building...")
try:
    import build_fa3_v3
    import flashcore_fa3_v3
    print("✅ Build OK")
except Exception as e:
    print(f"❌ Build failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print()

# Config
print(f"[2/4] Config: B={B}, H={H}, S={S}, D={D}, causal={is_causal}")
print()

# Test data
Q = torch.randn(B, H, S, D, dtype=torch.float16, device="cuda")
K = torch.randn(B, H, S, D, dtype=torch.float16, device="cuda")
V = torch.randn(B, H, S, D, dtype=torch.float16, device="cuda")

# Correctness
print("[3/4] Correctness...")
try:
    with torch.no_grad():
        ref = torch.nn.functional.scaled_dot_product_attention(Q, K, V, is_causal=is_causal)
        out = flashcore_fa3_v3.forward(Q, K, V, is_causal)
        
        max_err = (out - ref).abs().max().item()
        mean_err = (out - ref).abs().mean().item()
        
        print(f"  Max error:  {max_err:.6f}")
        print(f"  Mean error: {mean_err:.6f}")
        
        if max_err < 0.1 and mean_err < 0.03:
            print("  ✅ PASS")
            correct = True
        else:
            print(f"  ❌ FAIL (max {max_err:.4f} or mean {mean_err:.4f} out of tolerance)")
            correct = False
            print("\n  First 8 values:")
            print(f"  FA-3: {out[0,0,0,:8].tolist()}")
            print(f"  SDPA: {ref[0,0,0,:8].tolist()}")
            
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
    for _ in range(100):
        flashcore_fa3_v3.forward(Q, K, V, is_causal)
    torch.cuda.synchronize()
    
    def benchmark(fn, iters=500):
        times = []
        for _ in range(iters):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            fn()
            end.record()
            torch.cuda.synchronize()
            times.append(start.elapsed_time(end) * 1000.0)
        p50 = statistics.median(times)
        p90 = statistics.quantiles(times, n=10)[8]
        return p50, p90
    
    fa3_p50, fa3_p90 = benchmark(lambda: flashcore_fa3_v3.forward(Q, K, V, is_causal))
    sdpa_p50, sdpa_p90 = benchmark(lambda: torch.nn.functional.scaled_dot_product_attention(Q, K, V, is_causal=is_causal))
    
    print(f"  FA-3 v3:     p50={fa3_p50:.2f} μs, p90={fa3_p90:.2f} μs")
    print(f"  PyTorch SDPA: p50={sdpa_p50:.2f} μs, p90={sdpa_p90:.2f} μs")
    
except Exception as e:
    print(f"  ❌ Benchmark failed: {e}")
    import traceback
    traceback.print_exc()
    fa3_p50 = float('inf')
    sdpa_p50 = 45.0

print()
print("=" * 70)
print("FINAL RESULTS")
print("=" * 70)
print()

print(f"Correctness: {'✅ PASS' if correct else '❌ FAIL'}")
print(f"  Max error: {max_err:.6f}")
print(f"  Mean error: {mean_err:.6f}")
print()

print(f"Performance:")
print(f"  FA-3 v3:      {fa3_p50:.2f} μs")
print(f"  PyTorch SDPA: {sdpa_p50:.2f} μs (baseline)")
print()

if correct:
    improvement = (sdpa_p50 - fa3_p50) / sdpa_p50 * 100.0
    print(f"Improvement: {improvement:+.1f}%")
    print()
    
    target = 40.0
    
    if fa3_p50 < target:
        margin = ((target - fa3_p50) / target) * 100
        print("🎉 SUCCESS! BEATS TARGET!")
        print(f"   {fa3_p50:.2f} μs < {target} μs ({margin:.1f}% margin)")
        if fa3_p50 < sdpa_p50:
            print(f"   Also beats PyTorch SDPA ({improvement:.1f}% faster)")
        print()
        print("🚀 MISSION ACCOMPLISHED!")
        print("   22 days of research delivered!")
        print("   Standing on giants' shoulders! 💪")
        
    elif fa3_p50 < sdpa_p50:
        print(f"✅ BEATS PyTorch SDPA by {improvement:.1f}%!")
        print(f"   {fa3_p50:.2f} μs < {sdpa_p50:.2f} μs")
        gap = fa3_p50 - target
        print(f"   Close to target (gap: {gap:.2f} μs)")
        print()
        print("Next steps to reach <40 μs:")
        print("  1. Profile with NCU to find bottlenecks")
        print("  2. Add WMMA for Q·K^T (10-20% faster)")
        print("  3. Vectorize global loads (5-10% faster)")
        print("  4. Tune N_TILE (try 32, 128)")
        print("  5. Reduce register pressure")
        
    elif fa3_p50 < 100:
        slowdown = (fa3_p50 - sdpa_p50) / sdpa_p50 * 100
        print(f"⚠️  Slower than PyTorch by {slowdown:.1f}%")
        print(f"   But significant improvement from v2 (5259 → {fa3_p50:.2f} μs)")
        print()
        print("Progress:")
        print(f"  v2 (per-row reload): 5259 μs")
        print(f"  v3 (inverted loops):  {fa3_p50:.2f} μs")
        print(f"  Speedup: {5259/fa3_p50:.1f}×")
        print()
        print("Next: Add WMMA + micro-optimizations")
        
    else:
        print(f"⚠️  Needs investigation: {fa3_p50:.2f} μs")
        print("  Run NCU profiling to identify bottlenecks")
else:
    print("❌ Fix correctness before optimizing")

print()

