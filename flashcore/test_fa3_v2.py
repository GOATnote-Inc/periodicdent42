#!/usr/bin/env python3
"""
Test FA-3 v2 kernel - Compare against PyTorch SDPA (the baseline to beat)
Goal: Beat 44 μs (PyTorch SDPA on L4)
"""

import torch
import argparse
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

# Parse args
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
print("FA-3 v2 Kernel Test")
print("=" * 70)
print()

# Build
print("[1/4] Building...")
try:
    import build_fa3_v2
    import flashcore_fa3
    print("✅ Build OK")
except Exception as e:
    print(f"❌ Build failed: {e}")
    sys.exit(1)

print()

# Config
print(f"[2/4] Config: B={B}, H={H}, S={S}, D={D}, causal={is_causal}")
print()

# Test data
Q = torch.randn(B, H, S, D, dtype=torch.float16, device="cuda")
K = torch.randn(B, H, S, D, dtype=torch.float16, device="cuda")
V = torch.randn(B, H, S, D, dtype=torch.float16, device="cuda")

# Correctness vs SDPA
print("[3/4] Correctness (vs PyTorch SDPA)...")
try:
    with torch.no_grad():
        ref = torch.nn.functional.scaled_dot_product_attention(
            Q, K, V, is_causal=is_causal
        )
        out = flashcore_fa3.forward(Q, K, V, is_causal)
        
        max_err = (out - ref).abs().max().item()
        mean_err = (out - ref).abs().mean().item()
        
        print(f"  Max error:  {max_err:.6f}")
        print(f"  Mean error: {mean_err:.6f}")
        
        # FP16 tolerance
        if max_err < 0.1 and mean_err < 0.03:
            print("  ✅ PASS (within FP16 tolerance)")
            correct = True
        else:
            print(f"  ❌ FAIL (max {max_err:.4f} >= 0.1 or mean {mean_err:.4f} >= 0.03)")
            correct = False
            
            # Debug
            print("\n  Debug (first row, first 8 values):")
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
        flashcore_fa3.forward(Q, K, V, is_causal)
    torch.cuda.synchronize()
    
    def timeit(op, iters=500):
        times = []
        for _ in range(iters):
            s, e = torch.cuda.Event(True), torch.cuda.Event(True)
            s.record()
            op()
            e.record()
            torch.cuda.synchronize()
            times.append(s.elapsed_time(e) * 1000.0)  # μs
        return statistics.median(times), statistics.quantiles(times, n=10)[8]
    
    fa3_p50, fa3_p90 = timeit(lambda: flashcore_fa3.forward(Q, K, V, is_causal))
    sdpa_p50, sdpa_p90 = timeit(lambda: torch.nn.functional.scaled_dot_product_attention(Q, K, V, is_causal=is_causal))
    
    print(f"  FA-3 v2:     p50={fa3_p50:.2f} μs, p90={fa3_p90:.2f} μs")
    print(f"  PyTorch SDPA: p50={sdpa_p50:.2f} μs, p90={sdpa_p90:.2f} μs")
    
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
print(f"Correctness: {'✅ PASS' if correct else '❌ FAIL'}")
print(f"  Max error: {max_err:.6f}")
print(f"  Mean error: {mean_err:.6f}")
print()

print(f"Performance:")
print(f"  FA-3 v2:      {fa3_p50:.2f} μs")
print(f"  PyTorch SDPA: {sdpa_p50:.2f} μs (baseline to beat)")
print()

# Comparison
if correct:
    improvement = (sdpa_p50 - fa3_p50) / sdpa_p50 * 100.0
    print(f"Δ vs SDPA: {improvement:+.1f}%")
    print()
    
    target_us = 40.0
    
    if fa3_p50 < target_us:
        margin = ((target_us - fa3_p50) / target_us) * 100
        print("🎉 SUCCESS! BEATS TARGET!")
        print(f"   {fa3_p50:.2f} μs < {target_us} μs ({margin:.1f}% margin)")
        if fa3_p50 < sdpa_p50:
            print(f"   Also beats PyTorch SDPA ({improvement:.1f}% faster)")
        print()
        print("🚀 MISSION ACCOMPLISHED!")
        print("   Standing on giants' shoulders and going further!")
        
    elif fa3_p50 < sdpa_p50:
        print(f"✅ BEATS PyTorch SDPA by {improvement:.1f}%!")
        print(f"   {fa3_p50:.2f} μs < {sdpa_p50:.2f} μs")
        print(f"   Close to target (gap: {fa3_p50 - target_us:.2f} μs)")
        print()
        print("Next: Micro-optimizations")
        print("  - Try N_TILE=64 for better occupancy")
        print("  - Add vectorized loads")
        print("  - Consider WMMA for Tensor Cores")
        
    else:
        slowdown = (fa3_p50 - sdpa_p50) / sdpa_p50 * 100
        print(f"⚠️  SLOWER than PyTorch by {slowdown:.1f}%")
        print(f"   {fa3_p50:.2f} μs > {sdpa_p50:.2f} μs")
        print()
        print("Needs further optimization:")
        print("  - Profile with NCU")
        print("  - Check occupancy")
        print("  - Verify no register spills")
else:
    print("❌ Fix correctness first before optimizing performance")

print()

