#!/usr/bin/env python3
"""Test script for FlashCore 64x64 kernel."""

import torch
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from build_64x64 import build_64x64

def test_flashcore_64x64():
    """Test the 64x64 kernel."""
    
    print("=" * 80)
    print("FlashCore 64×64 Tiles + FP32 P - Test")
    print("=" * 80)
    print()
    
    # Build kernel
    print("Building kernel...")
    ext = build_64x64()
    print("✅ Build successful")
    print()
    
    # Test configuration
    device = torch.device('cuda')
    dtype = torch.float16
    
    test_shapes = [
        ("mission", 1, 8, 512, 64),
        ("short", 1, 8, 256, 64),
        ("long", 1, 8, 1024, 64),
    ]
    
    results = {"name": "FlashCore 64x64", "shapes": {}}
    
    for name, B, H, S, D in test_shapes:
        print(f"Testing shape: {name} (B={B}, H={H}, S={S}, D={D})")
        
        # Create inputs with fixed seed for reproducibility
        torch.manual_seed(42)
        Q = torch.randn(B, H, S, D, device=device, dtype=dtype)
        K = torch.randn(B, H, S, D, device=device, dtype=dtype)
        V = torch.randn(B, H, S, D, device=device, dtype=dtype)
        
        # Allocate output
        O_kernel = torch.zeros(B, H, S, D, device=device, dtype=dtype)
        
        # Warmup
        for _ in range(5):
            ext.forward(Q, K, V, O_kernel)
        torch.cuda.synchronize()
        
        # Run kernel
        ext.forward(Q, K, V, O_kernel)
        torch.cuda.synchronize()
        
        # PyTorch reference
        scale = 1.0 / (D ** 0.5)
        scores = torch.matmul(Q, K.transpose(-2, -1)) * scale
        attn = torch.softmax(scores, dim=-1)
        O_ref = torch.matmul(attn, V)
        
        # Check correctness
        max_err = (O_ref - O_kernel).abs().max().item()
        mean_err = (O_ref - O_kernel).abs().mean().item()
        rel_err = ((O_ref - O_kernel).abs() / (O_ref.abs() + 1e-6)).mean().item()
        
        print(f"  Correctness:")
        print(f"    max_err:  {max_err:.6f}")
        print(f"    mean_err: {mean_err:.6f}")
        print(f"    rel_err:  {rel_err:.6f}")
        
        # Performance (20 warmup + 100 test)
        torch.cuda.synchronize()
        for _ in range(20):
            ext.forward(Q, K, V, O_kernel)
        
        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        
        # Measure 100 iterations
        times = []
        for _ in range(100):
            start.record()
            ext.forward(Q, K, V, O_kernel)
            end.record()
            torch.cuda.synchronize()
            times.append(start.elapsed_time(end) * 1000)  # Convert to microseconds
        
        # Statistics
        times_sorted = sorted(times)
        p50 = times_sorted[50]
        p90 = times_sorted[90]
        p99 = times_sorted[99]
        
        print(f"  Performance:")
        print(f"    p50: {p50:.2f} μs")
        print(f"    p90: {p90:.2f} μs")
        print(f"    p99: {p99:.2f} μs")
        
        # Check pass/fail
        threshold = 0.10  # Target for FP32 P
        passed = max_err < threshold
        print(f"  {'✅ PASS' if passed else '❌ FAIL'}: max_err {'<' if passed else '>='} {threshold}")
        
        # Performance analysis
        if name == "mission":
            speedup_vs_baseline = 1398 / p50  # vs original baseline
            speedup_vs_current = 279 / p50    # vs current 32x32
            print(f"  Analysis:")
            print(f"    Speedup vs baseline: {speedup_vs_baseline:.1f}×")
            print(f"    Speedup vs 32×32:    {speedup_vs_current:.1f}×")
            
            if p50 < 140:
                print(f"    🎯 ON TRACK for <40 μs target!")
            elif p50 < 160:
                print(f"    ⚠️ Slightly behind, but still good")
            else:
                print(f"    ❌ Performance regression, investigate")
        
        print()
        
        results["shapes"][name] = {
            "max_err": max_err,
            "mean_err": mean_err,
            "rel_err": rel_err,
            "p50_us": p50,
            "p90_us": p90,
            "p99_us": p99,
            "passed": passed
        }
    
    # Summary
    print("=" * 80)
    print("FlashCore 64×64 - Test Summary")
    print("=" * 80)
    print()
    
    all_passed = all(r["passed"] for r in results["shapes"].values())
    mission_perf = results["shapes"]["mission"]["p50_us"]
    mission_err = results["shapes"]["mission"]["max_err"]
    
    print(f"Mission shape results:")
    print(f"  Performance: {mission_perf:.2f} μs")
    print(f"  Error:       {mission_err:.6f}")
    print()
    
    if all_passed and mission_perf < 140:
        print("✅ PHASE 2A SUCCESS!")
        print()
        print("Achievements:")
        print(f"  ✓ Error fixed:      0.51 → {mission_err:.3f} ({0.51/mission_err:.1f}× improvement)")
        print(f"  ✓ Performance gain: 279 → {mission_perf:.0f} μs ({279/mission_perf:.1f}× speedup)")
        print(f"  ✓ Both goals met in single optimization!")
        print()
        print("Next steps:")
        print("  → Phase 2B: cp.async pipeline (target: 55 μs)")
        print("  → Phase 2C: Micro-optimizations (target: 38 μs)")
        print("  → Final: <40 μs ACHIEVED! 🎯")
    elif all_passed:
        print("⚠️ PARTIAL SUCCESS - Error fixed but performance needs work")
        print(f"  Current: {mission_perf:.0f} μs (target was <140)")
        print("  Investigate: Memory bottleneck, occupancy")
    else:
        print("❌ TESTS FAILED - Debug required")
        print("  Check: Union usage, synchronization barriers")
    
    return results

if __name__ == "__main__":
    results = test_flashcore_64x64()

