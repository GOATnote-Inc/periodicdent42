#!/usr/bin/env python3
"""Test script for FlashCore cp.async kernel."""

import torch
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from build_cpasync import build_cpasync

def test_flashcore_cpasync():
    """Test the cp.async kernel."""
    
    print("=" * 80)
    print("FlashCore cp.async (64×32 tiles) - Test")
    print("=" * 80)
    print()
    
    # Build kernel
    print("Building kernel...")
    ext = build_cpasync()
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
    
    results = {"name": "FlashCore cp.async", "shapes": {}}
    
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
        threshold = 0.40  # Similar to 32×32 baseline
        passed = max_err < threshold
        print(f"  {'✅ PASS' if passed else '❌ FAIL'}: max_err {'<' if passed else '>='} {threshold}")
        
        # Performance analysis
        if name == "mission":
            speedup_vs_baseline = 1398 / p50  # vs original baseline
            speedup_vs_32x32 = 279 / p50      # vs 32×32 FP16
            print(f"  Analysis:")
            print(f"    Speedup vs baseline (1398 μs): {speedup_vs_baseline:.2f}×")
            print(f"    Speedup vs 32×32 (279 μs):     {speedup_vs_32x32:.2f}×")
            
            if p50 < 100:
                print(f"    🎯 🎯 🎯 EXCELLENT! On track for <40 μs with micro-tuning!")
            elif p50 < 150:
                print(f"    ✅ GOOD! Expected 110-140 μs range")
            elif p50 < 200:
                print(f"    ⚠️  Slightly slower than expected, but still progress")
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
    print("FlashCore cp.async - Test Summary")
    print("=" * 80)
    print()
    
    all_passed = all(r["passed"] for r in results["shapes"].values())
    mission_perf = results["shapes"]["mission"]["p50_us"]
    mission_err = results["shapes"]["mission"]["max_err"]
    
    print(f"Mission shape results:")
    print(f"  Performance: {mission_perf:.2f} μs")
    print(f"  Error:       {mission_err:.6f}")
    print()
    
    baseline_32x32 = 279
    
    if all_passed and mission_perf < 100:
        print("✅✅✅ PHASE 2B SUCCESS! EXCEPTIONAL PERFORMANCE!")
        print()
        print("Achievements:")
        print(f"  ✓ cp.async implemented successfully")
        print(f"  ✓ Performance: 279 → {mission_perf:.0f} μs ({baseline_32x32/mission_perf:.2f}× speedup!)")
        print(f"  ✓ Error maintained: {mission_err:.3f}")
        print(f"  ✓ <100 μs achieved - excellent foundation!")
        print()
        print("Next steps:")
        print("  → Micro-tuning: launch_bounds, register cap, L2 residency")
        print("  → Target: {:.0f} → 60 μs (1.{:.0f}× more)".format(mission_perf, mission_perf/60))
        print("  → Then: 60 → <40 μs (final push!)")
        print()
        print("  🎯 <40 μs IS WITHIN REACH! 🚀")
    elif all_passed and mission_perf < 150:
        print("✅ PHASE 2B SUCCESS!")
        print()
        print("Achievements:")
        print(f"  ✓ cp.async working: 279 → {mission_perf:.0f} μs ({baseline_32x32/mission_perf:.2f}× speedup)")
        print(f"  ✓ Error maintained: {mission_err:.3f}")
        print(f"  ✓ Within expected 110-140 μs range")
        print()
        print("Next steps:")
        print("  → Micro-tuning checklist (expect 20-30% more)")
        print("  → Target: <40 μs with all optimizations")
    elif all_passed:
        print("⚠️ PARTIAL SUCCESS - cp.async works but slower than expected")
        print(f"  Current: {mission_perf:.0f} μs (expected 110-140)")
        print("  Investigate: Occupancy, bank conflicts, staging logic")
    else:
        print("❌ TESTS FAILED - Debug required")
        print("  Check: cp.async synchronization, buffer management")
    
    return results

if __name__ == "__main__":
    results = test_flashcore_cpasync()

