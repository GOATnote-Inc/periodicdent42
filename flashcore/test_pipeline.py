#!/usr/bin/env python3
"""Test script for FlashCore pipeline intrinsics kernel."""

import torch
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from build_pipeline import build_pipeline

def test_flashcore_pipeline():
    """Test the pipeline intrinsics kernel."""
    
    print("=" * 80)
    print("FlashCore Pipeline Intrinsics (64×32 tiles) - Test")
    print("=" * 80)
    print()
    
    # Build kernel
    print("Building kernel...")
    ext = build_pipeline()
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
    
    results = {"name": "FlashCore pipeline", "shapes": {}}
    
    for name, B, H, S, D in test_shapes:
        print(f"Testing shape: {name} (B={B}, H={H}, S={S}, D={D})")
        
        torch.manual_seed(42)
        Q = torch.randn(B, H, S, D, device=device, dtype=dtype)
        K = torch.randn(B, H, S, D, device=device, dtype=dtype)
        V = torch.randn(B, H, S, D, device=device, dtype=dtype)
        
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
        
        times = []
        for _ in range(100):
            start.record()
            ext.forward(Q, K, V, O_kernel)
            end.record()
            torch.cuda.synchronize()
            times.append(start.elapsed_time(end) * 1000)  # μs
        
        times_sorted = sorted(times)
        p50 = times_sorted[50]
        p90 = times_sorted[90]
        p99 = times_sorted[99]
        
        print(f"  Performance:")
        print(f"    p50: {p50:.2f} μs")
        print(f"    p90: {p90:.2f} μs")
        print(f"    p99: {p99:.2f} μs")
        
        threshold = 0.40
        passed = max_err < threshold
        print(f"  {'✅ PASS' if passed else '❌ FAIL'}: max_err {'<' if passed else '>='} {threshold}")
        
        if name == "mission":
            speedup_vs_baseline = 1398 / p50
            speedup_vs_32x32 = 279 / p50
            print(f"  Analysis:")
            print(f"    Speedup vs baseline (1398 μs): {speedup_vs_baseline:.2f}×")
            print(f"    Speedup vs 32×32 (279 μs):     {speedup_vs_32x32:.2f}×")
            
            if p50 < 100:
                print(f"    🎯 🎯 🎯 EXCELLENT! Pipeline intrinsics working!")
            elif p50 < 150:
                print(f"    ✅ GOOD! Expected 110-140 μs range")
            else:
                print(f"    ⚠️  Slower than expected, but still progress")
        
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
    print("FlashCore Pipeline - Test Summary")
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
        print("✅✅✅ PHASE 2B SUCCESS! PIPELINE INTRINSICS ROCK!")
        print()
        print("Achievements:")
        print(f"  ✓ __pipeline_memcpy_async implemented successfully")
        print(f"  ✓ Performance: 279 → {mission_perf:.0f} μs ({baseline_32x32/mission_perf:.2f}× speedup!)")
        print(f"  ✓ Error maintained: {mission_err:.3f}")
        print(f"  ✓ <100 μs achieved!")
        print()
        print("Next steps:")
        print("  → Remove sS buffer (register softmax)")
        print("  → 64×64 tiles")
        print("  → Persistent CTAs")
        print("  → Target: <40 μs ✅")
    elif all_passed and mission_perf < 150:
        print("✅ PHASE 2B SUCCESS!")
        print()
        print(f"  ✓ Pipeline intrinsics: 279 → {mission_perf:.0f} μs ({baseline_32x32/mission_perf:.2f}× speedup)")
        print(f"  ✓ Error: {mission_err:.3f}")
        print("  ✓ Within expected 110-140 μs range")
        print()
        print("Next steps: Micro-tuning for <40 μs")
    else:
        print(f"{'⚠️ ' if all_passed else '❌'} Status: {'Slower' if all_passed else 'Failed'}")
        print(f"  Current: {mission_perf:.0f} μs, Error: {mission_err:.3f}")
    
    return results

if __name__ == "__main__":
    results = test_flashcore_pipeline()

