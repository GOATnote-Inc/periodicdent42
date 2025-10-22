#!/usr/bin/env python3
"""
Test script for FlashCore Fused WMMA kernel
Uses the test_framework.py for systematic evaluation
"""
import torch
import statistics
import sys

def test_flashcore_fused():
    """Comprehensive test for fused kernel"""
    print("="*80)
    print("FlashCore Fused WMMA Kernel - Comprehensive Test")
    print("="*80)
    print()
    
    # Build kernel
    print("Building kernel...")
    from build_fused import build_fused
    ext = build_fused()
    print("✅ Build successful\n")
    
    # Test shapes
    shapes = [
        (1, 8, 512, 64, "mission"),   # Primary target
        (1, 8, 256, 64, "short"),     # Generalization
        (1, 8, 1024, 64, "long"),     # Generalization
    ]
    
    results = {}
    target_us = 40  # Stretch goal
    prev_us = 634    # Multi-query baseline
    
    for B, H, S, D, name in shapes:
        print(f"Testing shape: {name} (B={B}, H={H}, S={S}, D={D})")
        
        # ========================================
        # Correctness Test
        # ========================================
        Q = torch.randn(B, H, S, D, dtype=torch.float16, device='cuda')
        K = torch.randn(B, H, S, D, dtype=torch.float16, device='cuda')
        V = torch.randn(B, H, S, D, dtype=torch.float16, device='cuda')
        scale = 1.0 / (D ** 0.5)
        
        # Reference (PyTorch SDPA)
        with torch.no_grad():
            O_ref = torch.nn.functional.scaled_dot_product_attention(
                Q, K, V, scale=scale
            )
        
        # Our kernel
        with torch.no_grad():
            O_kernel = ext.forward(Q, K, V, scale)
        
        # Compute errors
        max_err = (O_ref - O_kernel).abs().max().item()
        mean_err = (O_ref - O_kernel).abs().mean().item()
        
        print(f"  Correctness:")
        print(f"    max_err:  {max_err:.6f}")
        print(f"    mean_err: {mean_err:.6f}")
        
        if max_err > 0.06:
            print(f"  ❌ FAIL: max_err exceeds threshold (0.06)\n")
            results[name] = {"pass": False, "max_err": max_err}
            continue
        
        print(f"  ✅ PASS (correctness)\n")
        
        # ========================================
        # Performance Test
        # ========================================
        warmup, iters = 20, 100
        
        # Warmup
        for _ in range(warmup):
            _ = ext.forward(Q, K, V, scale)
        torch.cuda.synchronize()
        
        # Timed runs
        times = []
        for _ in range(iters):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            
            start.record()
            O = ext.forward(Q, K, V, scale)
            end.record()
            
            torch.cuda.synchronize()
            times.append(start.elapsed_time(end) * 1000)  # Convert ms to μs
        
        p50 = statistics.median(times)
        p90 = statistics.quantiles(times, n=10)[8]
        speedup_vs_prev = prev_us / p50
        
        print(f"  Performance:")
        print(f"    p50: {p50:.2f} μs")
        print(f"    p90: {p90:.2f} μs")
        print(f"    Speedup vs baseline (634 μs): {speedup_vs_prev:.2f}×")
        print(f"    Target: {target_us:.2f} μs")
        
        if p50 <= target_us:
            print(f"  🎉 TARGET MET! ({p50:.2f} μs ≤ {target_us:.2f} μs)\n")
        elif p50 <= target_us * 1.2:
            print(f"  ✅ ACCEPTABLE (within 20% of target)\n")
        else:
            print(f"  ⚠️  MISSED TARGET (off by {p50/target_us:.2f}×)\n")
        
        results[name] = {
            "pass": True,
            "max_err": max_err,
            "mean_err": mean_err,
            "p50": p50,
            "p90": p90,
            "speedup": speedup_vs_prev,
            "target_met": p50 <= target_us * 1.2
        }
    
    # ========================================
    # Summary
    # ========================================
    print("="*80)
    print("FlashCore Fused WMMA - Test Summary")
    print("="*80)
    print()
    
    all_correct = all(r["pass"] for r in results.values())
    any_target_met = any(r.get("target_met", False) for r in results.values())
    
    if all_correct and any_target_met:
        print("✅ ALL TESTS PASSED - TARGET MET!")
    elif all_correct:
        print("✅ ALL TESTS PASSED (correctness)")
        print("⚠️  Performance target not met yet (continue optimizing)")
    else:
        print("⚠️  SOME TESTS FAILED - DEBUG REQUIRED")
        print("\nFailed shapes:")
        for name, r in results.items():
            if not r["pass"]:
                print(f"  - {name}: max_err={r['max_err']:.6f}")
    
    print("\nDetailed Results:")
    for name, r in results.items():
        if r["pass"]:
            print(f"  {name:10s}: {r['p50']:6.2f} μs ({r['speedup']:.2f}× speedup, max_err={r['max_err']:.6f})")
        else:
            print(f"  {name:10s}: FAILED (max_err={r['max_err']:.6f})")
    
    return results

if __name__ == '__main__':
    results = test_flashcore_fused()
    
    # Exit with appropriate code
    all_pass = all(r["pass"] for r in results.values())
    sys.exit(0 if all_pass else 1)

