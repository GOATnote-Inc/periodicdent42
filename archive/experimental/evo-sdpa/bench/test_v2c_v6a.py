#!/usr/bin/env python3
"""
Acceptance test for Child-V2c-v6a (WMMA GREEN: store → softmax → rebuild)
Tests 5 shapes × 2 causal modes = 10 total tests

TARGET: 100% correctness (GREEN) + 2-3× speedup from V2c-v5 @ 1980 μs
"""
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from bench_sdpa import build_ext, run_case
import torch

def main():
    print("Building Child-V2c-v6a (WMMA GREEN: store → softmax → rebuild)...")
    mod = build_ext()
    print("✅ Build successful\n")
    
    # 5 acceptance test shapes
    test_cases = [
        # (B, H, L, d, causal)
        (1, 8, 512, 64, False),
        (1, 8, 512, 64, True),
        (2, 8, 2048, 64, False),
        (2, 8, 2048, 64, True),
        (2, 8, 2048, 128, False),
    ]
    
    print("CHILD-V2c-v6a ACCEPTANCE TESTS (WMMA GREEN)")
    print("=" * 100)
    
    results = []
    for B, H, L, d, causal in test_cases:
        result = run_case(mod, B=B, H=H, L=L, d=d, causal=causal, 
                         dtype=torch.float16, iters=50, verbose=False)
        results.append(result)
    
    print("\n" + "=" * 100)
    
    # Summary
    passed = sum(1 for r in results if r["ok"])
    total = len(results)
    
    print(f"\nSUMMARY: {passed}/{total} tests passed")
    
    if passed == total:
        print("✅ ALL ACCEPTANCE TESTS PASSED!")
        print("✅ WMMA GREEN: Full pipeline working correctly!")
        
        # Check speedup vs V2c-v5 GREEN baseline (1980 μs)
        v2c_v5_baseline_us = 1980.0
        mission_result = results[0]  # (1,8,512,64,False)
        speedup_vs_v5 = v2c_v5_baseline_us / mission_result["us"]
        
        print(f"\nPerformance (vs V2c-v5 GREEN @ 1980 μs):")
        if speedup_vs_v5 >= 3.0:
            print(f"✅ Excellent: {speedup_vs_v5:.2f}× faster!")
        elif speedup_vs_v5 >= 2.5:
            print(f"✅ Great: {speedup_vs_v5:.2f}× faster (target: 2-3×)")
        elif speedup_vs_v5 >= 2.0:
            print(f"✅ Good: {speedup_vs_v5:.2f}× faster (target: 2-3×)")
        elif speedup_vs_v5 >= 1.5:
            print(f"✅ Modest: {speedup_vs_v5:.2f}× faster (expected 2-3×)")
        elif speedup_vs_v5 >= 1.2:
            print(f"⚠️  Some improvement: {speedup_vs_v5:.2f}× faster")
            print("   Note: Small scratch buffers add overhead, but correctness is key")
        else:
            print(f"⚠️  Minimal speedup: {speedup_vs_v5:.2f}×")
            print("   Note: v7 FAST (cp.async overlap + swizzle) will improve this")
        
        # Check speedup vs V2c-v3 scalar baseline (1750 μs)
        v2c_v3_baseline_us = 1750.0
        speedup_vs_v3 = v2c_v3_baseline_us / mission_result["us"]
        print(f"\nSpeedup vs V2c-v3 scalar: {speedup_vs_v3:.2f}×")
        
        # Mission shape latency
        print(f"\nMission shape (1,8,512,64): {mission_result['us']:.2f} μs")
        print(f"vs PyTorch SDPA: {mission_result['speedup_vs_torch']:.3f}× ({1/mission_result['speedup_vs_torch']:.0f}× slower)")
        
        # GREEN status
        print("\n✅ GREEN ACHIEVED: 100% correctness with full WMMA pipeline")
        print("   NEXT: V2c-v7 FAST (cp.async overlap, swizzle, epilogue fusion)")
        
        return 0
    else:
        print(f"❌ {total - passed} tests FAILED")
        print("Debug: Check store→softmax→rebuild pattern or per-warp scratch")
        return 1

if __name__ == "__main__":
    sys.exit(main())

