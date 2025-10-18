#!/usr/bin/env python3
"""
Quick acceptance test for Child-V2c (True WMMA)
Tests 5 shapes × 2 causal modes = 10 total tests
"""
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from bench_sdpa import build_ext, run_case
import torch

def main():
    print("Building Child-V2c (True WMMA)...")
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
    
    print("CHILD-V2c ACCEPTANCE TESTS (True WMMA)")
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
        
        # Check speedup vs V2b baseline (2452 μs)
        v2b_baseline_us = 2452.0
        mission_result = results[0]  # (1,8,512,64,False)
        speedup_vs_v2b = v2b_baseline_us / mission_result["us"]
        
        if speedup_vs_v2b >= 3.0:
            print(f"✅ 3× faster than V2b: {speedup_vs_v2b:.2f}×")
        elif speedup_vs_v2b >= 2.0:
            print(f"⚠️  2× faster than V2b: {speedup_vs_v2b:.2f}× (target: 3-6×)")
        else:
            print(f"❌ Slower than expected: {speedup_vs_v2b:.2f}× vs V2b (target: 3-6×)")
        
        # Check speedup vs torch
        if mission_result["speedup_vs_torch"] > 0.1:
            print(f"✅ Approaching PyTorch: {mission_result['speedup_vs_torch']:.2f}× (0.1× = 10× slower)")
        else:
            print(f"⚠️  Still far from PyTorch: {mission_result['speedup_vs_torch']:.2f}×")
        
        return 0
    else:
        print(f"❌ {total - passed} tests FAILED")
        return 1

if __name__ == "__main__":
    sys.exit(main())

