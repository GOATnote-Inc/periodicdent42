#!/usr/bin/env python3
"""
Quick acceptance test for Child-V2b
Tests 5 shapes × 2 causal modes = 10 total tests
"""
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from bench_sdpa import build_ext, run_case
import torch

def main():
    print("Building Child-V2b...")
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
    
    print("CHILD-V2b ACCEPTANCE TESTS")
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
        
        # Check speedup vs torch on mission-critical shape
        mission_result = results[0]  # (1,8,512,64,False)
        if mission_result["speedup_vs_torch"] > 1.0:
            print(f"✅ Faster than PyTorch SDPA: {mission_result['speedup_vs_torch']:.2f}×")
        else:
            print(f"⚠️  Slower than PyTorch SDPA: {mission_result['speedup_vs_torch']:.2f}×")
        
        # Check speedup vs baseline (assuming baseline ~1400 μs)
        baseline_us = 1400.0
        if mission_result["us"] < baseline_us / 10:
            print(f"✅ > 10× faster than scalar baseline: {baseline_us / mission_result['us']:.1f}×")
        else:
            print(f"⚠️  Not 10× faster than baseline: {baseline_us / mission_result['us']:.1f}×")
        
        return 0
    else:
        print(f"❌ {total - passed} tests FAILED")
        return 1

if __name__ == "__main__":
    sys.exit(main())


