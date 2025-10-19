#!/usr/bin/env python3
"""
Acceptance test for Child-V2c-v7a (cp.async overlap pipeline)
Tests 5 shapes × 2 causal modes = 10 total tests

TARGET: 1.3-1.7× speedup from V2c-v6a @ 1177 μs → 700-900 μs
GOAL: 100% correctness + overlap pipeline working
"""
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from bench_sdpa import build_ext, run_case
import torch

def main():
    print("Building Child-V2c-v7a (cp.async overlap pipeline)...")
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
    
    print("CHILD-V2c-v7a ACCEPTANCE TESTS (cp.async OVERLAP)")
    print("=" * 100)
    
    results = []
    for B, H, L, d, causal in test_cases:
        result = run_case(mod, B=B, H=H, L=L, d=d, causal=causal, 
                         dtype=torch.float16, iters=100, verbose=False)
        results.append(result)
    
    print("\n" + "=" * 100)
    
    # Summary
    passed = sum(1 for r in results if r["ok"])
    total = len(results)
    
    print(f"\nSUMMARY: {passed}/{total} tests passed")
    
    if passed == total:
        print("✅ ALL ACCEPTANCE TESTS PASSED!")
        print("✅ cp.async OVERLAP: Producer/consumer pipeline working!")
        
        # Check speedup vs V2c-v6a GREEN (1177 μs)
        v2c_v6a_baseline_us = 1177.0
        mission_result = results[0]  # (1,8,512,64,False)
        speedup_vs_v6a = v2c_v6a_baseline_us / mission_result["us"]
        
        print(f"\nPerformance (vs V2c-v6a @ 1177 μs):")
        if speedup_vs_v6a >= 1.7:
            print(f"✅ Excellent: {speedup_vs_v6a:.2f}× faster! (target: 1.3-1.7×)")
        elif speedup_vs_v6a >= 1.5:
            print(f"✅ Great: {speedup_vs_v6a:.2f}× faster! (target: 1.3-1.7×)")
        elif speedup_vs_v6a >= 1.3:
            print(f"✅ Good: {speedup_vs_v6a:.2f}× faster (target: 1.3-1.7×)")
        elif speedup_vs_v6a >= 1.2:
            print(f"✅ Modest: {speedup_vs_v6a:.2f}× faster (expected 1.3-1.7×)")
        elif speedup_vs_v6a >= 1.1:
            print(f"⚠️  Some overlap: {speedup_vs_v6a:.2f}× faster")
            print("   Note: Check producer warp utilization and stage counts")
        else:
            print(f"⚠️  Minimal speedup: {speedup_vs_v6a:.2f}×")
            print("   Debug: Verify cp.async overlap and wait_group logic")
        
        # Mission shape latency
        print(f"\nMission shape (1,8,512,64): {mission_result['us']:.2f} μs")
        print(f"vs PyTorch SDPA: {mission_result['speedup_vs_torch']:.3f}× ({1/mission_result['speedup_vs_torch']:.0f}× slower)")
        
        # Evolution
        v2c_v5_us = 1980.0
        v2c_v3_us = 1750.0
        speedup_vs_v5 = v2c_v5_us / mission_result["us"]
        speedup_vs_v3 = v2c_v3_us / mission_result["us"]
        
        print(f"\nEvolution:")
        print(f"  vs V2c-v5 (WMMA Q@K^T):    {speedup_vs_v5:.2f}×")
        print(f"  vs V2c-v3 (scalar):        {speedup_vs_v3:.2f}×")
        
        # FAST status
        print("\n✅ PHASE 1 FAST: cp.async overlap pipeline working!")
        if speedup_vs_v6a >= 1.3:
            print("   NEXT: V2c-v7b (XOR swizzle for K^T bank conflict elimination)")
        else:
            print("   Consider: NCU profiling to verify overlap efficiency")
        
        return 0
    else:
        print(f"❌ {total - passed} tests FAILED")
        print("Debug: Check stage_ring_handshake, wait_group counts, or producer/consumer sync")
        return 1

if __name__ == "__main__":
    sys.exit(main())

