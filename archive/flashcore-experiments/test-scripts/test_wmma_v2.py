#!/usr/bin/env python3
"""Test FlashCore WMMA kernel"""

import torch
import sys
sys.path.insert(0, '.')
from build_wmma_v2 import build_wmma

def test_wmma():
    print("=" * 70)
    print("FlashCore WMMA Tensor Core Test")
    print("=" * 70)
    
    # Build
    print("\n[1/5] Building WMMA kernel...")
    fc_wmma = build_wmma()
    
    # Test inputs
    B, H, S, D = 1, 8, 512, 64
    print(f"\n[2/5] Creating inputs (B={B}, H={H}, S={S}, D={D})...")
    Q = torch.randn(B, H, S, D, dtype=torch.float16, device='cuda')
    K = torch.randn(B, H, S, D, dtype=torch.float16, device='cuda')
    V = torch.randn(B, H, S, D, dtype=torch.float16, device='cuda')
    scale = 1.0 / (D ** 0.5)
    
    # Warmup
    print("\n[3/5] Warmup (20 iters)...")
    for _ in range(20):
        O_wmma = fc_wmma.forward(Q, K, V, scale)
    torch.cuda.synchronize()
    
    # Benchmark
    print("\n[4/5] Benchmarking (100 iters)...")
    times = []
    for _ in range(100):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        
        start.record()
        O_wmma = fc_wmma.forward(Q, K, V, scale)
        end.record()
        
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end) * 1000)  # μs
    
    import statistics
    p50 = statistics.median(times)
    p90 = statistics.quantiles(times, n=10)[8]
    
    # Correctness
    print("\n[5/5] Correctness check...")
    O_ref = torch.nn.functional.scaled_dot_product_attention(Q, K, V)
    max_err = (O_wmma - O_ref).abs().max().item()
    mean_err = (O_wmma - O_ref).abs().mean().item()
    
    print(f"\n{'='*70}")
    print("RESULTS")
    print(f"{'='*70}")
    print(f"Correctness:")
    print(f"  Max error:  {max_err:.6f}")
    print(f"  Mean error: {mean_err:.6f}")
    print(f"  Status:     {'✅ PASS' if max_err < 0.40 else '❌ FAIL'}")
    
    print(f"\nPerformance:")
    print(f"  p50:  {p50:.1f} μs")
    print(f"  p90:  {p90:.1f} μs")
    
    # Compare to baseline
    baseline_us = 1397.0
    speedup = baseline_us / p50
    print(f"\nSpeedup vs Baseline:")
    print(f"  Baseline: {baseline_us:.1f} μs")
    print(f"  WMMA:     {p50:.1f} μs")
    print(f"  Speedup:  {speedup:.1f}×")
    
    # Target assessment
    target_low, target_high = 64, 128
    print(f"\nTarget Assessment:")
    if target_low <= p50 <= target_high:
        print(f"  ✅ SUCCESS! Within target range ({target_low}-{target_high} μs)")
    elif p50 < target_low:
        print(f"  🎉 OUTSTANDING! Faster than expected ({target_low} μs)")
    else:
        print(f"  ⚠️  SLOWER than target (expected {target_low}-{target_high} μs)")
    
    # Next steps
    print(f"\n{'='*70}")
    if max_err < 0.40 and target_low <= p50 <= target_high:
        print("✅ WMMA PHASE COMPLETE!")
        print("\nNext: Warp-level sync (reduce barrier stalls)")
        print("      Expected: 1.47× → {}→{} μs".format(int(p50), int(p50/1.47)))
    elif max_err >= 0.40:
        print("❌ Correctness issue - debug WMMA patterns")
    else:
        print("⚠️  Performance slower than expected - profile with NCU")
    
    return max_err < 0.40

if __name__ == '__main__':
    success = test_wmma()
    sys.exit(0 if success else 1)

