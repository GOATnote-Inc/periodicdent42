#!/usr/bin/env python3
"""Test FlashCore Phase 1 kernel"""

import torch
import time
from build_phase1 import build_phase1

def test_phase1():
    print("=" * 60)
    print("FlashCore Phase 1 Test")
    print("=" * 60)
    
    # Build kernel
    print("\n[1/4] Building Phase 1 kernel...")
    flashcore_phase1 = build_phase1()
    
    # Mission shape
    B, H, S, D = 1, 8, 512, 64
    device = 'cuda'
    
    print(f"\n[2/4] Creating test inputs (B={B}, H={H}, S={S}, D={D})...")
    Q = torch.randn(B, H, S, D, dtype=torch.float16, device=device)
    K = torch.randn(B, H, S, D, dtype=torch.float16, device=device)
    V = torch.randn(B, H, S, D, dtype=torch.float16, device=device)
    
    softmax_scale = 1.0 / (D ** 0.5)
    
    # Warmup
    print("\n[3/4] Warmup (20 iters)...")
    for _ in range(20):
        O_ours = flashcore_phase1.forward(Q, K, V, softmax_scale)
    torch.cuda.synchronize()
    
    # Benchmark
    print("\n[4/4] Benchmarking (100 iters)...")
    times = []
    for _ in range(100):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        
        start.record()
        O_ours = flashcore_phase1.forward(Q, K, V, softmax_scale)
        end.record()
        
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end) * 1000)  # Convert to μs
    
    # Statistics
    import statistics
    p50 = statistics.median(times)
    p90 = statistics.quantiles(times, n=10)[8]
    mean = statistics.mean(times)
    std = statistics.stdev(times)
    
    # Correctness (compare to PyTorch)
    print("\n[Correctness Check]")
    O_ref = torch.nn.functional.scaled_dot_product_attention(Q, K, V)
    max_err = (O_ours - O_ref).abs().max().item()
    mean_err = (O_ours - O_ref).abs().mean().item()
    
    print(f"  Max error:  {max_err:.6f}")
    print(f"  Mean error: {mean_err:.6f}")
    
    # Performance
    print("\n[Performance Results]")
    print(f"  p50:  {p50:.1f} μs")
    print(f"  p90:  {p90:.1f} μs")
    print(f"  mean: {mean:.1f} μs ± {std:.1f}")
    
    # Success criteria
    print("\n[Phase 1 Success Criteria]")
    target_low, target_high = 180, 220
    correctness_ok = max_err < 0.40
    perf_ok = target_low <= p50 <= target_high
    
    print(f"  ✅ Correctness: {'PASS' if correctness_ok else 'FAIL'} (error < 0.40)")
    print(f"  {'✅' if perf_ok else '⚠️ '} Performance: {p50:.1f} μs (target: {target_low}-{target_high} μs)")
    
    # Speedup vs baseline
    baseline_us = 279
    speedup = baseline_us / p50
    print(f"\n[Speedup vs Baseline]")
    print(f"  Baseline: {baseline_us} μs")
    print(f"  Phase 1:  {p50:.1f} μs")
    print(f"  Speedup:  {speedup:.2f}×")
    
    # Next steps
    if correctness_ok and perf_ok:
        print("\n🎉 Phase 1 SUCCESS! Proceed to Phase 2 (fused online softmax)")
    elif correctness_ok:
        print("\n⚠️  Phase 1 PARTIAL: Correct but slower than expected")
        print("   → Profile with NCU to find bottleneck")
    else:
        print("\n❌ Phase 1 FAILED: Correctness issue")
        print("   → Debug WMMA load/store patterns")
    
    return correctness_ok and perf_ok

if __name__ == '__main__':
    success = test_phase1()
    exit(0 if success else 1)

