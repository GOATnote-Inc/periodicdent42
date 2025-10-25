#!/usr/bin/env python3
"""Test v10: 3-Stage Pipeline (better latency hiding, no warp coordination)"""

import torch
import torch.nn.functional as F
from build_wmma import build_extension

def test_v10_3stage():
    """Test v10 - 3-stage pipeline for better memory/compute overlap"""
    print("\n" + "=" * 70)
    print("FlashCore v10 - 3-Stage Pipeline")
    print("=" * 70)
    
    # Build
    module = build_extension(verbose=False)
    
    B, H, S, D = 1, 8, 512, 64
    scale = 1.0 / (D ** 0.5)
    
    # Create inputs
    torch.manual_seed(42)
    Q = torch.randn(B, H, S, D, dtype=torch.half, device='cuda')
    K = torch.randn_like(Q)
    V = torch.randn_like(Q)
    
    print(f"\nConfiguration: B={B}, H={H}, S={S}, D={D}")
    print(f"Architecture:")
    print(f"  - Tile size: 48×32 (same as v8)")
    print(f"  - Warps: 12 (384 threads)")
    print(f"  - Pipeline: 3 stages (vs 2 in v8)")
    print(f"  - SMEM: ~73 KB (+ 13.5 KB for extra stage)")
    print(f"  - No warp specialization (all warps same)")
    print(f"  - Simple __syncthreads() (no deadlocks!)")
    
    # Reference
    with torch.no_grad():
        ref = F.scaled_dot_product_attention(Q, K, V, scale=scale, is_causal=False)
    
    # Test correctness
    print("\n--- Correctness Test ---")
    with torch.no_grad():
        out = module.v10_3stage(Q, K, V, scale)
    
    max_err = (out - ref).abs().max().item()
    mean_err = (out - ref).abs().mean().item()
    
    print(f"Max error: {max_err:.6f}")
    print(f"Mean error: {mean_err:.6f}")
    
    if max_err < 0.01:
        print("✅ Correctness: PASS")
    else:
        print(f"❌ Correctness: FAIL (error {max_err:.6f} > 0.01)")
        return False
    
    # Benchmark
    print("\n--- Performance Benchmark ---")
    
    # Warmup
    for _ in range(20):
        with torch.no_grad():
            _ = module.v10_3stage(Q, K, V, scale)
    torch.cuda.synchronize()
    
    # Benchmark v10
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    
    start.record()
    for _ in range(200):
        with torch.no_grad():
            _ = module.v10_3stage(Q, K, V, scale)
    end.record()
    torch.cuda.synchronize()
    v10_time = (start.elapsed_time(end) / 200) * 1000
    
    # Benchmark v8 for comparison
    for _ in range(20):
        with torch.no_grad():
            _ = module.v8_dynamic(Q, K, V, scale)
    torch.cuda.synchronize()
    
    start.record()
    for _ in range(200):
        with torch.no_grad():
            _ = module.v8_dynamic(Q, K, V, scale)
    end.record()
    torch.cuda.synchronize()
    v8_time = (start.elapsed_time(end) / 200) * 1000
    
    # Benchmark PyTorch SDPA
    for _ in range(20):
        with torch.no_grad():
            _ = F.scaled_dot_product_attention(Q, K, V, scale=scale, is_causal=False)
    torch.cuda.synchronize()
    
    start.record()
    for _ in range(200):
        with torch.no_grad():
            _ = F.scaled_dot_product_attention(Q, K, V, scale=scale, is_causal=False)
    end.record()
    torch.cuda.synchronize()
    sdpa_time = (start.elapsed_time(end) / 200) * 1000
    
    print(f"\n[v8 Dynamic (2-stage)]   {v8_time:.2f} μs")
    print(f"[v10 3-Stage]            {v10_time:.2f} μs")
    print(f"[PyTorch SDPA]           {sdpa_time:.2f} μs")
    print(f"")
    print(f"Speedup (v10 vs v8):     {v8_time / v10_time:.2f}×")
    print(f"Gap to SDPA:             {v10_time / sdpa_time:.2f}× slower")
    print(f"")
    
    # Progress assessment
    if v10_time < 40:
        print(f"🎉🎉🎉 TARGET ACHIEVED: {v10_time:.2f} μs < 40 μs! 🎉🎉🎉")
        print("🚀 MISSION ACCOMPLISHED! Sub-40 μs achieved! 🚀")
    elif v10_time < 50:
        print(f"✅ EXCELLENT: {v10_time:.2f} μs")
        print(f"Very close to <40 μs target!")
    elif v10_time < 75:
        print(f"✅ VERY GOOD: {v10_time:.2f} μs")
        print(f"Meeting Phase 2.4 target (<75 μs)!")
    elif v10_time < 100:
        print(f"✅ GOOD: {v10_time:.2f} μs")
        print(f"Progress from v8!")
    else:
        print(f"📊 Result: {v10_time:.2f} μs")
    
    # Total progress
    baseline = 986  # Phase 1.1 baseline
    total_speedup = baseline / v10_time
    print(f"\n📈 Total Journey: {baseline} μs → {v10_time:.2f} μs")
    print(f"Total Speedup: {total_speedup:.1f}× from Phase 1.1 baseline")
    
    # Analyze speedup from v8
    if v10_time < v8_time:
        speedup = v8_time / v10_time
        print(f"\n🎯 3-Stage Pipeline Impact: {speedup:.2f}×")
        if speedup >= 1.2:
            print(f"✅ Met target (1.2-1.3×)! Pipeline delivering!")
        elif speedup >= 1.1:
            print(f"✅ Good progress ({speedup:.2f}×)")
        else:
            print(f"📊 Modest gain ({speedup:.2f}×)")
    else:
        print(f"\n⚠️  v10 slower than v8 by {v10_time / v8_time:.2f}×")
        print(f"Possible cause: 73 KB SMEM may reduce occupancy")
        print(f"(v8: 49 KB → 2 CTAs/SM, v10: 73 KB → 1 CTA/SM)")
    
    print("\n" + "=" * 70)
    print(f"v10 3-Stage Pipeline Complete: {v10_time:.2f} μs")
    if v10_time < 40:
        print("🚀 <40 μs ACHIEVED! MISSION ACCOMPLISHED! 🚀")
    elif v10_time < v8_time:
        print(f"✅ Faster than v8 by {v8_time / v10_time:.2f}×!")
    elif v10_time < sdpa_time:
        print(f"✅ Faster than PyTorch SDPA by {sdpa_time / v10_time:.2f}×!")
    else:
        print(f"📊 Continue optimization")
    print("=" * 70)
    
    # Calculate remaining gap
    if v10_time >= 40:
        remaining_speedup = v10_time / 40.0
        print(f"\n📊 Gap Analysis:")
        print(f"Current: {v10_time:.2f} μs")
        print(f"Target: 40 μs")
        print(f"Need: {remaining_speedup:.2f}× more speedup")
        
        # Estimate what's left
        if remaining_speedup < 1.5:
            print(f"✅ Achievable with micro-optimizations!")
            print(f"   - Occupancy tuning")
            print(f"   - Register optimization")
            print(f"   - Better unrolling")
        elif remaining_speedup < 2.0:
            print(f"⚠️  Will need significant optimization")
            print(f"   - Consider smaller tiles")
            print(f"   - More aggressive pipelining")
        else:
            print(f"⚠️  Large gap remaining")
            print(f"   - May need architectural changes")
    
    return True

if __name__ == "__main__":
    test_v10_3stage()

