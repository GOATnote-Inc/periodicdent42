#!/usr/bin/env python3
"""Test v9: Warp Specialization (12 compute + 4 producer warps)"""

import torch
import torch.nn.functional as F
from build_wmma import build_extension

def test_v9_warp_spec():
    """Test v9 - Warp specialization for producer-consumer overlap"""
    print("\n" + "=" * 70)
    print("FlashCore v9 - Warp Specialization")
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
    print(f"  - Tile size: 48×32 (asymmetric, from v8)")
    print(f"  - Warps: 16 total (512 threads)")
    print(f"    • Compute warps: 12 (warps 0-11)")
    print(f"    • Producer warps: 4 (warps 12-15)")
    print(f"  - SMEM: ~59 KB (dynamic allocation)")
    print(f"  - Pipeline: Producer-consumer overlap")
    print(f"  - Synchronization: Flag-based coordination")
    
    # Reference
    with torch.no_grad():
        ref = F.scaled_dot_product_attention(Q, K, V, scale=scale, is_causal=False)
    
    # Test correctness
    print("\n--- Correctness Test ---")
    try:
        with torch.no_grad():
            out = module.v9_warp_spec(Q, K, V, scale)
        
        max_err = (out - ref).abs().max().item()
        mean_err = (out - ref).abs().mean().item()
        
        print(f"Max error: {max_err:.6f}")
        print(f"Mean error: {mean_err:.6f}")
        
        if max_err < 0.01:
            print("✅ Correctness: PASS")
        else:
            print(f"❌ Correctness: FAIL (error {max_err:.6f} > 0.01)")
            print("\n⚠️  v9 needs debugging - continuing with v8 for now")
            return False
    except Exception as e:
        print(f"❌ Runtime Error: {e}")
        print("\n⚠️  v9 needs debugging - this is expected for first iteration")
        return False
    
    # Benchmark
    print("\n--- Performance Benchmark ---")
    
    # Warmup
    for _ in range(20):
        with torch.no_grad():
            _ = module.v9_warp_spec(Q, K, V, scale)
    torch.cuda.synchronize()
    
    # Benchmark v9
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    
    start.record()
    for _ in range(200):
        with torch.no_grad():
            _ = module.v9_warp_spec(Q, K, V, scale)
    end.record()
    torch.cuda.synchronize()
    v9_time = (start.elapsed_time(end) / 200) * 1000
    
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
    
    print(f"\n[v8 Dynamic (48×32)]     {v8_time:.2f} μs")
    print(f"[v9 Warp Spec (48×32)]   {v9_time:.2f} μs")
    print(f"[PyTorch SDPA]           {sdpa_time:.2f} μs")
    print(f"")
    print(f"Speedup (v9 vs v8):      {v8_time / v9_time:.2f}×")
    print(f"Gap to SDPA:             {v9_time / sdpa_time:.2f}× slower")
    print(f"")
    
    # Progress assessment
    if v9_time < 40:
        print(f"🎉🎉🎉 TARGET ACHIEVED: {v9_time:.2f} μs < 40 μs! 🎉🎉🎉")
        print("🚀 MISSION ACCOMPLISHED! Sub-40 μs achieved! 🚀")
    elif v9_time < 60:
        print(f"✅ EXCELLENT: {v9_time:.2f} μs")
        print(f"Gap to <40 μs: {(v9_time / 40):.2f}×")
    elif v9_time < 75:
        print(f"✅ VERY GOOD: {v9_time:.2f} μs (warp spec working!)")
        print(f"Gap to <40 μs: {(v9_time / 40):.2f}×")
    elif v9_time < 100:
        print(f"✅ GOOD: {v9_time:.2f} μs")
        print(f"Gap to <40 μs: {(v9_time / 40):.2f}×")
    else:
        print(f"📊 Progress: {v9_time:.2f} μs")
        if v9_time > v8_time:
            print(f"⚠️  v9 slower than v8 - warp spec needs tuning")
    
    # Total progress
    baseline = 986  # Phase 1.1 baseline
    total_speedup = baseline / v9_time
    print(f"\n📈 Total Journey: {baseline} μs → {v9_time:.2f} μs")
    print(f"Total Speedup: {total_speedup:.1f}× from Phase 1.1 baseline")
    
    # Analyze speedup from v8
    if v9_time < v8_time:
        speedup = v8_time / v9_time
        print(f"\n🎯 Warp Specialization Impact: {speedup:.2f}×")
        if speedup >= 1.3:
            print(f"✅ Met target (1.3-1.4×)! Warp spec delivering!")
        elif speedup >= 1.15:
            print(f"✅ Good progress ({speedup:.2f}×), close to 1.3× target")
        else:
            print(f"📊 Modest gain ({speedup:.2f}×), may need tuning")
    else:
        print(f"\n⚠️  v9 slower than v8 by {v9_time / v8_time:.2f}×")
        print("Possible causes:")
        print("  - Producer-consumer coordination overhead")
        print("  - Insufficient work for 16 warps")
        print("  - SMEM bank conflicts")
        print("  - Need better synchronization")
    
    print("\n" + "=" * 70)
    print(f"v9 Warp Specialization Complete: {v9_time:.2f} μs")
    if v9_time < 40:
        print("🚀 <40 μs ACHIEVED! MISSION ACCOMPLISHED! 🚀")
    elif v9_time < v8_time:
        print(f"✅ Faster than v8 by {v8_time / v9_time:.2f}×!")
        if v9_time < 75:
            print(f"✅ Target for Phase 2.3 (<75 μs) achieved!")
    else:
        print(f"📊 Debugging needed - v9 slower than v8")
    print("=" * 70)
    
    return True

if __name__ == "__main__":
    test_v9_warp_spec()

