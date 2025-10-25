#!/usr/bin/env python3
"""Test v8: Proper dynamic SMEM with 48×32 asymmetric tiles"""

import torch
import torch.nn.functional as F
from build_wmma import build_extension

def test_v8_dynamic():
    """Test v8 - Expert-architected dynamic SMEM kernel"""
    print("\n" + "=" * 70)
    print("FlashCore v8 - Dynamic SMEM (Expert Architecture)")
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
    print(f"  - Tile size: 48×32 (asymmetric, SMEM-optimal)")
    print(f"  - Warps: 12 (384 threads)")
    print(f"  - SMEM: ~49 KB (dynamic allocation)")
    print(f"  - Padding: N+16 for WMMA safety")
    print(f"  - Pipeline: Double-buffered cp.async")
    
    # Reference
    with torch.no_grad():
        ref = F.scaled_dot_product_attention(Q, K, V, scale=scale, is_causal=False)
    
    # Test correctness
    print("\n--- Correctness Test ---")
    with torch.no_grad():
        out = module.v8_dynamic(Q, K, V, scale)
    
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
            _ = module.v8_dynamic(Q, K, V, scale)
    torch.cuda.synchronize()
    
    # Benchmark v8
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    
    start.record()
    for _ in range(200):
        with torch.no_grad():
            _ = module.v8_dynamic(Q, K, V, scale)
    end.record()
    torch.cuda.synchronize()
    v8_time = (start.elapsed_time(end) / 200) * 1000
    
    # Benchmark Phase 2.1 for comparison
    for _ in range(20):
        with torch.no_grad():
            _ = module.fused(Q, K, V, scale)
    torch.cuda.synchronize()
    
    start.record()
    for _ in range(200):
        with torch.no_grad():
            _ = module.fused(Q, K, V, scale)
    end.record()
    torch.cuda.synchronize()
    phase2_1_time = (start.elapsed_time(end) / 200) * 1000
    
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
    
    print(f"\n[Phase 2.1 (32×32)]      {phase2_1_time:.2f} μs")
    print(f"[v8 Dynamic (48×32)]     {v8_time:.2f} μs")
    print(f"[PyTorch SDPA]           {sdpa_time:.2f} μs")
    print(f"")
    print(f"Speedup (v8 vs 2.1):     {phase2_1_time / v8_time:.2f}×")
    print(f"Gap to SDPA:             {v8_time / sdpa_time:.2f}× slower")
    print(f"")
    
    # Progress assessment
    if v8_time < 40:
        print(f"🎉🎉🎉 TARGET ACHIEVED: {v8_time:.2f} μs < 40 μs! 🎉🎉🎉")
        print("🚀 MISSION ACCOMPLISHED! Sub-40 μs achieved! 🚀")
    elif v8_time < 60:
        print(f"✅ EXCELLENT: {v8_time:.2f} μs")
        print(f"Gap to <40 μs: {(v8_time / 40):.2f}× (very close!)")
    elif v8_time < 80:
        print(f"✅ VERY GOOD: {v8_time:.2f} μs")
        print(f"Gap to <40 μs: {(v8_time / 40):.2f}×")
    elif v8_time < 100:
        print(f"✅ GOOD: {v8_time:.2f} μs")
        print(f"Gap to <40 μs: {(v8_time / 40):.2f}×")
    else:
        print(f"📊 Progress: {v8_time:.2f} μs")
        print(f"Gap to <40 μs: {(v8_time / 40):.2f}×")
    
    # Total progress
    baseline = 986  # Phase 1.1 baseline
    total_speedup = baseline / v8_time
    print(f"\n📈 Total Journey: {baseline} μs → {v8_time:.2f} μs")
    print(f"Total Speedup: {total_speedup:.1f}× from Phase 1.1 baseline")
    
    print("\n" + "=" * 70)
    print(f"v8 Dynamic SMEM Complete: {v8_time:.2f} μs")
    if v8_time < 40:
        print("🚀 <40 μs ACHIEVED! MISSION ACCOMPLISHED! 🚀")
    elif v8_time < sdpa_time:
        print(f"✅ Faster than PyTorch SDPA by {sdpa_time / v8_time:.2f}×!")
    else:
        print(f"Path forward: Need {(v8_time / 40):.2f}× more for <40 μs")
    print("=" * 70)
    
    return True

if __name__ == "__main__":
    test_v8_dynamic()

