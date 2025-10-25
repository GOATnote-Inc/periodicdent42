#!/usr/bin/env python3
"""Test Phase 2.2: 48×48 tiles + optimized cp.async"""

import torch
import torch.nn.functional as F
from build_wmma import build_extension

def test_phase2_2():
    """Test Phase 2.2 kernel - aggressive optimization for <40 μs"""
    print("\n" + "=" * 70)
    print("FlashCore Phase 2.2 - Final Sprint to <40 μs")
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
    print(f"Tile size: 40×40 (fit in 48 KB SMEM)")
    print(f"Warps: 10 (320 threads)")
    print(f"SMEM: ~45 KB (fits in static limit)")
    
    # Reference
    with torch.no_grad():
        ref = F.scaled_dot_product_attention(Q, K, V, scale=scale, is_causal=False)
    
    # Test correctness
    print("\n--- Correctness Test ---")
    with torch.no_grad():
        out = module.fused_phase2_2(Q, K, V, scale)
    
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
            _ = module.fused_phase2_2(Q, K, V, scale)
    torch.cuda.synchronize()
    
    # Benchmark Phase 2.2
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    
    start.record()
    for _ in range(200):
        with torch.no_grad():
            _ = module.fused_phase2_2(Q, K, V, scale)
    end.record()
    torch.cuda.synchronize()
    phase2_2_time = (start.elapsed_time(end) / 200) * 1000  # Convert to μs
    
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
    
    print(f"\n[Phase 2.1 (32×32)]   {phase2_1_time:.2f} μs")
    print(f"[Phase 2.2 (48×48)]   {phase2_2_time:.2f} μs")
    print(f"[PyTorch SDPA]        {sdpa_time:.2f} μs")
    print(f"")
    print(f"Speedup (2.2 vs 2.1): {phase2_1_time / phase2_2_time:.2f}×")
    print(f"Gap to SDPA:          {phase2_2_time / sdpa_time:.2f}× slower")
    print(f"")
    
    if phase2_2_time < 40:
        print(f"🎉🎉🎉 TARGET ACHIEVED: {phase2_2_time:.2f} μs < 40 μs! 🎉🎉🎉")
    elif phase2_2_time < 60:
        print(f"✅ Excellent: {phase2_2_time:.2f} μs (close to target!)")
    elif phase2_2_time < 80:
        print(f"✅ Very Good: {phase2_2_time:.2f} μs (major progress!)")
    elif phase2_2_time < 100:
        print(f"✅ Good: {phase2_2_time:.2f} μs (solid improvement!)")
    else:
        print(f"⚠️  Progress: {phase2_2_time:.2f} μs (still optimizing...)")
    
    print("\n" + "=" * 70)
    print(f"Phase 2.2 Complete: {phase2_2_time:.2f} μs")
    if phase2_2_time < 40:
        print("🚀 MISSION ACCOMPLISHED! <40 μs achieved! 🚀")
    else:
        print(f"Gap to <40 μs: {phase2_2_time / 40:.2f}× (need {40 / phase2_2_time:.2f}× more speedup)")
    print("=" * 70)
    
    return True

if __name__ == "__main__":
    test_phase2_2()

