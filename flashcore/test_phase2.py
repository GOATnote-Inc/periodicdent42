#!/usr/bin/env python3
"""Test Phase 2.0: Dynamic SMEM 64×64 tiles kernel"""

import torch
from build_wmma import build_extension

def test_phase2_correctness():
    """Test Phase 2 kernel correctness vs PyTorch SDPA"""
    print("\n=== Phase 2.0 Correctness Test (64×64 Dynamic SMEM) ===")
    
    # Build extension
    module = build_extension(verbose=False)
    
    # Test configuration
    B, H, S, D = 1, 8, 512, 64
    scale = 1.0 / (D ** 0.5)
    
    # Create inputs
    torch.manual_seed(42)
    Q = torch.randn(B, H, S, D, dtype=torch.half, device='cuda')
    K = torch.randn(B, H, S, D, dtype=torch.half, device='cuda')
    V = torch.randn(B, H, S, D, dtype=torch.half, device='cuda')
    
    print(f"Shape: Q/K/V [{B}, {H}, {S}, {D}]")
    
    # Reference: PyTorch SDPA
    with torch.no_grad():
        ref = torch.nn.functional.scaled_dot_product_attention(
            Q, K, V, scale=scale, is_causal=False)
    
    # Test: Phase 2 kernel
    with torch.no_grad():
        out = module.fused_phase2(Q, K, V, scale)
    
    # Compare
    max_err = (out - ref).abs().max().item()
    mean_err = (out - ref).abs().mean().item()
    
    print(f"Output shape: {list(out.shape)}")
    print(f"Max error: {max_err:.6f}")
    print(f"Mean error: {mean_err:.6f}")
    
    if max_err < 0.01:
        print("✅ Phase 2: PASS")
        return True
    else:
        print(f"❌ Phase 2: FAIL (error {max_err:.6f} > 0.01)")
        return False

def benchmark_phase2():
    """Benchmark Phase 2 vs Phase 1.3 and PyTorch SDPA"""
    print("\n=== Phase 2.0 Performance Benchmark ===")
    
    module = build_extension(verbose=False)
    
    B, H, S, D = 1, 8, 512, 64
    scale = 1.0 / (D ** 0.5)
    
    Q = torch.randn(B, H, S, D, dtype=torch.half, device='cuda')
    K = torch.randn(B, H, S, D, dtype=torch.half, device='cuda')
    V = torch.randn(B, H, S, D, dtype=torch.half, device='cuda')
    
    # Warmup
    for _ in range(10):
        with torch.no_grad():
            _ = module.fused_phase2(Q, K, V, scale)
    torch.cuda.synchronize()
    
    # Benchmark Phase 2
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    
    start.record()
    for _ in range(100):
        with torch.no_grad():
            _ = module.fused_phase2(Q, K, V, scale)
    end.record()
    torch.cuda.synchronize()
    
    phase2_time = start.elapsed_time(end) / 100
    
    # Benchmark Phase 1.3 (32×32)
    for _ in range(10):
        with torch.no_grad():
            _ = module.fused(Q, K, V, scale)
    torch.cuda.synchronize()
    
    start.record()
    for _ in range(100):
        with torch.no_grad():
            _ = module.fused(Q, K, V, scale)
    end.record()
    torch.cuda.synchronize()
    
    phase1_time = start.elapsed_time(end) / 100
    
    # Benchmark PyTorch SDPA
    for _ in range(10):
        with torch.no_grad():
            _ = torch.nn.functional.scaled_dot_product_attention(
                Q, K, V, scale=scale, is_causal=False)
    torch.cuda.synchronize()
    
    start.record()
    for _ in range(100):
        with torch.no_grad():
            _ = torch.nn.functional.scaled_dot_product_attention(
                Q, K, V, scale=scale, is_causal=False)
    end.record()
    torch.cuda.synchronize()
    
    sdpa_time = start.elapsed_time(end) / 100
    
    print(f"Configuration: B={B}, H={H}, S={S}, D={D}")
    print(f"")
    print(f"[Phase 1.3 (32×32)] {phase1_time*1000:.2f} µs")
    print(f"[Phase 2.0 (64×64)] {phase2_time*1000:.2f} µs")
    print(f"[PyTorch SDPA]     {sdpa_time*1000:.2f} µs")
    print(f"")
    print(f"Speedup (Phase 2.0 vs Phase 1.3): {phase1_time/phase2_time:.2f}×")
    print(f"Gap to SDPA: {phase2_time/sdpa_time:.2f}× slower")
    print(f"")
    
    if phase2_time < 80:
        print(f"✅ Phase 2.0 target (<80 µs): ACHIEVED!")
    else:
        print(f"⚠️ Phase 2.0 target (<80 µs): {phase2_time*1000:.2f} µs")
    
    return phase2_time

if __name__ == "__main__":
    print("=" * 60)
    print("FlashCore Phase 2.0 Test Suite")
    print("=" * 60)
    
    # Test correctness
    correct = test_phase2_correctness()
    
    if correct:
        # Benchmark performance
        phase2_latency = benchmark_phase2()
        
        print("\n" + "=" * 60)
        print("✅ Phase 2.0 Tests Complete")
        print("=" * 60)
        print(f"Final Latency: {phase2_latency*1000:.2f} µs")
        print(f"Target: <80 µs (Phase 2.0)")
        print(f"Ultimate Target: <40 µs (Phase 2.3)")
    else:
        print("\n" + "=" * 60)
        print("❌ Phase 2.0 correctness test failed!")
        print("=" * 60)

