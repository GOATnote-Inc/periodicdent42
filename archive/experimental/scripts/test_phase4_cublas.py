#!/usr/bin/env python3
"""Test Phase 4 cuBLAS kernel for correctness and performance."""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import time
from bench.build_phase4_cublas import build_phase4_cublas

def test_phase4_cublas():
    """Test Phase 4 cuBLAS kernel."""
    
    print("=" * 60)
    print("Phase 4 cuBLAS Test (Hybrid TC Q@K^T)")
    print("=" * 60)
    
    # Build
    fa_phase4_cublas = build_phase4_cublas()
    
    # Test config
    B, H, S, D = 1, 8, 512, 64
    scale = 1.0 / (D ** 0.5)
    
    print(f"\nTest: B={B}, H={H}, S={S}, D={D}")
    
    # Generate inputs
    torch.manual_seed(42)
    q = torch.randn(B, H, S, D, dtype=torch.float16, device='cuda:0')
    k = torch.randn(B, H, S, D, dtype=torch.float16, device='cuda:0')
    v = torch.randn(B, H, S, D, dtype=torch.float16, device='cuda:0')
    
    # Phase 4 cuBLAS forward
    print("\n1) Running Phase 4 cuBLAS...")
    torch.cuda.synchronize()
    start = time.time()
    o_cublas = fa_phase4_cublas.forward(q, k, v, scale)
    torch.cuda.synchronize()
    time_cublas = (time.time() - start) * 1e6
    print(f"   Time: {time_cublas:.2f} μs")
    
    # PyTorch SDPA reference
    print("\n2) Running PyTorch SDPA...")
    torch.cuda.synchronize()
    start = time.time()
    o_torch = torch.nn.functional.scaled_dot_product_attention(
        q, k, v, attn_mask=None, dropout_p=0.0, is_causal=False, scale=scale
    )
    torch.cuda.synchronize()
    time_torch = (time.time() - start) * 1e6
    print(f"   Time: {time_torch:.2f} μs")
    
    # Correctness
    print("\n3) Correctness Check")
    max_diff = (o_cublas.float() - o_torch.float()).abs().max().item()
    mean_diff = (o_cublas.float() - o_torch.float()).abs().mean().item()
    
    passed = torch.allclose(o_cublas, o_torch, atol=1e-3, rtol=1e-3)
    
    print(f"   Max diff:  {max_diff:.6f}")
    print(f"   Mean diff: {mean_diff:.6f}")
    print(f"   Status: {'✅ PASS' if passed else '❌ FAIL'}")
    
    # Performance summary
    print("\n" + "=" * 60)
    print("PERFORMANCE SUMMARY")
    print("=" * 60)
    print(f"Phase 4 cuBLAS:  {time_cublas:8.2f} μs")
    print(f"PyTorch SDPA:    {time_torch:8.2f} μs")
    print(f"Speedup:         {time_torch/time_cublas:8.2f}×")
    print("=" * 60)
    
    if not passed:
        print("\n⚠️  Correctness test FAILED")
        return False
    
    # Compare to Phase 4 scalar baseline (839 μs)
    baseline_us = 839
    speedup_vs_scalar = baseline_us / time_cublas
    
    print(f"\nvs Phase 4 Scalar ({baseline_us} μs): {speedup_vs_scalar:.2f}×")
    
    if speedup_vs_scalar >= 1.5:
        print("✅ Significant speedup achieved!")
    elif speedup_vs_scalar >= 1.1:
        print("✅ Modest speedup achieved")
    else:
        print("⚠️  Slower than scalar baseline")
    
    return passed

if __name__ == "__main__":
    success = test_phase4_cublas()
    sys.exit(0 if success else 1)

