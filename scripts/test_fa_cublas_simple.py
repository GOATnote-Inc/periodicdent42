#!/usr/bin/env python3
"""Test simple cuBLAS attention."""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import time
from bench.build_fa_cublas_simple import build_fa_cublas_simple

def test_fa_cublas_simple():
    print("=" * 60)
    print("cuBLAS TensorCore Attention Test")
    print("=" * 60)
    
    # Build
    fa_cublas = build_fa_cublas_simple()
    
    # Test
    B, H, S, D = 1, 8, 512, 64
    scale = 1.0 / (D ** 0.5)
    
    print(f"\nTest: B={B}, H={H}, S={S}, D={D}")
    
    torch.manual_seed(42)
    q = torch.randn(B, H, S, D, dtype=torch.float16, device='cuda:0')
    k = torch.randn(B, H, S, D, dtype=torch.float16, device='cuda:0')
    v = torch.randn(B, H, S, D, dtype=torch.float16, device='cuda:0')
    
    # cuBLAS
    print("\n1) cuBLAS TensorCore...")
    torch.cuda.synchronize()
    start = time.time()
    o_cublas = fa_cublas.forward(q, k, v, scale)
    torch.cuda.synchronize()
    time_cublas = (time.time() - start) * 1e6
    print(f"   Time: {time_cublas:.2f} μs")
    
    # PyTorch SDPA
    print("\n2) PyTorch SDPA...")
    torch.cuda.synchronize()
    start = time.time()
    o_torch = torch.nn.functional.scaled_dot_product_attention(
        q, k, v, attn_mask=None, dropout_p=0.0, is_causal=False, scale=scale
    )
    torch.cuda.synchronize()
    time_torch = (time.time() - start) * 1e6
    print(f"   Time: {time_torch:.2f} μs")
    
    # Correctness
    print("\n3) Correctness")
    max_diff = (o_cublas.float() - o_torch.float()).abs().max().item()
    mean_diff = (o_cublas.float() - o_torch.float()).abs().mean().item()
    passed = torch.allclose(o_cublas, o_torch, atol=1e-3, rtol=1e-3)
    
    print(f"   Max diff:  {max_diff:.6f}")
    print(f"   Mean diff: {mean_diff:.6f}")
    print(f"   Status: {'✅ PASS' if passed else '❌ FAIL'}")
    
    # Summary
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"cuBLAS TC:       {time_cublas:8.2f} μs")
    print(f"PyTorch SDPA:    {time_torch:8.2f} μs")
    print(f"Phase 4 Scalar:  {839:8.2f} μs (baseline)")
    print("=" * 60)
    print(f"Speedup vs SDPA:   {time_torch/time_cublas:.2f}×")
    print(f"Speedup vs Phase4: {839/time_cublas:.2f}×")
    print("=" * 60)
    
    return passed

if __name__ == "__main__":
    success = test_fa_cublas_simple()
    sys.exit(0 if success else 1)

