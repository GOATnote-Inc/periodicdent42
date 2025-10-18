#!/usr/bin/env python3
"""Test FIXED Q@K^T kernel (one warp per (m,n) pair)."""

import sys
import math
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.absolute()))

import torch
from torch.utils.cpp_extension import load

def quantize_simple(tensor, scale):
    """Simple per-tensor quantization."""
    fp8_max = 448.0
    tensor_scaled = tensor / scale
    tensor_clipped = tensor_scaled.clamp(-fp8_max, fp8_max)
    tensor_uint8 = ((tensor_clipped + fp8_max) / (2 * fp8_max) * 255).round().to(torch.uint8)
    return tensor_uint8

def dequantize(u8, scale):
    """Dequantize (must match CUDA exactly!)."""
    fp8_max = 448.0
    return ((u8.float() / 255.0) * (2 * fp8_max) - fp8_max) * scale

def test_qkt_fixed():
    print("=" * 80)
    print("TEST: FIXED Q@K^T (One Warp Per Dot Product)")
    print("=" * 80)
    print()
    
    # Freeze randomness
    torch.manual_seed(0)
    
    M, N, D = 4, 8, 64
    
    Q_fp16 = torch.randn(M, D, device='cuda', dtype=torch.float16)
    K_fp16 = torch.randn(N, D, device='cuda', dtype=torch.float16)
    
    print(f"Q shape: {Q_fp16.shape}")
    print(f"K shape: {K_fp16.shape}")
    print(f"Q range: [{Q_fp16.min():.4f}, {Q_fp16.max():.4f}]")
    print(f"K range: [{K_fp16.min():.4f}, {K_fp16.max():.4f}]")
    print()
    
    # Quantize
    Q_scale = Q_fp16.abs().max().item() / 448.0
    K_scale = K_fp16.abs().max().item() / 448.0
    
    print(f"Q_scale: {Q_scale:.6f}")
    print(f"K_scale: {K_scale:.6f}")
    print()
    
    Q_fp8 = quantize_simple(Q_fp16, Q_scale)
    K_fp8 = quantize_simple(K_fp16, K_scale)
    
    # Build FIXED kernel
    print("Building FIXED kernel...")
    kernel_path = Path(__file__).parent.parent / 'cudadent42' / 'bench' / 'kernels' / 'test_fp8_qkt_fixed.cu'
    bindings_path = Path(__file__).parent.parent / 'cudadent42' / 'bench' / 'kernels' / 'test_fp8_qkt_fixed_bindings.cpp'
    
    test_qkt_fixed = load(
        name='test_fp8_qkt_fixed',
        sources=[str(kernel_path), str(bindings_path)],
        extra_cflags=['-std=c++17'],
        extra_cuda_cflags=[
            '-O3', '-use_fast_math', '-std=c++17',
            '--generate-code=arch=compute_89,code=sm_89',
        ],
        verbose=True,
    )
    print("✅ Build successful")
    print()
    
    # Test WITHOUT inv_sqrt_d first
    print("Test 1: WITHOUT inv_sqrt_d (apply_inv_sqrt_d=False)")
    print("-" * 80)
    S_cuda = test_qkt_fixed.forward(Q_fp8, K_fp8, Q_scale, K_scale, False)
    
    # Reference: dequantize then matmul
    Q_dequant = dequantize(Q_fp8, Q_scale).to(torch.float16)
    K_dequant = dequantize(K_fp8, K_scale).to(torch.float16)
    S_ref = torch.matmul(Q_dequant, K_dequant.T)
    
    diff = (S_cuda - S_ref).abs()
    max_diff = diff.max().item()
    mean_diff = diff.mean().item()
    
    print(f"Max diff: {max_diff:.6f}")
    print(f"Mean diff: {mean_diff:.6f}")
    print(f"First row CUDA: {S_cuda[0, :4].cpu().numpy()}")
    print(f"First row Ref:  {S_ref[0, :4].cpu().numpy()}")
    print()
    
    success1 = max_diff < 0.1  # Reasonable threshold for FP8
    
    if success1:
        print("✅ Test 1 PASSED!")
    else:
        print(f"❌ Test 1 FAILED (max_diff={max_diff:.6f} > 0.1)")
    print()
    
    # Test WITH inv_sqrt_d
    print("Test 2: WITH inv_sqrt_d (apply_inv_sqrt_d=True)")
    print("-" * 80)
    S_cuda_scaled = test_qkt_fixed.forward(Q_fp8, K_fp8, Q_scale, K_scale, True)
    
    # Reference with scaling
    inv_sqrt_d = 1.0 / math.sqrt(D)
    S_ref_scaled = S_ref * inv_sqrt_d
    
    diff2 = (S_cuda_scaled - S_ref_scaled).abs()
    max_diff2 = diff2.max().item()
    mean_diff2 = diff2.mean().item()
    
    print(f"Max diff: {max_diff2:.6f}")
    print(f"Mean diff: {mean_diff2:.6f}")
    print(f"inv_sqrt_d: {inv_sqrt_d:.6f}")
    print()
    
    success2 = max_diff2 < 0.015  # Tighter threshold with scaling
    
    if success2:
        print("✅ Test 2 PASSED!")
    else:
        print(f"❌ Test 2 FAILED (max_diff={max_diff2:.6f} > 0.015)")
    print()
    
    # Final verdict
    print("=" * 80)
    print("FINAL RESULTS")
    print("=" * 80)
    
    if success1 and success2:
        print("✅✅ ALL TESTS PASSED! FP8 Q@K^T is CORRECT!")
        print()
        print("Root cause fixed: One warp per (m,n) pair (not 32 rows per warp)")
        print()
        return True
    else:
        print("❌ Some tests failed")
        return False

if __name__ == '__main__':
    success = test_qkt_fixed()
    sys.exit(0 if success else 1)

