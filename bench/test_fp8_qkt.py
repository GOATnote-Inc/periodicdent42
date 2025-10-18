#!/usr/bin/env python3
"""Test minimal Q@K^T kernel to isolate FP8 bug."""

import sys
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

def test_qkt():
    print("=" * 80)
    print("TEST: Minimal Q@K^T with FP8")
    print("=" * 80)
    print()
    
    # Very small test case
    M, N, D = 4, 8, 64
    
    torch.manual_seed(42)
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
    
    print(f"Q_fp8 range: [{Q_fp8.min()}, {Q_fp8.max()}]")
    print(f"K_fp8 range: [{K_fp8.min()}, {K_fp8.max()}]")
    print()
    
    # Build kernel
    print("Building test kernel...")
    kernel_path = Path(__file__).parent.parent / 'cudadent42' / 'bench' / 'kernels' / 'test_fp8_qkt.cu'
    bindings_path = Path(__file__).parent.parent / 'cudadent42' / 'bench' / 'kernels' / 'test_fp8_qkt_bindings.cpp'
    
    test_qkt_module = load(
        name='test_fp8_qkt',
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
    
    # Run kernel
    print("Running kernel...")
    S_cuda = test_qkt_module.forward(Q_fp8, K_fp8, Q_scale, K_scale)
    
    print(f"S_cuda shape: {S_cuda.shape}")
    print(f"S_cuda range: [{S_cuda.min():.4f}, {S_cuda.max():.4f}]")
    print()
    
    # Reference: Dequantize first, then matmul (apples to apples!)
    fp8_max = 448.0
    Q_dequant = ((Q_fp8.float() / 255.0 * (2 * fp8_max) - fp8_max) * Q_scale).to(torch.float16)
    K_dequant = ((K_fp8.float() / 255.0 * (2 * fp8_max) - fp8_max) * K_scale).to(torch.float16)
    
    S_ref = torch.matmul(Q_dequant, K_dequant.T)
    print(f"S_ref (dequantized) range: [{S_ref.min():.4f}, {S_ref.max():.4f}]")
    print()
    
    # Also compute with original FP16 for comparison
    S_ref_fp16 = torch.matmul(Q_fp16, K_fp16.T)
    print(f"S_ref (original FP16) range: [{S_ref_fp16.min():.4f}, {S_ref_fp16.max():.4f}]")
    print()
    
    # Compare
    diff = (S_cuda - S_ref).abs()
    max_diff = diff.max().item()
    mean_diff = diff.mean().item()
    
    print("=" * 80)
    print("RESULTS")
    print("=" * 80)
    print(f"Max diff: {max_diff:.6f}")
    print(f"Mean diff: {mean_diff:.6f}")
    print()
    
    # Show individual values for debugging
    print("First row comparison (Q[0] · K):")
    print("CUDA:", S_cuda[0, :8].cpu().numpy())
    print("Ref: ", S_ref[0, :8].cpu().numpy())
    print("Diff:", diff[0, :8].cpu().numpy())
    print()
    
    # Expected error from quantization
    expected_error = (Q_scale + K_scale) * D * 0.01  # Rough estimate
    print(f"Expected error: ~{expected_error:.4f}")
    
    if max_diff < 1.0:  # Reasonable threshold for FP8
        print("✅ PASS: Q@K^T computation is correct!")
        return True
    else:
        print("❌ FAIL: Q@K^T computation is broken!")
        
        # Find worst case
        max_idx = diff.argmax()
        m, n = max_idx // N, max_idx % N
        print()
        print(f"Worst case at [{m}, {n}]:")
        print(f"  CUDA: {S_cuda[m, n].item():.6f}")
        print(f"  Ref:  {S_ref[m, n].item():.6f}")
        print(f"  Diff: {diff[m, n].item():.6f}")
        return False

if __name__ == '__main__':
    success = test_qkt()
    sys.exit(0 if success else 1)

