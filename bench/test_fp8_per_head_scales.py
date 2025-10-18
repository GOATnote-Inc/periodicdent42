#!/usr/bin/env python3
"""
Test Q@K^T with PER-HEAD scales (matches full SDPA).
Goal: Isolate if scale handling is the bug.
"""

import sys
import math
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.absolute()))

import torch
from torch.utils.cpp_extension import load

def quantize_per_head(tensor, per_head=True):
    """
    Quantize with per-head scales (matches full SDPA).
    
    Args:
        tensor: [B, H, S, D] or [H, S, D]
    
    Returns:
        fp8_data: uint8 tensor
        scales: [H] per-head scales
    """
    fp8_max = 448.0
    
    if len(tensor.shape) == 4:
        B, H, S, D = tensor.shape
        # Per-head max
        abs_max = tensor.abs().view(B, H, -1).max(dim=2)[0]  # [B, H]
        scales = abs_max / fp8_max
        scales = scales.clamp(min=1e-12)
        
        # Expand for broadcasting
        scales_expanded = scales.view(B, H, 1, 1)
        
        # Quantize
        tensor_scaled = tensor / scales_expanded
        tensor_clipped = tensor_scaled.clamp(-fp8_max, fp8_max)
        tensor_uint8 = ((tensor_clipped + fp8_max) / (2 * fp8_max) * 255).round().to(torch.uint8)
        
        # Return per-head scales [H] (assuming B=1 for simplicity)
        return tensor_uint8, scales[0, :].to(dtype=torch.float32, device='cuda')
    
    elif len(tensor.shape) == 3:
        H, S, D = tensor.shape
        # Per-head max
        abs_max = tensor.abs().view(H, -1).max(dim=1)[0]  # [H]
        scales = abs_max / fp8_max
        scales = scales.clamp(min=1e-12)
        
        # Expand for broadcasting
        scales_expanded = scales.view(H, 1, 1)
        
        # Quantize
        tensor_scaled = tensor / scales_expanded
        tensor_clipped = tensor_scaled.clamp(-fp8_max, fp8_max)
        tensor_uint8 = ((tensor_clipped + fp8_max) / (2 * fp8_max) * 255).round().to(torch.uint8)
        
        return tensor_uint8, scales.to(dtype=torch.float32, device='cuda')
    else:
        raise ValueError(f"Unexpected shape: {tensor.shape}")

def dequantize_per_head(fp8_data, scales):
    """Dequantize with per-head scales."""
    fp8_max = 448.0
    
    if len(fp8_data.shape) == 4:
        B, H, S, D = fp8_data.shape
        scales_expanded = scales.view(1, H, 1, 1)
    elif len(fp8_data.shape) == 3:
        H, S, D = fp8_data.shape
        scales_expanded = scales.view(H, 1, 1)
    else:
        raise ValueError(f"Unexpected shape: {fp8_data.shape}")
    
    tensor_float = ((fp8_data.float() / 255.0) * (2 * fp8_max) - fp8_max) * scales_expanded
    return tensor_float.to(torch.float16)

def test_per_head_scales():
    print("=" * 80)
    print("TEST: Q@K^T with PER-HEAD SCALES")
    print("=" * 80)
    print()
    
    torch.manual_seed(0)
    
    # Test with multiple heads (like full SDPA)
    B, H, M, N, D = 1, 8, 4, 8, 64
    
    # Generate per-head data (each head has different scale)
    Q_fp16 = torch.randn(B, H, M, D, device='cuda', dtype=torch.float16)
    K_fp16 = torch.randn(B, H, N, D, device='cuda', dtype=torch.float16)
    
    print(f"Q shape: {Q_fp16.shape} (B, H, M, D)")
    print(f"K shape: {K_fp16.shape} (B, H, N, D)")
    print(f"Q range: [{Q_fp16.min():.4f}, {Q_fp16.max():.4f}]")
    print(f"K range: [{K_fp16.min():.4f}, {K_fp16.max():.4f}]")
    print()
    
    # Quantize with per-head scales
    Q_fp8, Q_scales = quantize_per_head(Q_fp16, per_head=True)
    K_fp8, K_scales = quantize_per_head(K_fp16, per_head=True)
    
    print(f"Q_scales: {Q_scales.cpu().numpy()}")
    print(f"K_scales: {K_scales.cpu().numpy()}")
    print()
    
    # Build kernel (reuse fixed kernel)
    print("Building kernel...")
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
        verbose=False,
    )
    print("✅ Build successful")
    print()
    
    # Test each head separately (kernel expects 2D input)
    print("Testing per-head Q@K^T...")
    print("-" * 80)
    
    all_correct = True
    max_diff_overall = 0.0
    
    for h in range(H):
        # Extract head h
        Q_h = Q_fp8[0, h, :, :]  # [M, D]
        K_h = K_fp8[0, h, :, :]  # [N, D]
        
        Q_scale_h = Q_scales[h].item()
        K_scale_h = K_scales[h].item()
        
        # Run kernel (per-tensor API, but with per-head scale)
        S_cuda = test_qkt_fixed.forward(Q_h, K_h, Q_scale_h, K_scale_h, False)
        
        # Reference
        Q_h_dequant = dequantize_per_head(Q_fp8[0:1, h:h+1, :, :], Q_scales[h:h+1])
        K_h_dequant = dequantize_per_head(K_fp8[0:1, h:h+1, :, :], K_scales[h:h+1])
        
        S_ref = torch.matmul(Q_h_dequant[0, 0, :, :], K_h_dequant[0, 0, :, :].T)
        
        # Compare
        diff = (S_cuda - S_ref).abs()
        max_diff_h = diff.max().item()
        max_diff_overall = max(max_diff_overall, max_diff_h)
        
        correct_h = max_diff_h < 0.1
        
        status = "✅" if correct_h else "❌"
        print(f"  Head {h}: max_diff={max_diff_h:.6f} {status}")
        
        if not correct_h:
            all_correct = False
            print(f"    Q_scale: {Q_scale_h:.6f}, K_scale: {K_scale_h:.6f}")
            print(f"    CUDA[0,:4]: {S_cuda[0, :4].cpu().numpy()}")
            print(f"    Ref[0,:4]:  {S_ref[0, :4].cpu().numpy()}")
    
    print()
    print("=" * 80)
    print("RESULTS")
    print("=" * 80)
    print(f"Max diff (all heads): {max_diff_overall:.6f}")
    print(f"All correct: {all_correct}")
    print()
    
    if all_correct:
        print("✅✅ PER-HEAD SCALES WORK!")
        print()
        print("Conclusion: Scale handling is CORRECT")
        print("→ Bug must be in full SDPA (softmax/P@V/output)")
        return True
    else:
        print("❌ PER-HEAD SCALES BROKEN!")
        print()
        print("Conclusion: Scale handling needs fix")
        print("→ Apply same fix to full SDPA")
        return False

if __name__ == '__main__':
    success = test_per_head_scales()
    sys.exit(0 if success else 1)

