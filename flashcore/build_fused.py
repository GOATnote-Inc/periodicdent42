#!/usr/bin/env python3
"""
Build script for FlashCore Fused WMMA kernel
"""
import torch
from torch.utils.cpp_extension import load
import os

def build_fused(extra_cflags=None):
    """Build the fused WMMA kernel
    
    Args:
        extra_cflags: List of additional compiler flags (e.g., ['-DDEBUG_QK_ONLY=1'])
    """
    kernel_dir = os.path.join(os.path.dirname(__file__), 'kernels')
    
    sources = [
        os.path.join(kernel_dir, 'flashcore_fused_wmma.cu'),
        os.path.join(kernel_dir, 'flashcore_fused_bindings.cu'),
    ]
    
    # Verify files exist
    for src in sources:
        if not os.path.exists(src):
            raise FileNotFoundError(f"Source file not found: {src}")
    
    extra_cuda_cflags = [
        '-O3',
        '-arch=sm_89',  # L4 GPU (Ada Lovelace)
        '--use_fast_math',
        '-lineinfo',
        '-Xptxas', '-v',  # Print resource usage
        '-std=c++17',
    ]
    
    # Add any extra flags (e.g., debug flags)
    if extra_cflags:
        extra_cuda_cflags.extend(extra_cflags)
    
    print("Building FlashCore Fused WMMA kernel...")
    print(f"Sources: {sources}")
    print(f"CUDA flags: {extra_cuda_cflags}")
    
    ext = load(
        name='flashcore_fused_ext',
        sources=sources,
        extra_cuda_cflags=extra_cuda_cflags,
        verbose=True,
        with_cuda=True,
    )
    
    print("✅ Build successful!")
    return ext

if __name__ == '__main__':
    ext = build_fused()
    print("\nTesting basic forward pass...")
    
    # Test on mission shape
    B, H, S, D = 1, 8, 512, 64
    Q = torch.randn(B, H, S, D, dtype=torch.float16, device='cuda')
    K = torch.randn(B, H, S, D, dtype=torch.float16, device='cuda')
    V = torch.randn(B, H, S, D, dtype=torch.float16, device='cuda')
    scale = 1.0 / (D ** 0.5)
    
    O = ext.forward(Q, K, V, scale)
    
    print(f"Output shape: {O.shape}")
    print(f"Output dtype: {O.dtype}")
    print(f"Output device: {O.device}")
    print(f"Output range: [{O.min().item():.4f}, {O.max().item():.4f}]")
    print("\n✅ Basic test passed!")

