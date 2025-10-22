#!/usr/bin/env python3
"""Build script for FlashCore 64x64 kernel."""

import os
from torch.utils.cpp_extension import load

def build_64x64():
    """Build the 64x64 kernel extension."""
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    sources = [
        os.path.join(script_dir, 'kernels', 'flashcore_fused_wmma_64x64.cu'),
        os.path.join(script_dir, 'kernels', 'flashcore_fused_64x64_bindings.cu'),
    ]
    
    for src in sources:
        if not os.path.exists(src):
            raise FileNotFoundError(f"Source file not found: {src}")
    
    print("Building FlashCore 64x64 kernel...")
    print(f"Sources: {sources}")
    
    extra_cuda_cflags = [
        '-O3',
        '-arch=sm_89',
        '--use_fast_math',
        '-lineinfo',
        '-Xptxas', '-v',
        '-std=c++17',
        '-maxrregcount=110',  # Limit registers for better occupancy
    ]
    
    print(f"CUDA flags: {extra_cuda_cflags}")
    
    ext = load(
        name='flashcore_64x64_ext',
        sources=sources,
        extra_cuda_cflags=extra_cuda_cflags,
        verbose=True,
    )
    
    print("✅ Build successful")
    return ext

if __name__ == '__main__':
    build_64x64()

