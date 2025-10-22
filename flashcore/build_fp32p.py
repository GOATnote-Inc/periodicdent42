#!/usr/bin/env python3
"""Build script for FlashCore Fused FP32 P kernel."""

import os
from torch.utils.cpp_extension import load

def build_fp32p():
    """Build the FP32 P kernel extension."""
    
    # Get the directory containing this script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    sources = [
        os.path.join(script_dir, 'kernels', 'flashcore_fused_wmma_fp32p.cu'),
        os.path.join(script_dir, 'kernels', 'flashcore_fused_fp32p_bindings.cu'),
    ]
    
    # Check that source files exist
    for src in sources:
        if not os.path.exists(src):
            raise FileNotFoundError(f"Source file not found: {src}")
    
    print("Building FlashCore Fused FP32 P kernel...")
    print(f"Sources: {sources}")
    
    extra_cuda_cflags = [
        '-O3',
        '-arch=sm_89',
        '--use_fast_math',
        '-lineinfo',
        '-Xptxas', '-v',
        '-std=c++17',
    ]
    
    print(f"CUDA flags: {extra_cuda_cflags}")
    
    ext = load(
        name='flashcore_fp32p_ext',
        sources=sources,
        extra_cuda_cflags=extra_cuda_cflags,
        verbose=True,
    )
    
    print("✅ Build successful")
    return ext

if __name__ == '__main__':
    build_fp32p()

