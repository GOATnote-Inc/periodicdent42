#!/usr/bin/env python3
"""Build script for FlashCore cp.async kernel."""

import os
from torch.utils.cpp_extension import load

def build_cpasync():
    """Build the cp.async kernel extension."""
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    sources = [
        os.path.join(script_dir, 'kernels', 'flashcore_fused_wmma_cpasync.cu'),
        os.path.join(script_dir, 'kernels', 'flashcore_cpasync_bindings.cu'),
    ]
    
    for src in sources:
        if not os.path.exists(src):
            raise FileNotFoundError(f"Source file not found: {src}")
    
    print("Building FlashCore cp.async kernel...")
    print(f"Sources: {sources}")
    
    extra_cuda_cflags = [
        '-O3',
        '-arch=sm_89',
        '--use_fast_math',
        '-lineinfo',
        '-Xptxas', '-v',
        '-Xptxas', '-O3',
        '-std=c++17',
        '--expt-relaxed-constexpr',
        '--expt-extended-lambda',
    ]
    
    print(f"CUDA flags: {extra_cuda_cflags}")
    
    ext = load(
        name='flashcore_cpasync_ext',
        sources=sources,
        extra_cuda_cflags=extra_cuda_cflags,
        verbose=True,
    )
    
    print("✅ Build successful")
    return ext

if __name__ == '__main__':
    build_cpasync()

