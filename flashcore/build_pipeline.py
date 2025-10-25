#!/usr/bin/env python3
"""Build script for FlashCore pipeline intrinsics kernel."""

import os
from torch.utils.cpp_extension import load

def build_pipeline():
    """Build the pipeline kernel extension."""
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    sources = [
        os.path.join(script_dir, 'kernels', 'flashcore_fused_wmma_pipeline.cu'),
        os.path.join(script_dir, 'kernels', 'flashcore_pipeline_bindings.cu'),
    ]
    
    for src in sources:
        if not os.path.exists(src):
            raise FileNotFoundError(f"Source file not found: {src}")
    
    print("Building FlashCore pipeline kernel...")
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
    ]
    
    print(f"CUDA flags: {extra_cuda_cflags}")
    
    ext = load(
        name='flashcore_pipeline_ext',
        sources=sources,
        extra_cuda_cflags=extra_cuda_cflags,
        verbose=True,
    )
    
    print("✅ Build successful")
    return ext

if __name__ == '__main__':
    build_pipeline()

