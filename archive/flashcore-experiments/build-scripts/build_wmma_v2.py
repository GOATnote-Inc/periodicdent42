#!/usr/bin/env python3
"""Build FlashCore WMMA kernel"""

import os
from torch.utils.cpp_extension import load

def build_wmma():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    sources = [
        os.path.join(script_dir, 'kernels', 'flashcore_wmma_v2.cu'),
        os.path.join(script_dir, 'kernels', 'flashcore_wmma_v2_bindings.cu'),
    ]
    
    extra_cuda_cflags = [
        '-O3',
        '-arch=sm_89',
        '--use_fast_math',
        '-lineinfo',
        '-Xptxas', '-v',
        '-std=c++17',
    ]
    
    print("=" * 70)
    print("FlashCore WMMA Tensor Core Kernel")
    print("=" * 70)
    print("Goal: 10-20× speedup over baseline (1397 → 64-128 μs)")
    print("Features:")
    print("  - WMMA for Q@K^T (16×16×16 tiles)")
    print("  - WMMA for P@V (16×16×16 tiles)")
    print("  - FP32 accumulation")
    print("  - Online softmax")
    print("=" * 70)
    
    return load(
        name='flashcore_wmma_v2',
        sources=sources,
        extra_cuda_cflags=extra_cuda_cflags,
        verbose=True
    )

if __name__ == '__main__':
    build_wmma()
    print("\n✅ WMMA kernel built successfully!")

