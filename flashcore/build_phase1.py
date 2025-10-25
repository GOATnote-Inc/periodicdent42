#!/usr/bin/env python3
"""Build script for FlashCore Phase 1 (Proven WMMA pattern)"""

import os
from torch.utils.cpp_extension import load

def build_phase1():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    sources = [
        os.path.join(script_dir, 'kernels', 'flashcore_phase1_proven_wmma.cu'),
        os.path.join(script_dir, 'kernels', 'flashcore_phase1_bindings.cu'),
    ]
    
    extra_cuda_cflags = [
        '-O3',
        '-arch=sm_89',  # L4 GPU
        '--use_fast_math',
        '-lineinfo',
        '-std=c++17',
        '--expt-relaxed-constexpr',
        '-Xptxas', '-v',  # Verbose register/SMEM usage
        '-Xptxas', '--warn-spills',
    ]
    
    print("=" * 60)
    print("FlashCore Phase 1: Proven WMMA Pattern")
    print("=" * 60)
    print("Goal: 279 → 180-220 μs (1.3-1.5× speedup)")
    print("Changes:")
    print("  - K stored as [N][D] (col-major WMMA)")
    print("  - Exact pattern from sdpa_fp8_stage_c_wmma.cu")
    print("=" * 60)
    
    return load(
        name='flashcore_phase1',
        sources=sources,
        extra_cuda_cflags=extra_cuda_cflags,
        verbose=True
    )

if __name__ == '__main__':
    build_phase1()
    print("\n✅ Phase 1 kernel built successfully!")
    print("Next: python test_phase1.py")

