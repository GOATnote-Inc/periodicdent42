#!/usr/bin/env python3
"""
Build script for fa_phase3_stable kernel (numerically stable version)

Features:
- Safe exponentials (clamped to [-20, 20])
- NaN/Inf guards at all critical points
- Division by zero protection (EPSILON=1e-8)
"""

import os
import sys
from pathlib import Path
from torch.utils.cpp_extension import load

def build_phase3_stable():
    """Build Phase 3 stable kernel with compile-time parameters"""
    
    # Get parameters from environment or use defaults
    BLOCK_M = int(os.environ.get('BLOCK_M', '32'))
    HEAD_DIM = int(os.environ.get('HEAD_DIM', '64'))
    NUM_WARPS = int(os.environ.get('NUM_WARPS', '8'))
    VEC_WIDTH = int(os.environ.get('VEC_WIDTH', '4'))
    SYNC_POLICY = int(os.environ.get('SYNC_POLICY', '2'))
    
    print(f"Building Phase 3 Stable kernel:")
    print(f"  BLOCK_M:     {BLOCK_M}")
    print(f"  HEAD_DIM:    {HEAD_DIM}")
    print(f"  NUM_WARPS:   {NUM_WARPS}")
    print(f"  VEC_WIDTH:   {VEC_WIDTH}")
    print(f"  SYNC_POLICY: {SYNC_POLICY}")
    
    # Source files
    kernel_file = "cudadent42/bench/kernels/fa_phase3_stable.cu"
    bindings_file = "cudadent42/bench/kernels/fa_phase3_stable_bindings.cu"
    
    # Check if files exist
    if not Path(kernel_file).exists():
        print(f"❌ Error: {kernel_file} not found")
        sys.exit(1)
    if not Path(bindings_file).exists():
        print(f"❌ Error: {bindings_file} not found")
        sys.exit(1)
    
    # Compile flags
    extra_cuda_cflags = [
        '-O3',
        '-use_fast_math',
        '--std=c++17',
        '-gencode=arch=compute_89,code=sm_89',  # L4 Ada
        f'-DBLOCK_M={BLOCK_M}',
        f'-DHEAD_DIM={HEAD_DIM}',
        f'-DNUM_WARPS={NUM_WARPS}',
        f'-DVEC_WIDTH={VEC_WIDTH}',
        f'-DSYNC_POLICY={SYNC_POLICY}',
        '-Xptxas=-v',  # Print register usage
    ]
    
    print("\nCompiling...")
    try:
        module = load(
            name='fa_phase3_stable',
            sources=[kernel_file, bindings_file],
            extra_cuda_cflags=extra_cuda_cflags,
            verbose=True
        )
        
        print("\n✅ Build successful!")
        print(f"   Module: fa_phase3_stable")
        print(f"   Function: forward(Q, K, V, scale)")
        return module
        
    except Exception as e:
        print(f"\n❌ Build failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    build_phase3_stable()

