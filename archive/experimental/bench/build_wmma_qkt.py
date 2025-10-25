"""
Build script for Phase C.1: WMMA Q@K^T kernel
"""

import torch
from torch.utils.cpp_extension import load
from pathlib import Path
import os
import sys

def build_wmma_qkt():
    """Build WMMA Q@K^T kernel with PyTorch C++ extension"""
    
    # Configuration
    BLOCK_M = int(os.environ.get('BLOCK_M', '32'))
    HEAD_DIM = int(os.environ.get('HEAD_DIM', '64'))
    BLOCK_N = int(os.environ.get('BLOCK_N', '64'))
    NUM_WARPS = int(os.environ.get('NUM_WARPS', '8'))
    
    print("Building Phase C.1: WMMA Q@K^T kernel:")
    print(f"  BLOCK_M:     {BLOCK_M}")
    print(f"  HEAD_DIM:    {HEAD_DIM}")
    print(f"  BLOCK_N:     {BLOCK_N}")
    print(f"  NUM_WARPS:   {NUM_WARPS}")
    print()
    
    # Source files
    kernel_file = "cudadent42/bench/kernels/fa_wmma_qkt.cu"
    bindings_file = "cudadent42/bench/kernels/fa_wmma_qkt_bindings.cu"
    
    # Check files exist
    if not Path(kernel_file).exists():
        print(f"❌ Error: {kernel_file} not found")
        sys.exit(1)
    if not Path(bindings_file).exists():
        print(f"❌ Error: {bindings_file} not found")
        sys.exit(1)
    
    # Compilation flags
    extra_cuda_cflags = [
        '-O3',
        '-use_fast_math',
        '--std=c++17',
        f'-gencode=arch=compute_89,code=sm_89',
        f'-DBLOCK_M={BLOCK_M}',
        f'-DHEAD_DIM={HEAD_DIM}',
        f'-DBLOCK_N={BLOCK_N}',
        f'-DNUM_WARPS={NUM_WARPS}',
        '-Xptxas=-v',  # Verbose PTX assembly
    ]
    
    extra_cflags = ['-std=c++17']
    
    print("Compiling...")
    print(f"Flags: {' '.join(extra_cuda_cflags)}")
    print()
    
    # Build extension
    try:
        module = load(
            name='fa_wmma_qkt',
            sources=[kernel_file, bindings_file],
            extra_cuda_cflags=extra_cuda_cflags,
            extra_cflags=extra_cflags,
            verbose=True,
            with_cuda=True
        )
        print("✅ Build successful!")
        return module
    except Exception as e:
        print(f"❌ Build failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    module = build_wmma_qkt()
    print(f"\n✅ Module built: {module}")

