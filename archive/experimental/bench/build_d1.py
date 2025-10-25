"""
Build script for Phase D.1: Minimal Custom Kernel

Pure CUDA implementation - no PyTorch backend wrappers!
"""

import torch
from torch.utils.cpp_extension import load
from pathlib import Path
import sys

def build_d1():
    """Build Phase D.1 minimal kernel"""
    
    print("=" * 70)
    print("Building Phase D.1: Minimal Custom FlashAttention Kernel")
    print("=" * 70)
    print()
    print("This is PURE CUDA - no PyTorch backend wrappers!")
    print("Expected performance: 100-200 μs (baseline for optimization)")
    print()
    
    # Source files
    kernel_file = "csrc/kernels/fa_d1_minimal.cu"
    bindings_file = "csrc/kernels/fa_d1_bindings.cu"
    
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
        '-gencode=arch=compute_89,code=sm_89',  # L4 (Ada)
        '-DBLOCK_M=32',
        '-DBLOCK_N=64',
        '-DHEAD_DIM=64',
        '-DNUM_WARPS=8',
        '-Xptxas=-v',  # Verbose register/SMEM usage
    ]
    
    extra_cflags = ['-std=c++17']
    
    print("Compiling with flags:")
    print(f"  {' '.join(extra_cuda_cflags)}")
    print()
    
    # Build extension
    try:
        module = load(
            name='fa_d1_minimal',
            sources=[kernel_file, bindings_file],
            extra_cuda_cflags=extra_cuda_cflags,
            extra_cflags=extra_cflags,
            verbose=True,
            with_cuda=True
        )
        print()
        print("=" * 70)
        print("✅ Phase D.1 Build Successful!")
        print("=" * 70)
        print()
        print("Next: Run scripts/test_d1.py to benchmark")
        print()
        return module
    except Exception as e:
        print()
        print("=" * 70)
        print(f"❌ Build failed: {e}")
        print("=" * 70)
        sys.exit(1)

if __name__ == "__main__":
    module = build_d1()

