#!/usr/bin/env python3
"""
Build FlashAttention-3 style kernel for PyTorch
Target: <40 μs on NVIDIA L4 (sm_89)
"""

import os
import sys
from pathlib import Path
from torch.utils.cpp_extension import load

flashcore_dir = Path(__file__).parent

print("=" * 70)
print("Building FlashCore FA-3 Kernel")
print("=" * 70)
print(f"FlashCore dir: {flashcore_dir}")
print()

# Source files
sources = [
    str(flashcore_dir / "kernels" / "flashcore_fa3_kernel.cu"),
    str(flashcore_dir / "kernels" / "flashcore_fa3_bindings.cu"),
]

# CUDA flags
cuda_flags = [
    '-O3',
    '-arch=sm_89',       # L4 (Ada)
    '-std=c++17',
    '--use_fast_math',
    '-Xptxas', '-v',
    '--maxrregcount=128',  # Limit registers
    '-lineinfo',
    # Tile configuration
    '-DM_TILE=64',
    '-DN_TILE=128',
    '-DWARPS_PER_BLOCK=4',
]

print("Building with:")
print(f"  Sources: {len(sources)} files")
print(f"  CUDA flags: sm_89, O3, maxreg=128")
print(f"  Tiles: M=64, N=128, Warps=4")
print()

try:
    flashcore_fa3 = load(
        name='flashcore_fa3',
        sources=sources,
        extra_cuda_cflags=cuda_flags,
        verbose=True,
        with_cuda=True,
    )
    
    print()
    print("=" * 70)
    print("✅ BUILD SUCCESSFUL!")
    print("=" * 70)
    print(f"Module: {flashcore_fa3}")
    print()
    print("Available functions:")
    print(f"  - flashcore_fa3.forward(Q, K, V) -> O")
    print()
    
except Exception as e:
    print()
    print("=" * 70)
    print("❌ BUILD FAILED!")
    print("=" * 70)
    print(f"Error: {e}")
    print()
    sys.exit(1)

