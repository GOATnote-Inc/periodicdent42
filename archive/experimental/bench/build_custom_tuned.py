#!/usr/bin/env python3
"""
Build custom kernel with occupancy tuning.

Environment variables:
  REGCAP: Register cap (-maxrregcount=N)
  LB_THREADS: Launch bounds threads per block
  LB_MIN: Launch bounds min blocks per SM
  TILE_M: Tile size M dimension
"""

import os
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.absolute()))

from torch.utils.cpp_extension import load

def build_custom_tuned():
    """Build custom attention kernel with tuning parameters."""
    
    # Get tuning parameters from environment
    regcap = os.environ.get('REGCAP', '')
    lb_threads = os.environ.get('LB_THREADS', '')
    lb_min = os.environ.get('LB_MIN', '2')
    tile_m = os.environ.get('TILE_M', '64')
    
    print(f"Building with:")
    print(f"  REGCAP: {regcap or 'none'}")
    print(f"  LB_THREADS: {lb_threads or 'none'}")
    print(f"  LB_MIN: {lb_min}")
    print(f"  TILE_M: {tile_m}")
    print()
    
    # Build flags
    extra_cuda_cflags = [
        '-O3',
        '-use_fast_math',
        '-lineinfo',
        '-Xptxas', '-v',
        '-std=c++17',
        '--generate-code=arch=compute_89,code=sm_89',
    ]
    
    # Add register cap if specified
    if regcap:
        extra_cuda_cflags.extend(['-maxrregcount', regcap])
    
    # Add launch bounds as defines
    extra_cflags = []
    if lb_threads:
        extra_cflags.append(f'-DLAUNCH_BOUNDS_THREADS={lb_threads}')
        extra_cflags.append(f'-DLAUNCH_BOUNDS_MIN={lb_min}')
    
    # Add tile size
    extra_cflags.append(f'-DTILE_M={tile_m}')
    
    # Kernel source (placeholder - would be actual optimized kernel)
    kernel_path = Path(__file__).parent.parent / 'cudadent42' / 'bench' / 'kernels' / 'fa_phase4.cu'
    bindings_path = Path(__file__).parent.parent / 'cudadent42' / 'bench' / 'kernels' / 'fa_phase4_bindings.cpp'
    
    if not kernel_path.exists():
        print(f"❌ Kernel not found: {kernel_path}")
        print("   Using Phase 4 as baseline for tuning")
        sys.exit(1)
    
    print(f"Compiling {kernel_path}...")
    
    try:
        module = load(
            name='fa_custom_tuned',
            sources=[str(kernel_path), str(bindings_path)],
            extra_cflags=extra_cflags,
            extra_cuda_cflags=extra_cuda_cflags,
            verbose=True,
        )
        
        print("✅ Build successful!")
        return module
        
    except Exception as e:
        print(f"❌ Build failed: {e}")
        sys.exit(1)

if __name__ == '__main__':
    build_custom_tuned()

