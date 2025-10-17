#!/usr/bin/env python3
"""
Build Phase D tuned kernel with register pressure optimization.

Environment variables:
  REGCAP: Register cap (-maxrregcount=N)
  LB_THREADS: Launch bounds threads per block
  LB_MIN: Launch bounds min blocks per SM
  TILE_M: Tile size M dimension
"""

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.absolute()))

from torch.utils.cpp_extension import load

def build_phaseD():
    """Build Phase D tuned kernel."""
    
    # Get tuning parameters
    regcap = os.environ.get('REGCAP', '')
    lb_threads = os.environ.get('LB_THREADS', '192')
    lb_min = os.environ.get('LB_MIN', '2')
    tile_m = os.environ.get('TILE_M', '32')
    
    print("=" * 80)
    print("BUILDING PHASE D: Register Pressure Optimized Kernel")
    print("=" * 80)
    print()
    print(f"Configuration:")
    print(f"  REGCAP:     {regcap or 'none (compiler decides)'}")
    print(f"  LB_THREADS: {lb_threads}")
    print(f"  LB_MIN:     {lb_min}")
    print(f"  TILE_M:     {tile_m}")
    print()
    
    # Build flags
    extra_cuda_cflags = [
        '-O3',
        '-use_fast_math',
        '-lineinfo',
        '-Xptxas', '-v',  # Verbose: show register usage
        '-std=c++17',
        '--generate-code=arch=compute_89,code=sm_89',
    ]
    
    # Add register cap if specified
    if regcap:
        extra_cuda_cflags.extend(['-maxrregcount', regcap])
        print(f"✅ Register cap: {regcap}")
    
    # Add launch bounds as defines
    extra_cflags = [
        f'-DLAUNCH_BOUNDS_THREADS={lb_threads}',
        f'-DLAUNCH_BOUNDS_MIN={lb_min}',
        f'-DTILE_M={tile_m}',
        f'-DTHREADS_PER_BLOCK={lb_threads}',
        f'-DMIN_BLOCKS_PER_SM={lb_min}',
    ]
    
    print(f"✅ Launch bounds: __launch_bounds__({lb_threads}, {lb_min})")
    print(f"✅ Tile size: {tile_m}")
    print()
    
    # Kernel source
    kernel_path = Path(__file__).parent.parent / 'cudadent42' / 'bench' / 'kernels' / 'fa_phaseD_tuned.cu'
    bindings_path = Path(__file__).parent.parent / 'cudadent42' / 'bench' / 'kernels' / 'fa_phaseD_tuned_bindings.cpp'
    
    if not kernel_path.exists():
        print(f"❌ Kernel not found: {kernel_path}")
        sys.exit(1)
    
    print(f"Compiling {kernel_path.name}...")
    print()
    
    try:
        module = load(
            name='fa_phaseD_tuned',
            sources=[str(kernel_path), str(bindings_path)],
            extra_cflags=extra_cflags,
            extra_cuda_cflags=extra_cuda_cflags,
            verbose=True,
        )
        
        print()
        print("=" * 80)
        print("✅ BUILD SUCCESSFUL")
        print("=" * 80)
        print()
        print("Module: fa_phaseD_tuned")
        print(f"  Config: REGCAP={regcap or 'none'}, THREADS={lb_threads}, MIN_BLOCKS={lb_min}, TILE_M={tile_m}")
        print()
        print("Next: Run benchmark with python bench/run_phaseD.py")
        print()
        
        return module
        
    except Exception as e:
        print()
        print("=" * 80)
        print("❌ BUILD FAILED")
        print("=" * 80)
        print()
        print(f"Error: {e}")
        print()
        print("Troubleshooting:")
        print("  1. Check CUDA version: nvcc --version")
        print("  2. Check PyTorch CUDA: python -c 'import torch; print(torch.version.cuda)'")
        print("  3. Try without REGCAP: unset REGCAP && python bench/build_phaseD.py")
        print()
        sys.exit(1)

if __name__ == '__main__':
    build_phaseD()

