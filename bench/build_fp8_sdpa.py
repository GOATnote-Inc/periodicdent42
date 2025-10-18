#!/usr/bin/env python3
"""
Build FP8 SDPA kernel for L4 (sm_89).

Environment variables:
  CYCLE: Which cycle to build (1-6, default 1)
"""

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.absolute()))

from torch.utils.cpp_extension import load

def build_fp8_sdpa():
    """Build FP8 SDPA kernel."""
    
    cycle = os.environ.get('CYCLE', '1')
    
    print("=" * 80)
    print(f"BUILDING FP8 SDPA: Cycle {cycle}")
    print("=" * 80)
    print()
    print(f"Target: L4 (sm_89, Ada)")
    print(f"Precision: FP8 E4M3 (inputs), FP32 (compute), FP16 (output)")
    print()
    
    # Build flags for sm_89
    extra_cuda_cflags = [
        '-O3',
        '-use_fast_math',
        '-lineinfo',
        '-Xptxas', '-v',
        '-std=c++17',
        '--generate-code=arch=compute_89,code=sm_89',
        '-DCUDA_ARCH=89',
    ]
    
    extra_cflags = ['-std=c++17']
    
    # Kernel source (baseline for Cycle 1)
    kernel_path = Path(__file__).parent.parent / 'cudadent42' / 'bench' / 'kernels' / 'sdpa_fp8_baseline.cu'
    bindings_path = Path(__file__).parent.parent / 'cudadent42' / 'bench' / 'kernels' / 'sdpa_fp8_baseline_bindings.cpp'
    
    if not kernel_path.exists():
        print(f"❌ Kernel not found: {kernel_path}")
        sys.exit(1)
    
    print(f"Compiling {kernel_path.name}...")
    print()
    
    try:
        module = load(
            name='sdpa_fp8_baseline',
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
        print(f"Module: sdpa_fp8_baseline (Cycle {cycle})")
        print()
        print("Next: Run benchmark with python bench/run_fp8_sdpa.py")
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
        print("  2. Verify sm_89 support: check L4 GPU")
        print("  3. Check FP8 support in CUDA toolkit")
        print()
        sys.exit(1)

if __name__ == '__main__':
    build_fp8_sdpa()

