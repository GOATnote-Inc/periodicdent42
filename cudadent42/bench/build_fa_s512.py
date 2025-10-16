#!/usr/bin/env python3
"""
Build fa_s512.cu kernel (existing working baseline)
This is the starting point for EvoEngineer optimization
"""

import torch
import torch.utils.cpp_extension
from pathlib import Path
import sys

def build_fa_s512():
    """Build fa_s512 kernel with existing bindings"""
    
    kernel_cu = Path("cudadent42/bench/kernels/fa_s512.cu")
    bindings_cpp = Path("ext/fa_s512_bindings.cpp")
    
    print("=" * 80)
    print("🔧 Building fa_s512.cu (Existing Baseline for EvoEngineer)")
    print("=" * 80)
    print(f"Kernel:   {kernel_cu}")
    print(f"Bindings: {bindings_cpp}")
    print()
    
    # Validate files exist
    if not kernel_cu.exists():
        raise FileNotFoundError(f"Kernel not found: {kernel_cu}")
    if not bindings_cpp.exists():
        raise FileNotFoundError(f"Bindings not found: {bindings_cpp}")
    
    # Build flags (Iteration 1: SMEM overflow fixed)
    extra_cuda_cflags = [
        "-O3",
        "-use_fast_math",
        "-lineinfo",
        "-Xptxas", "-v",
        "-std=c++17",
        "-gencode=arch=compute_89,code=sm_89",  # L4
        "-DBLOCK_M=128",     # ✅ Iteration 1: 2× larger tiles
        "-DBLOCK_N=64",      # Kept at 64 for SMEM budget
        "-DBLOCK_K=32",
        "-DNUM_WARPS=8",     # ✅ Iteration 1: Doubled for larger tiles
        "-DSTAGES=1",
        "-DNDEBUG",
    ]
    
    print("Build flags:")
    for flag in extra_cuda_cflags:
        print(f"  {flag}")
    print()
    
    print("Configuration (Iteration 1: SMEM overflow fixed):")
    print("  BLOCK_M=128, BLOCK_N=64, NUM_WARPS=8, STAGES=1")
    print("  Expected: ~200-240 μs (1.3-1.6× speedup)")
    print()
    
    # Compile
    print("Compiling...")
    module = torch.utils.cpp_extension.load(
        name="flash_attention_s512",
        sources=[str(bindings_cpp), str(kernel_cu)],
        extra_cuda_cflags=extra_cuda_cflags,
        extra_cflags=["-std=c++17"],
        with_cuda=True,
        verbose=True,
    )
    
    print()
    print("=" * 80)
    print("✅ fa_s512.cu build complete")
    print("=" * 80)
    print()
    print("Next: Run baseline test")
    print("  python3 scripts/test_fa_s512_baseline.py")
    print()
    
    return module


if __name__ == "__main__":
    try:
        module = build_fa_s512()
        print(f"Module loaded: {module}")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Build failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

