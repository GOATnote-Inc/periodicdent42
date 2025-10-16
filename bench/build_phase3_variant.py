#!/usr/bin/env python3
"""
Parameterized build script for Phase 3 variants
Reads tunable parameters from environment variables
"""

import torch
import torch.utils.cpp_extension
from pathlib import Path
import sys
import os

def build_phase3_variant():
    """Build Phase 3 with environment-driven parameter overrides"""
    
    # Get absolute paths
    bench_dir = Path(__file__).parent.absolute()
    kernel_dir = bench_dir / "kernels"
    kernel_cu = kernel_dir / "fa_phase3_wmma.cu"
    bindings_cpp = kernel_dir / "fa_phase3_wmma_bindings.cpp"
    
    # Verify files exist
    if not kernel_cu.exists():
        raise FileNotFoundError(f"Kernel file not found: {kernel_cu}")
    if not bindings_cpp.exists():
        raise FileNotFoundError(f"Bindings file not found: {bindings_cpp}")
    
    # Base CUDA flags
    extra_cuda_cflags = [
        "-O3",
        "-use_fast_math",
        "-lineinfo",
        "-Xptxas", "-v",
        "-std=c++17",
        "-gencode=arch=compute_89,code=sm_89",
    ]
    
    # Read tunable parameters from environment
    tunable_params = ['BLOCK_M', 'NUM_WARPS', 'VEC_WIDTH', 'SMEM_STAGE', 'USE_WMMA']
    
    print("Build parameters:")
    for param in tunable_params:
        value = os.environ.get(param)
        if value is not None:
            extra_cuda_cflags.append(f"-D{param}={value}")
            print(f"  {param}={value}")
    
    # Handle REDUCE as a string macro
    reduce_strategy = os.environ.get('REDUCE')
    if reduce_strategy:
        extra_cuda_cflags.append(f'-DREDUCE_STR=\\"{reduce_strategy}\\"')
        print(f"  REDUCE=\"{reduce_strategy}\"")
    
    # Compile
    try:
        module = torch.utils.cpp_extension.load(
            name="fa_phase3",
            sources=[str(bindings_cpp), str(kernel_cu)],
            extra_cuda_cflags=extra_cuda_cflags,
            extra_cflags=["-std=c++17"],
            with_cuda=True,
            verbose=False,  # Suppress build output for cleaner sweep logs
        )
        
        print("✅ Build successful")
        return 0
    
    except Exception as e:
        print(f"❌ Build failed: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(build_phase3_variant())

