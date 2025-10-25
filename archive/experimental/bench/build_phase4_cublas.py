#!/usr/bin/env python3
"""Build Phase 4 cuBLAS kernel with JIT compilation."""

import os
import torch
from torch.utils.cpp_extension import load

def build_phase4_cublas():
    """Build and return the Phase 4 cuBLAS module."""
    
    # Paths
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    kernel_cu = os.path.join(repo_root, "cudadent42/bench/kernels/fa_phase4_cublas.cu")
    bindings_cpp = os.path.join(repo_root, "cudadent42/bench/kernels/fa_phase4_cublas_bindings.cpp")
    
    # Verify files exist
    assert os.path.exists(kernel_cu), f"Kernel not found: {kernel_cu}"
    assert os.path.exists(bindings_cpp), f"Bindings not found: {bindings_cpp}"
    
    # Build
    print("Building Phase 4 cuBLAS kernel...")
    print(f"  Kernel: {kernel_cu}")
    print(f"  Bindings: {bindings_cpp}")
    
    module = load(
        name="fa_phase4_cublas",
        sources=[kernel_cu, bindings_cpp],
        extra_cuda_cflags=[
            "-O3",
            "-std=c++17",
            "-gencode=arch=compute_89,code=sm_89",
            "-use_fast_math",
            "--expt-relaxed-constexpr",
        ],
        extra_ldflags=["-lcublas"],
        verbose=True
    )
    
    print("✅ Build complete")
    return module

if __name__ == "__main__":
    build_phase4_cublas()

