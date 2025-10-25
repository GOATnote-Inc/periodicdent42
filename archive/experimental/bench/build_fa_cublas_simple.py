#!/usr/bin/env python3
"""Build simple cuBLAS attention kernel."""

import os
import torch
from torch.utils.cpp_extension import load

def build_fa_cublas_simple():
    """Build and return the module."""
    
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    kernel_cu = os.path.join(repo_root, "cudadent42/bench/kernels/fa_cublas_simple.cu")
    bindings_cpp = os.path.join(repo_root, "cudadent42/bench/kernels/fa_cublas_simple_bindings.cpp")
    
    assert os.path.exists(kernel_cu), f"Kernel not found: {kernel_cu}"
    assert os.path.exists(bindings_cpp), f"Bindings not found: {bindings_cpp}"
    
    print("Building simple cuBLAS attention...")
    
    module = load(
        name="fa_cublas_simple",
        sources=[kernel_cu, bindings_cpp],
        extra_cuda_cflags=[
            "-O3",
            "-std=c++17",
            "-gencode=arch=compute_89,code=sm_89",
            "-use_fast_math",
        ],
        extra_ldflags=["-lcublas"],
        verbose=True
    )
    
    print("✅ Build complete")
    return module

if __name__ == "__main__":
    build_fa_cublas_simple()

