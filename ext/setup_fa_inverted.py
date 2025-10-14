"""
Build script for FlashAttention Inverted kernel
Author: periodicdent42
Date: October 14, 2025

Builds a pre-compiled CUDA extension for the inverted FlashAttention kernel.
"""

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
import os

# Get CUDAFLAGS from environment, split into a list
cudaflags = os.environ.get("CUDAFLAGS", "").split()

# If no flags specified, use expert defaults
if not cudaflags:
    cudaflags = [
        '-O3',
        '--use_fast_math',
        '-lineinfo',
        '--expt-relaxed-constexpr',
        '--expt-extended-lambda',
        '-std=c++17',
        '-gencode=arch=compute_89,code=sm_89',  # L4 (SM_89)
    ]

setup(
    name="fa_inverted",
    ext_modules=[
        CUDAExtension(
            "fa_inverted",
            sources=[
                "fa_inverted_bindings.cpp",
                "../cudadent42/bench/kernels/fa_inverted.cu"
            ],
            extra_compile_args={
                "cxx": ["-O3", "-fno-omit-frame-pointer"],
                "nvcc": cudaflags
            },
        ),
    ],
    cmdclass={"build_ext": BuildExtension.with_options(no_python_abi_suffix=True)},
)

