"""
FlashMoE-Science: Split-K Build Configuration

Minimal build that only compiles flash_attention_science.cu (includes Split-K).
Excludes broken kernels.
"""

import os
import sys
from setuptools import setup, find_packages
from torch.utils.cpp_extension import BuildExtension, CUDAExtension, CUDA_HOME

# Check CUDA availability
if CUDA_HOME is None:
    print("WARNING: CUDA toolkit not found.")
    sys.exit(1)

# Auto-detect GPU architecture
try:
    import torch
    if torch.cuda.is_available():
        major, minor = torch.cuda.get_device_capability()
        a = f"{major}{minor}"
        gencodes = [f"-gencode=arch=compute_{a},code=sm_{a}"]
        print(f"Auto-detected GPU: SM_{a}")
    else:
        gencodes = ["-gencode=arch=compute_89,code=sm_89"]
        print("No GPU detected, defaulting to SM_89 (L4)")
except Exception:
    gencodes = ["-gencode=arch=compute_89,code=sm_89"]
    print("Could not detect GPU, defaulting to SM_89 (L4)")

# CUDA compilation flags
CUDA_FLAGS = [
    '-O3',
    '--use_fast_math',
    '-lineinfo',
    '--expt-relaxed-constexpr',
    '--expt-extended-lambda',
    '-Xcompiler=-fno-omit-frame-pointer',
    '-Xcompiler=-fno-common',
    '-Xfatbin=-compress-all',
    '-Xptxas=-v',
] + gencodes

# C++ compilation flags
CXX_FLAGS = [
    '-O3',
    '-std=c++17',
    '-fno-omit-frame-pointer',
    '-fno-common',
]

# CUDA extension modules - MINIMAL (only working kernels)
ext_modules = [
    CUDAExtension(
        name='flashmoe_science._C',
        sources=[
            'python/flashmoe_science/csrc/flash_attention_science.cu',  # Working kernel + Split-K
            'python/flashmoe_science/csrc/bindings_minimal.cpp',  # Minimal bindings (only 2 functions)
        ],
        include_dirs=[
            'kernels/attention/include',
            'kernels/moe/include',
            'kernels/utils',
        ],
        extra_compile_args={
            'cxx': CXX_FLAGS,
            'nvcc': CUDA_FLAGS,
        },
    ),
]

setup(
    name='flashmoe-science',
    version='0.1.0',
    author='GOATnote Autonomous Research Lab Initiative',
    author_email='b@thegoatnote.com',
    description='High-Performance CUDA Kernels for AI-Driven Scientific Discovery',
    url='https://github.com/GOATnote-Inc/periodicdent42/tree/main/flashmoe-science',
    license='MIT',
    packages=find_packages(where='python'),
    package_dir={'': 'python'},
    ext_modules=ext_modules,
    cmdclass={
        'build_ext': BuildExtension.with_options(no_python_abi_suffix=True),
    },
    install_requires=[
        'torch>=2.2.0',
        'numpy>=1.24.0',
    ],
    python_requires='>=3.10',
)

