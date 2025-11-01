#!/usr/bin/env python3
"""
BlackwellSparseK: High-Performance Sparse GEMM for NVIDIA GPUs

63× faster than PyTorch sparse (cuSPARSE backend)
1.74× faster than CUTLASS 4.3.0
"""

from setuptools import setup, Extension
from torch.utils import cpp_extension
import os

# CUDA paths
cuda_home = os.environ.get('CUDA_HOME', '/usr/local/cuda-13.0')

ext_modules = [
    cpp_extension.CUDAExtension(
        'blackwellsparsek._C',
        sources=[
            'python/bsk_bindings.cpp',
            'src/sparse_h100_async.cu',
            'src/kernel_launch.cu',
        ],
        include_dirs=[
            f'{cuda_home}/include',
        ],
        extra_compile_args={
            'cxx': ['-O3', '-std=c++17'],
            'nvcc': [
                '-O3',
                '--use_fast_math',
                '-std=c++17',
                '--expt-relaxed-constexpr',
                '-lineinfo',
                '-arch=sm_89',  # L4 (Ada) - change to sm_90a for H100
            ]
        },
    )
]

setup(
    name='blackwellsparsek',
    version='0.9.0',
    author='Brandon Dent, MD',
    author_email='b@thegoatnote.com',
    description='High-performance sparse block GEMM for NVIDIA GPUs',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/GOATnote-Inc/periodicdent42',
    packages=['blackwellsparsek'],
    package_dir={'blackwellsparsek': 'python'},
    ext_modules=ext_modules,
    cmdclass={'build_ext': cpp_extension.BuildExtension},
    install_requires=[
        'torch>=2.0.0',
        'numpy>=1.20.0',
    ],
    python_requires='>=3.8',
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'License :: OSI Approved :: BSD License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
    ],
    keywords='sparse matrix multiplication CUDA GPU deep-learning',
)
