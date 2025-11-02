from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
import os

setup(
    name='sparse_autotune',
    ext_modules=[
        CUDAExtension(
            name='sparse_autotune_cpp',
            sources=[
                'sparse_autotune.cpp',
                '../src/sparse/bsr_kernel_64.cu',
                '../src/sparse/cusparse_baseline.cu',
            ],
            include_dirs=[
                '../src/sparse',
                '/usr/local/cuda/include',
                '/opt/cutlass/include',
            ],
            library_dirs=['/usr/local/cuda/lib64'],
            libraries=['cudart', 'cusparse', 'curand'],
            extra_compile_args={
                'cxx': ['-O3', '-std=c++17'],
                'nvcc': [
                    '-O3',
                    '--expt-relaxed-constexpr',
                    '-lineinfo',
                    '-arch=sm_90',  # H100
                    '--use_fast_math',
                ],
            },
        )
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)
