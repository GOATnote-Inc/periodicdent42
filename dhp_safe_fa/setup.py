from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
import os

# CUDA 13.0 path
cuda_home = '/usr/local/cuda-13.0'

setup(
    name='dhp_kernels',
    ext_modules=[
        # I4: Baseline constant-time kernel
        CUDAExtension(
            name='dhp_i4_kernel',
            sources=[
                'kernels/i4_fused_softmax_pv.cu',
                'kernels/i4_wrapper.cu',
            ],
            include_dirs=[
                'include',
                os.path.join(cuda_home, 'include'),
            ],
            extra_compile_args={
                'cxx': ['-O3'],
                'nvcc': [
                    '-O3',
                    '-std=c++17',
                    '-arch=sm_90a',
                    '--ptxas-options=-v',
                    '--use_fast_math',
                    '-lineinfo',
                ]
            }
        ),
        # I5: Warp-cooperative optimized kernel
        CUDAExtension(
            name='dhp_i5_kernel',
            sources=[
                'kernels/i5_warp_cooperative.cu',
                'kernels/i5_wrapper.cu',
            ],
            include_dirs=[
                'include',
                os.path.join(cuda_home, 'include'),
            ],
            extra_compile_args={
                'cxx': ['-O3'],
                'nvcc': [
                    '-O3',
                    '-std=c++17',
                    '-arch=sm_90a',
                    '--ptxas-options=-v',
                    '--use_fast_math',
                    '-lineinfo',
                ]
            }
        ),
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)

