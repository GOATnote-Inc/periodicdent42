"""
FlashMoE-Science: High-Performance CUDA Kernels for Scientific AI

Build configuration for CUDA extensions.
"""

import os
import sys
from setuptools import setup, find_packages
from torch.utils.cpp_extension import BuildExtension, CUDAExtension, CUDA_HOME

# Check CUDA availability
if CUDA_HOME is None:
    print("WARNING: CUDA toolkit not found. CUDA extensions will not be built.")
    print("Please install CUDA toolkit and set CUDA_HOME environment variable.")
    sys.exit(1)

# CUDA compilation flags
CUDA_FLAGS = [
    '-O3',
    '--use_fast_math',
    '-lineinfo',  # For profiling with Nsight Compute
    '--expt-relaxed-constexpr',
    '--expt-extended-lambda',
    '-arch=sm_90',  # Hopper (H100) - adjust for your GPU
    # '-arch=sm_80',  # Ampere (A100)
]

# C++ compilation flags
CXX_FLAGS = [
    '-O3',
    '-std=c++17',
]

# Get version
def get_version():
    version_file = os.path.join(os.path.dirname(__file__), 'VERSION')
    if os.path.exists(version_file):
        with open(version_file) as f:
            return f.read().strip()
    return '0.1.0'

# Get long description
def get_long_description():
    readme_file = os.path.join(os.path.dirname(__file__), 'README.md')
    if os.path.exists(readme_file):
        with open(readme_file, encoding='utf-8') as f:
            return f.read()
    return ''

# CUDA extension modules
ext_modules = [
    CUDAExtension(
        name='flashmoe_science._C',
        sources=[
            'python/flashmoe_science/csrc/flash_attention_science.cu',
            'python/flashmoe_science/csrc/flash_attention_warp_specialized.cu',  # Phase 1: Warp specialization
            'python/flashmoe_science/csrc/flash_attention_backward.cu',
            'python/flashmoe_science/csrc/fused_moe.cu',
            'python/flashmoe_science/csrc/bindings.cpp',
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
    version=get_version(),
    author='GOATnote Autonomous Research Lab Initiative',
    author_email='b@thegoatnote.com',
    description='High-Performance CUDA Kernels for AI-Driven Scientific Discovery',
    long_description=get_long_description(),
    long_description_content_type='text/markdown',
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
    extras_require={
        'dev': [
            'pytest>=7.4.0',
            'pytest-benchmark>=4.0.0',
            'pytest-cov>=4.1.0',
            'black>=23.0.0',
            'flake8>=6.0.0',
            'mypy>=1.5.0',
        ],
        'integrations': [
            'vllm>=0.6.0',
            'sglang>=0.3.0',
        ],
        'benchmarks': [
            'wandb>=0.16.0',
            'tensorboard>=2.15.0',
            'matplotlib>=3.7.0',
            'seaborn>=0.13.0',
        ],
    },
    python_requires='>=3.10',
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Science/Research',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Programming Language :: Python :: 3.12',
        'Programming Language :: C++',
        'Programming Language :: CUDA',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Software Development :: Libraries :: Python Modules',
    ],
    keywords='cuda gpu optimization attention moe transformer scientific-computing',
)

