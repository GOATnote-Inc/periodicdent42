#!/usr/bin/env python3
"""
BlackwellSparseK: Production CUDA Kernels for Blackwell Sparse Attention

Build configuration for CUDA extensions using PyTorch's C++ extension API.
Supports dual-architecture builds: sm_90a (Hopper H100) and sm_100 (Blackwell B200).
"""

import os
import sys
from pathlib import Path
from setuptools import setup, find_packages
from torch.utils.cpp_extension import BuildExtension, CUDAExtension, CUDA_HOME

# ============================================================================
# BUILD CONFIGURATION
# ============================================================================

# Check CUDA availability
if CUDA_HOME is None:
    print("❌ ERROR: CUDA toolkit not found")
    print("   Please install CUDA 13.0.2 and set CUDA_HOME environment variable")
    print("   Example: export CUDA_HOME=/usr/local/cuda-13.0")
    sys.exit(1)

# CUTLASS path
CUTLASS_PATH = os.environ.get("CUTLASS_PATH", "/opt/cutlass")
if not Path(CUTLASS_PATH).exists():
    print(f"⚠️  WARNING: CUTLASS not found at {CUTLASS_PATH}")
    print("   Set CUTLASS_PATH environment variable if installed elsewhere")
    print("   Example: export CUTLASS_PATH=/opt/cutlass")

# Target architectures
# Can be overridden with BSK_CUDA_ARCHS environment variable
CUDA_ARCHS = os.environ.get("BSK_CUDA_ARCHS", "90a;100")
archs = [arch.strip() for arch in CUDA_ARCHS.split(";") if arch.strip()]

# Generate gencode flags
gencodes = []
for arch in archs:
    # Remove 'a' suffix if present (sm_90a -> 90)
    arch_clean = arch.replace("a", "")
    gencodes.append(f"-gencode=arch=compute_{arch_clean},code=sm_{arch}")

print("=" * 80)
print("BlackwellSparseK CUDA Extension Build")
print("=" * 80)
print(f"  CUDA Home:      {CUDA_HOME}")
print(f"  CUTLASS Path:   {CUTLASS_PATH}")
print(f"  Target Archs:   {', '.join(archs)}")
print(f"  Gencodes:       {', '.join(gencodes)}")
print("=" * 80)

# ============================================================================
# CUDA COMPILATION FLAGS
# ============================================================================

cuda_flags = [
    '-O3',
    '--use_fast_math',
    '-lineinfo',  # Enable profiling with Nsight Compute
    '--expt-relaxed-constexpr',
    '--expt-extended-lambda',
    '-std=c++17',
    f'-I{CUTLASS_PATH}/include',
    f'-I{CUDA_HOME}/include',
] + gencodes

# Optional: Add debug flags
if os.environ.get("BSK_DEBUG", "0") == "1":
    cuda_flags.extend(["-G", "-DDEBUG=1", "-O0"])
    print("  Debug Mode:     ENABLED")

# Optional: Add profiling flags
if os.environ.get("BSK_PROFILE", "0") == "1":
    cuda_flags.extend(["-DPROFILE=1"])
    print("  Profiling:      ENABLED")

# Compiler optimization flags
cxx_flags = [
    '-O3',
    '-fPIC',
    '-fopenmp',  # OpenMP for multi-threading
]

# ============================================================================
# SOURCE FILES
# ============================================================================

# Package directory
package_dir = Path(__file__).parent / "src" / "blackwell_sparsek"
kernel_dir = package_dir / "kernels"

sources = [
    str(kernel_dir / "attention_fmha.cu"),
    str(kernel_dir / "kernel_dispatch.cu"),
    str(kernel_dir / "kernel_bindings.cpp"),
]

# Verify sources exist
missing_sources = []
for src in sources:
    if not Path(src).exists():
        missing_sources.append(src)

if missing_sources:
    print("❌ ERROR: Missing source files:")
    for src in missing_sources:
        print(f"   - {src}")
    sys.exit(1)

# ============================================================================
# EXTENSION DEFINITION
# ============================================================================

ext_modules = [
    CUDAExtension(
        name='blackwell_sparsek._C',
        sources=sources,
        extra_compile_args={
            'nvcc': cuda_flags,
            'cxx': cxx_flags,
        },
        libraries=['cudart', 'cuda'],
    )
]

# ============================================================================
# SETUP
# ============================================================================

setup(
    name="blackwell-sparsek",
    version="0.1.0",
    author="periodicdent42",
    author_email="expert@cuda.dev",
    description="Production CUDA kernels for Blackwell sparse attention using CUTLASS 4.3.0",
    long_description=open("README.md").read() if Path("README.md").exists() else "",
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/periodicdent42/tree/main/BlackwellSparseK",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    ext_modules=ext_modules,
    cmdclass={
        'build_ext': BuildExtension.with_options(use_ninja=True)
    },
    install_requires=[
        'torch>=2.0.0',
        'numpy>=1.24',
        'packaging>=23.0',
    ],
    extras_require={
        'dev': [
            'pytest>=7.0',
            'pytest-benchmark>=4.0',
            'black>=24.0',
            'ruff>=0.1',
        ],
        'bench': [
            'pandas>=2.0',
            'matplotlib>=3.7',
            'seaborn>=0.12',
        ],
        'xformers': [
            'xformers>=0.0.32',
        ],
        'vllm': [
            'vllm>=0.11.0',
        ],
    },
    python_requires='>=3.11',
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Programming Language :: C++",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)

