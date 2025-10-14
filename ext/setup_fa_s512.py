#!/usr/bin/env python3
"""
Pre-Compiled Extension Setup for fa_s512 Kernel

Builds CUDA kernel via setuptools (not JIT) to avoid timeout issues.

Usage:
    # Build in-place
    cd ext && python setup_fa_s512.py build_ext --inplace
    
    # Test import
    python -c "import fa_s512; print('fa_s512 OK')"
    
    # Use in Python
    import fa_s512
    output = fa_s512.fa_s512(Q, K, V)  # Q/K/V are [B, H, 512, 64] FP16 tensors

Build Time:
    - First build: 5-15 minutes (cold cache, full compile)
    - Rebuild: 10-30 seconds (with ccache)

Environment:
    export TORCH_CUDA_ARCH_LIST="8.9"  # L4 only
    export MAX_JOBS=$(nproc)  # Parallel builds
    export CUDAFLAGS="-O3 --use_fast_math -lineinfo"
    export CCACHE_DIR="$HOME/.ccache"

Author: GOATnote Autonomous Research Lab Initiative
Date: 2025-10-14
"""

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
import os
import sys
from pathlib import Path

# Get environment variables
cuda_flags = os.environ.get("CUDAFLAGS", "").split()
if not cuda_flags:
    cuda_flags = ["-O3", "--use_fast_math", "-lineinfo"]

# Add architecture flag
arch = os.environ.get("TORCH_CUDA_ARCH_LIST", "8.9")
cuda_flags.append(f"-gencode=arch=compute_89,code=sm_89")

# Add standard flags
cuda_flags.extend([
    "--expt-relaxed-constexpr",
    "--expt-extended-lambda",
    "-std=c++17",
])

# Compiler optimization flags
cxx_flags = ["-O3", "-fno-omit-frame-pointer"]

# Source files (relative to ext/ directory)
# Note: fa_s512.cu needs to be compiled with specific defines
sources = [
    "fa_s512_bindings.cpp",
    "../cudadent42/bench/kernels/fa_s512.cu",
]

# Verify sources exist
for src in sources:
    src_path = Path(__file__).parent / src
    if not src_path.exists():
        print(f"❌ Source file not found: {src_path}")
        print(f"   Looking in: {src_path.parent}")
        print(f"   Files available: {list(src_path.parent.glob('*'))}")
        sys.exit(1)

print("🔧 Building fa_s512 extension")
print(f"  Sources: {sources}")
print(f"  CUDA flags: {cuda_flags}")
print(f"  CXX flags: {cxx_flags}")
print(f"  Architecture: {arch}")
print("")
print("⏱️  First build: 5-15 minutes (cold cache)")
print("⏱️  Rebuild: 10-30 seconds (with ccache)")
print("")

# Setup extension
setup(
    name="fa_s512",
    version="0.1.0",
    description="FlashAttention S=512 specialized kernel for L4 (SM_89)",
    author="GOATnote Autonomous Research Lab Initiative",
    ext_modules=[
        CUDAExtension(
            name="fa_s512",
            sources=sources,
            extra_compile_args={
                "cxx": cxx_flags,
                "nvcc": cuda_flags,
            },
            include_dirs=[
                "../cudadent42/bench/kernels",  # For any headers
            ],
        ),
    ],
    cmdclass={
        "build_ext": BuildExtension.with_options(
            no_python_abi_suffix=True,  # Simpler .so name
            use_ninja=True,  # Fast parallel builds
        )
    },
    python_requires=">=3.8",
    install_requires=[
        "torch>=2.0.0",
    ],
)

