#!/usr/bin/env python3
"""
Build CUTLASS FMHA kernel for PyTorch

This script compiles the CUTLASS FMHA wrapper with all necessary include paths.
CUTLASS must be cloned at ~/cutlass on the target machine.
"""

import os
import sys
from pathlib import Path
from torch.utils.cpp_extension import load

# Get paths
flashcore_dir = Path(__file__).parent
cutlass_dir = Path.home() / "cutlass"
cutlass_include = cutlass_dir / "include"
cutlass_examples = cutlass_dir / "examples" / "41_fused_multi_head_attention"
cutlass_tools = cutlass_dir / "tools" / "util" / "include"

# Verify CUTLASS exists
if not cutlass_dir.exists():
    print(f"❌ ERROR: CUTLASS not found at {cutlass_dir}")
    print(f"   Please run: git clone https://github.com/NVIDIA/cutlass.git ~/cutlass")
    sys.exit(1)

if not cutlass_examples.exists():
    print(f"❌ ERROR: CUTLASS FMHA example not found at {cutlass_examples}")
    print(f"   Please ensure CUTLASS is properly cloned.")
    sys.exit(1)

print("=" * 70)
print("Building FlashCore CUTLASS FMHA")
print("=" * 70)
print(f"CUTLASS dir: {cutlass_dir}")
print(f"FlashCore dir: {flashcore_dir}")
print()

# Source files
sources = [
    str(flashcore_dir / "kernels" / "flashcore_cutlass_wrapper.cu"),
    str(flashcore_dir / "kernels" / "flashcore_cutlass_bindings.cu"),
]

# Include paths
include_paths = [
    str(cutlass_include),
    str(cutlass_examples),
    str(cutlass_tools),
]

# CUDA flags
cuda_flags = [
    '-O3',
    '-arch=sm_89',  # L4 (Ada), fallback to sm_80 if needed
    '-std=c++17',   # CUTLASS requires C++17
    '--use_fast_math',
    '-Xptxas', '-v',
    '-lineinfo',
    '-DCUTLASS_ENABLE_TENSOR_CORES=1',
    # Suppress warnings from CUTLASS
    '-Xcudafe', '--diag_suppress=esa_on_defaulted_function_ignored',
]

print("Building with:")
print(f"  Sources: {len(sources)} files")
print(f"  Include paths: {len(include_paths)} dirs")
print(f"  CUDA flags: sm_89, C++17, O3")
print()

try:
    flashcore_cutlass = load(
        name='flashcore_cutlass',
        sources=sources,
        extra_include_paths=include_paths,
        extra_cuda_cflags=cuda_flags,
        verbose=True,
        with_cuda=True,
    )
    
    print()
    print("=" * 70)
    print("✅ BUILD SUCCESSFUL!")
    print("=" * 70)
    print(f"Module: {flashcore_cutlass}")
    print()
    print("Available functions:")
    print(f"  - flashcore_cutlass.fmha(Q, K, V) -> O")
    print()
    
except Exception as e:
    print()
    print("=" * 70)
    print("❌ BUILD FAILED!")
    print("=" * 70)
    print(f"Error: {e}")
    print()
    print("Troubleshooting:")
    print("  1. Check that CUTLASS is cloned at ~/cutlass")
    print("  2. Ensure CUDA 11.0+ is available")
    print("  3. Check that sm_89 is supported (or change to sm_80)")
    print()
    sys.exit(1)

