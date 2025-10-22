#!/usr/bin/env python3
"""Build simplified FA-3 kernel (no double-buffering)"""

from pathlib import Path
from torch.utils.cpp_extension import load

flashcore_dir = Path(__file__).parent

print("=" * 70)
print("Building FA-3 Simple Kernel (Debug Version)")
print("=" * 70)

sources = [
    str(flashcore_dir / "kernels" / "flashcore_fa3_simple.cu"),
    str(flashcore_dir / "kernels" / "flashcore_fa3_simple_bindings.cu"),
]

cuda_flags = [
    '-O3',
    '-arch=sm_89',
    '-std=c++17',
    '--use_fast_math',
    '-Xptxas', '-v',
    '--maxrregcount=128',
    '-lineinfo',
]

print(f"Sources: {len(sources)} files")
print(f"CUDA: sm_89, O3, maxreg=128")
print()

try:
    mod = load(
        name='flashcore_fa3_simple',
        sources=sources,
        extra_cuda_cflags=cuda_flags,
        verbose=True,
        with_cuda=True,
    )
    print("\n" + "=" * 70)
    print("✅ BUILD SUCCESSFUL!")
    print("=" * 70)
    print(f"Module: flashcore_fa3_simple.forward(Q, K, V)")
    print()
except Exception as e:
    print("\n" + "=" * 70)
    print("❌ BUILD FAILED!")
    print("=" * 70)
    print(f"Error: {e}")
    import sys
    sys.exit(1)

