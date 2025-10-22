#!/usr/bin/env python3
"""
Build FA-3 v2 kernel with correctness fixes
- Per-row K/V tile preload
- SMEM padding to avoid bank conflicts
- Better prefetch/barrier placement
"""

from torch.utils.cpp_extension import load
from pathlib import Path

print("=" * 70)
print("Building FA-3 v2 Kernel (Correctness Fixes)")
print("=" * 70)

src_dir = Path(__file__).parent / "kernels"

sources = [
    str(src_dir / "flashcore_fa3_kernel_v2.cu"),
    str(src_dir / "flashcore_fa3_bindings_v2.cu"),
]

cuda_flags = [
    "-O3",
    "-arch=sm_89",          # L4
    "-std=c++17",
    "--use_fast_math",
    "-Xptxas", "-v",
    "--maxrregcount=128",
    "-lineinfo",
    "-DM_TILE=64",
    "-DN_TILE=128",
    "-DWARPS_PER_BLOCK=4",
    "-DPAD=8",              # SMEM padding
]

print(f"Sources: {len(sources)} files")
print(f"Flags: M_TILE=64, N_TILE=128, WARPS=4, PAD=8")
print()

try:
    flashcore_fa3 = load(
        name="flashcore_fa3",
        sources=sources,
        extra_cuda_cflags=cuda_flags,
        verbose=True,
        with_cuda=True,
    )
    
    print("\n" + "=" * 70)
    print("✅ BUILD SUCCESSFUL!")
    print("=" * 70)
    print("Module: flashcore_fa3.forward(Q, K, V, is_causal=False)")
    print()
    
except Exception as e:
    print("\n" + "=" * 70)
    print("❌ BUILD FAILED!")
    print("=" * 70)
    print(f"Error: {e}")
    import sys
    sys.exit(1)

