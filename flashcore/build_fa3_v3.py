#!/usr/bin/env python3
"""
Build FA-3 v3 kernel
- Inverted loop structure (outer: K/V tiles, inner: query rows)
- WMMA basics for Tensor Cores
- Target: <40 μs on L4
"""

from torch.utils.cpp_extension import load
from pathlib import Path

print("=" * 70)
print("Building FA-3 v3 Kernel (Inverted Loops + WMMA)")
print("=" * 70)

src_dir = Path(__file__).parent / "kernels"

sources = [
    str(src_dir / "flashcore_fa3_v3.cu"),
    str(src_dir / "flashcore_fa3_v3_bindings.cu"),
]

cuda_flags = [
    "-O3",
    "-arch=sm_89",
    "-std=c++17",
    "--use_fast_math",
    "-Xptxas", "-v",
    "--maxrregcount=128",
    "-lineinfo",
    "-DM_TILE=64",
    "-DN_TILE=64",        # Smaller tile for better occupancy
    "-DWARPS_PER_BLOCK=4",
    "-DPAD=8",
]

print(f"Sources: {len(sources)} files")
print(f"Config: M_TILE=64, N_TILE=64, WARPS=4, PAD=8")
print(f"Architecture: Inverted loops (K/V outer, Q inner)")
print()

try:
    flashcore_fa3 = load(
        name="flashcore_fa3_v3",
        sources=sources,
        extra_cuda_cflags=cuda_flags,
        verbose=True,
        with_cuda=True,
    )
    
    print("\n" + "=" * 70)
    print("✅ BUILD SUCCESSFUL!")
    print("=" * 70)
    print("Module: flashcore_fa3_v3.forward(Q, K, V, is_causal=False)")
    print()
    
except Exception as e:
    print("\n" + "=" * 70)
    print("❌ BUILD FAILED!")
    print("=" * 70)
    print(f"Error: {e}")
    import sys
    sys.exit(1)

