#!/usr/bin/env python3
"""
Build FA-3 v5 kernel
- CORRECT loop order (K/V outer, Q inner)
- TRUE FlashAttention-3 architecture
- Expected: ~60 μs (50× faster than v3.1!)
"""

from torch.utils.cpp_extension import load
from pathlib import Path

print("=" * 70)
print("Building FA-3 v5 Kernel (OPTIMAL: K/V Outer + Shared State!)")
print("=" * 70)

src_dir = Path(__file__).parent / "kernels"

sources = [
    str(src_dir / "flashcore_fa3_v5.cu"),
    str(src_dir / "flashcore_fa3_v5_bindings.cu"),
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
    "-DN_TILE=64",
    "-DWARPS_PER_BLOCK=4",
    "-DPAD=8",
]

print(f"Sources: {len(sources)} files")
print(f"Config: M_TILE=64, N_TILE=64, WARPS=4, PAD=8")
print(f"Architecture: K/V outer loop (load ONCE!), Q inner loop ✅")
print(f"State: Register arrays per warp (persistent across tiles)")
print()

try:
    flashcore_fa3 = load(
        name="flashcore_fa3_v5",
        sources=sources,
        extra_cuda_cflags=cuda_flags,
        verbose=True,
        with_cuda=True,
    )
    
    print("\n" + "=" * 70)
    print("✅ BUILD SUCCESSFUL!")
    print("=" * 70)
    print("Module: flashcore_fa3_v5.forward(Q, K, V, is_causal=False)")
    print()
    
except Exception as e:
    print("\n" + "=" * 70)
    print("❌ BUILD FAILED!")
    print("=" * 70)
    print(f"Error: {e}")
    import sys
    sys.exit(1)

