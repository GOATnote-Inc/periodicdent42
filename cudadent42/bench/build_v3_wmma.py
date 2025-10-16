#!/usr/bin/env python3
"""
Build V3 WMMA kernel (Phase 3 implementation)
Uses Tensor Cores on Ada L4 (sm_89)
"""

import torch
import torch.utils.cpp_extension
from pathlib import Path
import sys
import os

def build_v3_wmma(debug: bool = False):
    """Build V3 WMMA kernel with Tensor Core optimizations"""
    
    kernel_dir = Path(__file__).parent / "kernels"
    kernel_cu = kernel_dir / "fa_s512_v3_wmma.cu"
    bindings_cpp = kernel_dir / "fa_s512_v3_wmma_bindings.cpp"
    
    # Validate files exist
    if not kernel_cu.exists():
        raise FileNotFoundError(f"Kernel not found: {kernel_cu}")
    if not bindings_cpp.exists():
        raise FileNotFoundError(f"Bindings not found: {bindings_cpp}")
    
    mode_str = "DEBUG MODE" if debug else "RELEASE MODE"
    print("=" * 80)
    print(f"🔧 Building V3 WMMA ({mode_str})")
    print("=" * 80)
    print(f"Kernel: {kernel_cu}")
    print(f"Bindings: {bindings_cpp}")
    print()
    
    # Get tile configuration from environment
    tile_m = os.environ.get("TILE_M", "128")
    tile_n = os.environ.get("TILE_N", "64")
    tile_k = os.environ.get("TILE_K", "32")
    stages = os.environ.get("STAGES", "2")
    accum_f32 = os.environ.get("ACCUM_F32", "1")
    
    print(f"WMMA Configuration:")
    print(f"  TILE_M:     {tile_m}")
    print(f"  TILE_N:     {tile_n}")
    print(f"  TILE_K:     {tile_k}")
    print(f"  STAGES:     {stages}")
    print(f"  ACCUM_F32:  {accum_f32}")
    print()
    
    # Base flags
    extra_cuda_cflags = [
        "-O3",                 # Maximum optimization
        "-use_fast_math",      # Fast math
        "-lineinfo",           # Keep line info for profiling
        "-Xptxas", "-v",       # Verbose ptxas (shows regs/thread, SMEM)
        "-std=c++17",
        "-gencode=arch=compute_89,code=sm_89",  # L4 Ada
        "-DUSE_WMMA",          # Enable WMMA code path
        f"-DTILE_M={tile_m}",
        f"-DTILE_N={tile_n}",
        f"-DTILE_K={tile_k}",
        f"-DSTAGES={stages}",
        f"-DACCUM_F32={accum_f32}",
    ]
    
    if debug:
        extra_cuda_cflags += ["-G", "-DDEBUG_V3"]  # Debug symbols + asserts
    else:
        extra_cuda_cflags += ["-DNDEBUG"]  # Disable asserts in release
    
    print("Build flags:")
    for flag in extra_cuda_cflags:
        print(f"  {flag}")
    print()
    
    # Compile
    print("Compiling...")
    module = torch.utils.cpp_extension.load(
        name="flash_attention_s512_v3_wmma",
        sources=[str(bindings_cpp), str(kernel_cu)],
        extra_cuda_cflags=extra_cuda_cflags,
        extra_cflags=["-std=c++17"],
        with_cuda=True,
        verbose=True,
    )
    
    print()
    print("=" * 80)
    print("✅ V3 WMMA build complete")
    print("=" * 80)
    print()
    print("⚠️  Check ptxas output above for:")
    print("  - Registers/thread (target: ≤ 64)")
    print("  - SMEM/CTA (target: ≤ 48 KB)")
    print("  - No 'local memory' warnings")
    print()
    
    return module


if __name__ == "__main__":
    try:
        module = build_v3_wmma()
        print(f"Module loaded: {module}")
        print()
        print("✅ Ready to test! Run:")
        print("   python3 scripts/test_v3_wmma.py")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Build failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

