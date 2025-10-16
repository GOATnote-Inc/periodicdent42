#!/usr/bin/env python3
"""
Build V3 kernel in RELEASE mode
"""

import torch
import torch.utils.cpp_extension
from pathlib import Path
import sys
import os

def build_v3_release(debug: bool = False):
    """Build V3 with release optimizations (or debug if flag set)"""
    
    kernel_dir = Path(__file__).parent / "kernels"
    kernel_cu = kernel_dir / "fa_s512_v3.cu"
    bindings_cpp = kernel_dir / "fa_s512_v3_bindings.cpp"
    
    mode_str = "DEBUG MODE" if debug else "RELEASE MODE"
    print("=" * 80)
    print(f"🔧 Building V3 ({mode_str})")
    print("=" * 80)
    print(f"Kernel: {kernel_cu}")
    print(f"Bindings: {bindings_cpp}")
    print()
    
    # Base flags (always included)
    extra_cuda_cflags = [
        "-O3",                 # Maximum optimization
        "-use_fast_math",      # Fast math
        "-lineinfo",           # Keep line info for profiling
        "-Xptxas", "-v",       # Verbose ptxas (shows regs/thread)
        "-std=c++17",
        "-gencode=arch=compute_89,code=sm_89",  # L4
    ]
    
    # WMMA toggle (default ON). Set USE_WMMA=0 to disable.
    if os.environ.get("USE_WMMA", "1") != "0":
        extra_cuda_cflags.append("-DUSE_WMMA")
        
        # WMMA tile configuration (Phase 3 jump)
        tile_m = os.environ.get("TILE_M", "128")
        tile_n = os.environ.get("TILE_N", "64")
        tile_k = os.environ.get("TILE_K", "32")
        stages = os.environ.get("STAGES", "2")
        accum_f32 = os.environ.get("ACCUM_F32", "1")
        
        extra_cuda_cflags += [
            f"-DTILE_M={tile_m}",
            f"-DTILE_N={tile_n}",
            f"-DTILE_K={tile_k}",
            f"-DSTAGES={stages}",
            f"-DACCUM_F32={accum_f32}",
        ]
        print(f"WMMA config: M={tile_m}, N={tile_n}, K={tile_k}, STAGES={stages}, FP32_ACCUM={accum_f32}")
    
    if debug:
        extra_cuda_cflags += ["-G", "-DDEBUG_V3"]  # Debug symbols + asserts
    else:
        extra_cuda_cflags += ["-DNDEBUG"]  # Disable asserts in release
    
    print("Build flags:")
    for flag in extra_cuda_cflags:
        print(f"  {flag}")
    print()
    
    # Compile
    module = torch.utils.cpp_extension.load(
        name="flash_attention_s512_v3_release",
        sources=[str(bindings_cpp), str(kernel_cu)],
        extra_cuda_cflags=extra_cuda_cflags,
        extra_cflags=["-std=c++17"],
        with_cuda=True,
        verbose=True,
    )
    
    print()
    print("=" * 80)
    print("✅ V3 release build complete")
    print("=" * 80)
    
    return module


if __name__ == "__main__":
    try:
        module = build_v3_release()
        print(f"\nModule loaded: {module}")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Build failed: {e}")
        sys.exit(1)

