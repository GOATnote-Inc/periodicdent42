#!/usr/bin/env python3
"""
Build V3 kernel in RELEASE mode
"""

import torch
import torch.utils.cpp_extension
from pathlib import Path
import sys

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
        "-DUSE_WMMA",          # enable WMMA path
        "-DDEBUG_V3",          # enable asserts for evidence
    ]
    
    if debug:
        extra_cuda_cflags += ["-G"]  # Debug symbols
    
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

