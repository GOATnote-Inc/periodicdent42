#!/usr/bin/env python3
"""
Build V3 kernel in RELEASE mode
"""

import torch
import torch.utils.cpp_extension
from pathlib import Path
import sys

def build_v3_release():
    """Build V3 with release optimizations"""
    
    kernel_dir = Path(__file__).parent / "kernels"
    kernel_cu = kernel_dir / "fa_s512_v3.cu"
    bindings_cpp = kernel_dir / "fa_s512_v3_bindings.cpp"
    
    print("=" * 80)
    print("🔧 Building V3 (RELEASE MODE)")
    print("=" * 80)
    print(f"Kernel: {kernel_cu}")
    print(f"Bindings: {bindings_cpp}")
    print()
    
    # Release build flags
    release_flags = [
        "-O3",                 # Maximum optimization
        # NOTE: Removed -use_fast_math to test if it causes numerical errors
        # "-use_fast_math",      # Fast math
        "-DNDEBUG",            # Disable assertions
        "-UDEBUG_V3",          # Disable DEBUG_V3
        "-lineinfo",           # Keep line info for profiling
        "-std=c++17",
        "-arch=sm_89",         # L4
        "--expt-relaxed-constexpr",
        "--ptxas-options=-v",  # Verbose ptxas (shows regs/thread)
        "-ftz=false",          # Don't flush denormals to zero
        "-prec-div=true",      # Precise division
        "-prec-sqrt=true",     # Precise sqrt
    ]
    
    print("Release flags:")
    for flag in release_flags:
        print(f"  {flag}")
    print()
    
    # Compile
    module = torch.utils.cpp_extension.load(
        name="flash_attention_s512_v3_release",
        sources=[str(bindings_cpp), str(kernel_cu)],
        extra_cuda_cflags=release_flags,
        verbose=True,
        with_cuda=True,
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

