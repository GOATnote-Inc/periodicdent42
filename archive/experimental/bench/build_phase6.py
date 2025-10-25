#!/usr/bin/env python3
"""
Build script for Phase 6 (Aggressive Scalar Optimization)
"""
import os
import sys
from pathlib import Path
import torch
import torch.utils.cpp_extension

def build_phase6():
    """Build Phase 6 kernel"""
    
    repo_root = Path(__file__).parent.parent.absolute()
    kernel_dir = repo_root / "cudadent42" / "bench" / "kernels"
    kernel_cu = kernel_dir / "fa_phase6_scalar.cu"
    bindings_cpp = kernel_dir / "fa_phase6_bindings.cpp"
    
    if not kernel_cu.exists():
        print(f"❌ Kernel not found: {kernel_cu}", file=sys.stderr)
        return 1
    
    if not bindings_cpp.exists():
        print(f"❌ Bindings not found: {bindings_cpp}", file=sys.stderr)
        return 1
    
    extra_cuda_cflags = [
        "-O3",
        "-use_fast_math",
        "-lineinfo",
        "-Xptxas", "-v",
        "-std=c++17",
        "-gencode=arch=compute_89,code=sm_89",
        "--expt-relaxed-constexpr",
    ]
    
    # Allow optional tile size override
    if 'TILE_M' in os.environ:
        extra_cuda_cflags.append(f"-DTILE_M={os.environ['TILE_M']}")
        print(f"  TILE_M={os.environ['TILE_M']}")
    
    if 'TILE_N' in os.environ:
        extra_cuda_cflags.append(f"-DTILE_N={os.environ['TILE_N']}")
        print(f"  TILE_N={os.environ['TILE_N']}")
    
    if 'NUM_THREADS' in os.environ:
        extra_cuda_cflags.append(f"-DNUM_THREADS={os.environ['NUM_THREADS']}")
        print(f"  NUM_THREADS={os.environ['NUM_THREADS']}")
    
    print("Building Phase 6 kernel...")
    try:
        module = torch.utils.cpp_extension.load(
            name="fa_phase6",
            sources=[str(bindings_cpp), str(kernel_cu)],
            extra_cuda_cflags=extra_cuda_cflags,
            extra_cflags=["-std=c++17"],
            with_cuda=True,
            verbose=False,
        )
        print("✅ Build successful")
        return 0
    except Exception as e:
        print(f"❌ Build failed: {e}", file=sys.stderr)
        return 1

if __name__ == "__main__":
    sys.exit(build_phase6())

