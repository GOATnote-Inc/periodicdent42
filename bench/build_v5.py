#!/usr/bin/env python3
"""Build V5 warp-specialized Tensor Core kernel."""

import os
from torch.utils.cpp_extension import load

def build_v5(M=64, N=64, K=32, STAGES=2, NUM_WARPS=8):
    """Build V5 kernel with specified tile configuration."""
    
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    kernel_cu = os.path.join(repo_root, "csrc/kernels/fa_v5_warp_spec.cu")
    bindings_cu = os.path.join(repo_root, "csrc/kernels/fa_v5_bindings.cu")
    
    assert os.path.exists(kernel_cu), f"Kernel not found: {kernel_cu}"
    assert os.path.exists(bindings_cu), f"Bindings not found: {bindings_cu}"
    
    cflags = [
        "-O3",
        "-use_fast_math",
        "-Xptxas=-v",
        "-std=c++17",
        "-gencode=arch=compute_89,code=sm_89",
        f"-DM_TILE={M}",
        f"-DN_TILE={N}",
        f"-DK_TILE={K}",
        f"-DSTAGES={STAGES}",
        f"-DNUM_WARPS={NUM_WARPS}",
        "--expt-relaxed-constexpr",
    ]
    
    module = load(
        name=f"fa_v5_M{M}_N{N}_K{K}_S{STAGES}_W{NUM_WARPS}",
        sources=[kernel_cu, bindings_cu],
        extra_cuda_cflags=cflags,
        verbose=False
    )
    
    return module

if __name__ == "__main__":
    print("Building V5 kernel...")
    mod = build_v5()
    print("✅ Build complete")

