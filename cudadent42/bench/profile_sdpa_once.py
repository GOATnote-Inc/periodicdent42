#!/usr/bin/env python3
"""
Single PyTorch SDPA Call for Nsight Compute Profiling

Runs a single forward pass of PyTorch SDPA for profiling with Nsight Compute.
Minimal overhead to ensure profile captures kernel execution cleanly.

Usage:
    python bench/profile_sdpa_once.py --b 32 --h 8 --s 512 --d 64
    
    ncu --set full -o profile python bench/profile_sdpa_once.py --b 32 --h 8 --s 512 --d 64

Author: GOATnote Autonomous Research Lab Initiative
Date: 2025-10-14
"""

import sys
from pathlib import Path

import torch
import torch.nn.functional as F

# Add project to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from cudadent42.bench.common.env_lock import lock_environment


def profile_sdpa_once(
    batch: int,
    heads: int,
    seq: int,
    dim: int
):
    """
    Run single SDPA forward pass
    
    Args:
        batch, heads, seq, dim: Tensor dimensions
    """
    device = torch.device("cuda")
    dtype = torch.float16
    
    # Create inputs
    Q = torch.randn(batch, heads, seq, dim, device=device, dtype=dtype)
    K = torch.randn(batch, heads, seq, dim, device=device, dtype=dtype)
    V = torch.randn(batch, heads, seq, dim, device=device, dtype=dtype)
    
    # Warmup (1 iteration to compile kernels)
    with torch.backends.cuda.sdp_kernel(enable_flash=True, enable_math=False, enable_mem_efficient=False):
        _ = F.scaled_dot_product_attention(Q, K, V)
    
    torch.cuda.synchronize()
    
    # Single profiled call
    with torch.backends.cuda.sdp_kernel(enable_flash=True, enable_math=False, enable_mem_efficient=False):
        output = F.scaled_dot_product_attention(Q, K, V)
    
    torch.cuda.synchronize()
    
    # Touch output to ensure it's computed
    _ = output.sum()
    
    print(f"Profiled SDPA: B={batch}, H={heads}, S={seq}, D={dim}")
    print(f"  Output shape: {output.shape}")
    print(f"  Output dtype: {output.dtype}")


def main():
    """CLI entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Run single SDPA call for profiling")
    parser.add_argument("--b", "--batch", type=int, default=32, help="Batch size")
    parser.add_argument("--h", "--heads", type=int, default=8, help="Number of attention heads")
    parser.add_argument("--s", "--seq", type=int, default=512, help="Sequence length")
    parser.add_argument("--d", "--dim", type=int, default=64, help="Head dimension")
    
    args = parser.parse_args()
    
    # Lock environment
    lock_environment()
    
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"CUDA: {torch.version.cuda}")
    print(f"PyTorch: {torch.__version__}")
    print(f"Environment: FP16, TF32 off, deterministic")
    print()
    
    # Run profiling
    profile_sdpa_once(
        batch=args.b,
        heads=args.h,
        seq=args.s,
        dim=args.d
    )
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

