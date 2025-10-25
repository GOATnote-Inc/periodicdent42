#!/usr/bin/env python3
"""
Run SDPA once for Nsight Compute profiling

Minimal script to profile a single SDPA call

Author: Brandon Dent (b@thegoatnote.com)
License: Apache 2.0
"""

import argparse
import sys
from pathlib import Path
import torch
import torch.nn.functional as F

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from cudadent42.bench.common.env_lock import lock_environment


def run_sdpa_once(
    batch: int = 32,
    heads: int = 8,
    seq: int = 512,
    dim: int = 64
):
    """
    Run SDPA once for profiling
    """
    # Lock environment
    lock_environment()
    
    # Verify TF32 is disabled
    assert torch.backends.cuda.matmul.allow_tf32 == False, "TF32 not disabled!"
    
    device = torch.device("cuda")
    dtype = torch.float16
    
    # Create inputs
    Q = torch.randn(batch, heads, seq, dim, device=device, dtype=dtype)
    K = torch.randn(batch, heads, seq, dim, device=device, dtype=dtype)
    V = torch.randn(batch, heads, seq, dim, device=device, dtype=dtype)
    
    # Warmup (3 iterations)
    with torch.backends.cuda.sdp_kernel(enable_flash=True, enable_math=False, enable_mem_efficient=True):
        for _ in range(3):
            _ = F.scaled_dot_product_attention(Q, K, V)
    
    torch.cuda.synchronize()
    
    # Single profiled call
    with torch.backends.cuda.sdp_kernel(enable_flash=True, enable_math=False, enable_mem_efficient=True):
        output = F.scaled_dot_product_attention(Q, K, V)
    
    torch.cuda.synchronize()
    
    print(f"✅ SDPA executed: B={batch}, H={heads}, S={seq}, D={dim}")
    print(f"   Output shape: {output.shape}")


def main():
    parser = argparse.ArgumentParser(
        description="Run SDPA once for Nsight profiling"
    )
    parser.add_argument("--b", "--batch", type=int, default=32, dest="batch", help="Batch size")
    parser.add_argument("--h", "--heads", type=int, default=8, dest="heads", help="Number of heads")
    parser.add_argument("--s", "--seq", type=int, default=512, dest="seq", help="Sequence length")
    parser.add_argument("--d", "--dim", type=int, default=64, dest="dim", help="Head dimension")
    
    args = parser.parse_args()
    
    if not torch.cuda.is_available():
        print("❌ CUDA not available")
        return 1
    
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print("TF32 matmul: ", torch.backends.cuda.matmul.allow_tf32)
    print()
    
    try:
        run_sdpa_once(
            batch=args.batch,
            heads=args.heads,
            seq=args.seq,
            dim=args.dim
        )
        return 0
    except Exception as e:
        print(f"❌ Failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())

