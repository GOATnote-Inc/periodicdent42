#!/usr/bin/env python3
"""
Collect N=100 SDPA latencies for S=512 baseline

Saves raw latencies to .npy file for statistical analysis

Author: Brandon Dent (b@thegoatnote.com)
License: Apache 2.0
"""

import argparse
import sys
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from cudadent42.bench.common.env_lock import lock_environment


def collect_sdpa_latencies(
    batch: int = 32,
    heads: int = 8,
    seq: int = 512,
    dim: int = 64,
    iterations: int = 100,
    warmup: int = 20,
    output_path: str = "cudadent42/bench/artifacts/sdpa_s512_latencies.npy"
) -> np.ndarray:
    """
    Collect SDPA latencies with environment locking
    
    Returns:
        Array of N latencies in milliseconds
    """
    # Lock environment
    lock_environment()
    
    print(f"Collecting SDPA latencies: B={batch}, H={heads}, S={seq}, D={dim}")
    print(f"Iterations: {iterations}, Warmup: {warmup}")
    print()
    
    # Verify TF32 is disabled
    assert torch.backends.cuda.matmul.allow_tf32 == False, "TF32 not disabled!"
    assert torch.backends.cudnn.allow_tf32 == False, "TF32 cuDNN not disabled!"
    
    device = torch.device("cuda")
    dtype = torch.float16
    
    # Create inputs
    Q = torch.randn(batch, heads, seq, dim, device=device, dtype=dtype)
    K = torch.randn(batch, heads, seq, dim, device=device, dtype=dtype)
    V = torch.randn(batch, heads, seq, dim, device=device, dtype=dtype)
    
    # Warmup
    print("Warmup...")
    with torch.backends.cuda.sdp_kernel(enable_flash=True, enable_math=False, enable_mem_efficient=True):
        for _ in range(warmup):
            _ = F.scaled_dot_product_attention(Q, K, V)
    
    torch.cuda.synchronize()
    
    # Collect latencies
    print("Collecting measurements...")
    latencies = []
    
    with torch.backends.cuda.sdp_kernel(enable_flash=True, enable_math=False, enable_mem_efficient=True):
        for i in range(iterations):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            
            start.record()
            output = F.scaled_dot_product_attention(Q, K, V)
            end.record()
            
            torch.cuda.synchronize()
            elapsed_ms = start.elapsed_time(end)
            latencies.append(elapsed_ms)
            
            if (i + 1) % 25 == 0:
                print(f"  Progress: {i + 1}/{iterations}")
    
    latencies_array = np.array(latencies, dtype=np.float32)
    
    # Save to file
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    np.save(output_file, latencies_array)
    
    # Print statistics
    print()
    print("Statistics:")
    print(f"  N:       {len(latencies_array)}")
    print(f"  Median:  {np.median(latencies_array):.4f} ms")
    print(f"  Mean:    {np.mean(latencies_array):.4f} ms")
    print(f"  Std:     {np.std(latencies_array, ddof=1):.4f} ms")
    print(f"  Min:     {np.min(latencies_array):.4f} ms")
    print(f"  Max:     {np.max(latencies_array):.4f} ms")
    print()
    print(f"✅ Saved to: {output_file}")
    
    return latencies_array


def main():
    parser = argparse.ArgumentParser(
        description="Collect SDPA latencies for statistical analysis"
    )
    parser.add_argument("--batch", type=int, default=32, help="Batch size")
    parser.add_argument("--heads", type=int, default=8, help="Number of heads")
    parser.add_argument("--seq", type=int, default=512, help="Sequence length")
    parser.add_argument("--dim", type=int, default=64, help="Head dimension")
    parser.add_argument("--iterations", type=int, default=100, help="Number of iterations")
    parser.add_argument("--warmup", type=int, default=20, help="Warmup iterations")
    parser.add_argument(
        "--output",
        type=str,
        default="cudadent42/bench/artifacts/sdpa_s512_latencies.npy",
        help="Output path for .npy file"
    )
    
    args = parser.parse_args()
    
    if not torch.cuda.is_available():
        print("❌ CUDA not available")
        return 1
    
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print()
    
    try:
        collect_sdpa_latencies(
            batch=args.batch,
            heads=args.heads,
            seq=args.seq,
            dim=args.dim,
            iterations=args.iterations,
            warmup=args.warmup,
            output_path=args.output
        )
        return 0
    except Exception as e:
        print(f"❌ Failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())

