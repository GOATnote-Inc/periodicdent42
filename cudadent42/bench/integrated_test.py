#!/usr/bin/env python3
"""
Integrated benchmark test for performance ratchet validation

Uses PyTorch SDPA (no custom CUDA kernel required)
Outputs JSON compatible with performance_ratchet.py

Author: Brandon Dent (b@thegoatnote.com)
License: Apache 2.0
"""

import argparse
import json
import time
from pathlib import Path
from typing import Optional
import torch
import torch.nn.functional as F


def benchmark_pytorch_sdpa(
    batch: int,
    heads: int,
    seq: int,
    dim: int,
    iterations: int = 100,
    warmup: int = 20
) -> dict:
    """
    Benchmark PyTorch scaled_dot_product_attention
    
    Returns:
        dict with performance metrics
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float16
    
    # Create inputs
    Q = torch.randn(batch, heads, seq, dim, device=device, dtype=dtype)
    K = torch.randn(batch, heads, seq, dim, device=device, dtype=dtype)
    V = torch.randn(batch, heads, seq, dim, device=device, dtype=dtype)
    
    # Warmup
    for _ in range(warmup):
        with torch.backends.cuda.sdp_kernel(enable_flash=True, enable_math=False, enable_mem_efficient=True):
            _ = F.scaled_dot_product_attention(Q, K, V)
    
    torch.cuda.synchronize()
    
    # Benchmark
    times = []
    for _ in range(iterations):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        
        start.record()
        with torch.backends.cuda.sdp_kernel(enable_flash=True, enable_math=False, enable_mem_efficient=True):
            output = F.scaled_dot_product_attention(Q, K, V)
        end.record()
        
        torch.cuda.synchronize()
        elapsed_ms = start.elapsed_time(end)
        times.append(elapsed_ms)
    
    # Compute statistics
    import statistics
    mean_ms = statistics.mean(times)
    std_ms = statistics.stdev(times) if len(times) > 1 else 0.0
    median_ms = statistics.median(times)
    
    # Compute throughput (GFLOPS)
    # Attention compute: 2 * B * H * S * S * D (for Q@K^T) + 2 * B * H * S * S * D (for attn@V)
    flops = 4 * batch * heads * seq * seq * dim
    throughput_gflops = (flops / (mean_ms / 1000)) / 1e9
    
    # Compute bandwidth (GB/s)
    # Memory traffic: 3 * B * H * S * D * 2 bytes (Q,K,V) + B * H * S * D * 2 bytes (output)
    bytes_transferred = 3 * batch * heads * seq * dim * 2 + batch * heads * seq * dim * 2
    bandwidth_gb_s = (bytes_transferred / (mean_ms / 1000)) / 1e9
    
    return {
        "mean_time_ms": mean_ms,
        "std_dev_ms": std_ms,
        "median_ms": median_ms,
        "ci_95_low": median_ms - 1.96 * std_ms / (len(times) ** 0.5),
        "ci_95_high": median_ms + 1.96 * std_ms / (len(times) ** 0.5),
        "throughput_gflops": throughput_gflops,
        "bandwidth_gb_s": min(bandwidth_gb_s, 242.0),  # Cap at L4 theoretical peak
        "iterations": len(times),
        "config": {
            "batch": batch,
            "heads": heads,
            "seq": seq,
            "dim": dim,
            "dtype": str(dtype)
        }
    }


def main():
    parser = argparse.ArgumentParser(
        description="Integrated benchmark test for ratchet validation"
    )
    parser.add_argument(
        "--batch",
        type=int,
        default=32,
        help="Batch size (default: 32)"
    )
    parser.add_argument(
        "--heads",
        type=int,
        default=8,
        help="Number of attention heads (default: 8)"
    )
    parser.add_argument(
        "--seq",
        type=int,
        default=512,
        help="Sequence length (default: 512)"
    )
    parser.add_argument(
        "--dim",
        type=int,
        default=64,
        help="Head dimension (default: 64)"
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=100,
        help="Number of benchmark iterations (default: 100)"
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=20,
        help="Number of warmup iterations (default: 20)"
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Output path for JSON results"
    )
    parser.add_argument(
        "--config",
        type=str,
        help="Config name (used for autotune compatibility)"
    )
    
    args = parser.parse_args()
    
    # Check CUDA
    if not torch.cuda.is_available():
        print("❌ CUDA not available")
        return 1
    
    print(f"🔧 GPU: {torch.cuda.get_device_name(0)}")
    print(f"📊 Config: B={args.batch}, H={args.heads}, S={args.seq}, D={args.dim}")
    print(f"⏱️  Iterations: {args.iterations} (warmup: {args.warmup})")
    
    # Run benchmark
    print("\n🚀 Running benchmark...")
    result = benchmark_pytorch_sdpa(
        batch=args.batch,
        heads=args.heads,
        seq=args.seq,
        dim=args.dim,
        iterations=args.iterations,
        warmup=args.warmup
    )
    
    # Print results
    print(f"\n✅ Results:")
    print(f"   Latency: {result['mean_time_ms']:.4f} ms (±{result['std_dev_ms']:.4f} ms)")
    print(f"   Median:  {result['median_ms']:.4f} ms")
    print(f"   95% CI:  [{result['ci_95_low']:.4f}, {result['ci_95_high']:.4f}] ms")
    print(f"   Throughput: {result['throughput_gflops']:.1f} GFLOPS")
    print(f"   Bandwidth: {result['bandwidth_gb_s']:.1f} GB/s")
    
    # Save to file if requested
    if args.output:
        # Format for ratchet compatibility
        output_data = {
            "name": args.config or f"B{args.batch}_H{args.heads}_S{args.seq}_D{args.dim}",
            "performance": result,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "gpu": torch.cuda.get_device_name(0),
            "pytorch_version": torch.__version__,
        }
        
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, 'w') as f:
            json.dump(output_data, f, indent=2)
        
        print(f"\n💾 Results saved to {args.output}")
    
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())

