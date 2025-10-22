#!/usr/bin/env python3
"""
FlashCore Latency Benchmark

Measures kernel performance with robust statistics (100-run medians).
Compares against PyTorch SDPA baseline.

Usage:
    python benchmarks/benchmark_latency.py --shape mission --iters 100
    python benchmarks/benchmark_latency.py --shape small --iters 10 --out results.json
    python benchmarks/benchmark_latency.py --all  # Benchmark all shapes

Output:
    - Console: Formatted table with latency and speedup
    - JSON (optional): Detailed results with timestamp, git info
"""

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

import torch

# Add project root
sys.path.insert(0, str(Path(__file__).parent.parent))

# ============================================================================
# Benchmark Functions
# ============================================================================

def benchmark_kernel(kernel, Q, K, V, scale, iters=100, warmup=20):
    """Benchmark kernel with CUDA event timing.
    
    Args:
        kernel: Compiled extension module
        Q, K, V: Input tensors
        scale: Softmax scale factor
        iters: Number of iterations
        warmup: Number of warmup iterations
    
    Returns:
        Dict with p50, p90, p99, mean, std, min, max (µs)
    """
    
    # Warmup
    for _ in range(warmup):
        _ = kernel.forward(Q, K, V, scale)
    torch.cuda.synchronize()
    
    # Measure
    times = []
    for _ in range(iters):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        
        start.record()
        O = kernel.forward(Q, K, V, scale)
        end.record()
        
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end) * 1000.0)  # Convert ms → µs
    
    return {
        "p50": statistics.median(times),
        "p90": statistics.quantiles(times, n=10)[8] if len(times) >= 10 else max(times),
        "p99": statistics.quantiles(times, n=100)[98] if len(times) >= 100 else max(times),
        "mean": statistics.mean(times),
        "std": statistics.stdev(times) if len(times) > 1 else 0.0,
        "min": min(times),
        "max": max(times),
    }

def benchmark_pytorch(Q, K, V, scale, iters=100, warmup=20):
    """Benchmark PyTorch SDPA.
    
    Returns:
        Dict with p50, p90, p99, mean, std (µs)
    """
    
    # Warmup
    for _ in range(warmup):
        with torch.backends.cuda.sdp_kernel(
            enable_flash=True, enable_math=True, enable_mem_efficient=True
        ):
            _ = torch.nn.functional.scaled_dot_product_attention(
                Q, K, V, attn_mask=None, dropout_p=0.0, is_causal=False, scale=scale
            )
    torch.cuda.synchronize()
    
    # Measure
    times = []
    for _ in range(iters):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        
        start.record()
        with torch.backends.cuda.sdp_kernel(
            enable_flash=True, enable_math=True, enable_mem_efficient=True
        ):
            O = torch.nn.functional.scaled_dot_product_attention(
                Q, K, V, attn_mask=None, dropout_p=0.0, is_causal=False, scale=scale
            )
        end.record()
        
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end) * 1000.0)  # ms → µs
    
    return {
        "p50": statistics.median(times),
        "p90": statistics.quantiles(times, n=10)[8] if len(times) >= 10 else max(times),
        "p99": statistics.quantiles(times, n=100)[98] if len(times) >= 100 else max(times),
        "mean": statistics.mean(times),
        "std": statistics.stdev(times) if len(times) > 1 else 0.0,
    }

# ============================================================================
# Shape Definitions
# ============================================================================

SHAPES = {
    "tiny": (1, 1, 32, 64),
    "small": (1, 2, 64, 64),
    "medium": (1, 4, 128, 64),
    "mission": (1, 8, 512, 64),
    "large": (1, 8, 1024, 64),
}

# ============================================================================
# Main Function
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="FlashCore latency benchmark")
    parser.add_argument("--shape", choices=list(SHAPES.keys()), default="mission",
                        help="Shape to benchmark (default: mission)")
    parser.add_argument("--all", action="store_true",
                        help="Benchmark all shapes")
    parser.add_argument("--iters", type=int, default=100,
                        help="Number of iterations (default: 100)")
    parser.add_argument("--warmup", type=int, default=20,
                        help="Number of warmup iterations (default: 20)")
    parser.add_argument("--out", help="Output JSON file")
    parser.add_argument("--no-pytorch", action="store_true",
                        help="Skip PyTorch baseline comparison")
    args = parser.parse_args()
    
    # Build kernel
    print("\nBuilding kernel...")
    from build import build_baseline
    kernel = build_baseline(verbose=False)
    
    # Select shapes to benchmark
    if args.all:
        shapes_to_bench = list(SHAPES.items())
    else:
        shapes_to_bench = [(args.shape, SHAPES[args.shape])]
    
    # Results storage
    all_results = {}
    
    # Print header
    print(f"\n{'='*100}")
    print(f"FlashCore Latency Benchmark (iters={args.iters}, warmup={args.warmup})")
    print(f"{'='*100}")
    print(f"{'Shape':<12} | {'Config':<20} | {'FlashCore (p50)':<15} | {'PyTorch (p50)':<15} | {'Speedup':<10}")
    print(f"{'-'*100}")
    
    # Benchmark each shape
    for shape_name, (B, H, S, D) in shapes_to_bench:
        # Create inputs
        torch.manual_seed(0)
        Q = torch.randn(B, H, S, D, dtype=torch.float16, device='cuda')
        K = torch.randn(B, H, S, D, dtype=torch.float16, device='cuda')
        V = torch.randn(B, H, S, D, dtype=torch.float16, device='cuda')
        scale = 1.0 / (D ** 0.5)
        
        # Benchmark FlashCore
        fc_stats = benchmark_kernel(kernel, Q, K, V, scale, args.iters, args.warmup)
        
        # Benchmark PyTorch (optional)
        if not args.no_pytorch:
            pt_stats = benchmark_pytorch(Q, K, V, scale, args.iters, args.warmup)
            speedup = pt_stats["p50"] / fc_stats["p50"]
            pt_str = f"{pt_stats['p50']:>8.1f} µs"
            speedup_str = f"{speedup:>6.2f}×"
        else:
            pt_stats = None
            pt_str = "N/A"
            speedup_str = "N/A"
        
        # Print results
        config_str = f"B={B},H={H},S={S},D={D}"
        print(f"{shape_name:<12} | {config_str:<20} | {fc_stats['p50']:>8.1f} µs     | {pt_str:<15} | {speedup_str:<10}")
        
        # Store results
        all_results[shape_name] = {
            "config": {"B": B, "H": H, "S": S, "D": D},
            "flashcore": fc_stats,
            "pytorch": pt_stats,
            "speedup": speedup if not args.no_pytorch else None,
        }
    
    print(f"{'='*100}\n")
    
    # Save results (if requested)
    if args.out:
        output = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "device": torch.cuda.get_device_name(0),
            "torch_version": torch.__version__,
            "cuda_version": torch.version.cuda,
            "iterations": args.iters,
            "warmup": args.warmup,
            "results": all_results,
        }
        
        with open(args.out, 'w') as f:
            json.dump(output, f, indent=2)
        
        print(f"Results saved to {args.out}\n")

if __name__ == "__main__":
    main()

