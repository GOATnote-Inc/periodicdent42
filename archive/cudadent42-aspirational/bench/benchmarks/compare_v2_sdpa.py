#!/usr/bin/env python3
"""
Benchmark V2 vs PyTorch SDPA
Release build, fair comparison (no fast_math for initial comparison)
"""

import torch
import torch.nn.functional as F
from pathlib import Path
import sys
import pandas as pd

# Import V2 kernel
sys.path.insert(0, str(Path(__file__).parent.parent))
from fa_inverted_v2_tensor_cores import flash_attention_inverted_forward as v2_forward


def benchmark_kernel(kernel_fn, Q, K, V, softmax_scale, is_causal, warmup=20, iters=100):
    """Benchmark a single kernel"""
    device = Q.device
    latencies = []
    
    # Warmup
    for _ in range(warmup):
        _ = kernel_fn(Q, K, V, softmax_scale, is_causal)
    torch.cuda.synchronize(device)
    
    # Benchmark
    for _ in range(iters):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        
        start.record()
        _ = kernel_fn(Q, K, V, softmax_scale, is_causal)
        end.record()
        
        torch.cuda.synchronize(device)
        latencies.append(start.elapsed_time(end))
    
    return latencies


def run_bench(B, S, H, D, warmup=20, iters=100, csv_out=None):
    """
    Benchmark V2 vs SDPA
    
    Args:
        B: Batch size
        S: Sequence length
        H: Number of heads
        D: Head dimension
        warmup: Warmup iterations
        iters: Benchmark iterations
        csv_out: Optional CSV output path
    
    Returns:
        DataFrame with results
    """
    
    device = "cuda"
    dtype = torch.float16
    
    print(f"\n{'=' * 80}")
    print(f"Benchmark: B={B}, S={S}, H={H}, D={D}")
    print(f"Warmup: {warmup}, Iterations: {iters}")
    print(f"{'=' * 80}\n")
    
    # Create inputs
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    Q = torch.randn(B, H, S, D, device=device, dtype=dtype)
    K = torch.randn(B, H, S, D, device=device, dtype=dtype)
    V = torch.randn(B, H, S, D, device=device, dtype=dtype)
    
    softmax_scale = 1.0 / (D ** 0.5)
    is_causal = False  # Non-causal for comparison
    
    results = []
    
    # Benchmark PyTorch SDPA
    print("Benchmarking PyTorch SDPA...")
    def sdpa_fn(Q, K, V, scale, causal):
        return F.scaled_dot_product_attention(Q, K, V, is_causal=causal, scale=scale)
    
    sdpa_latencies = benchmark_kernel(sdpa_fn, Q, K, V, softmax_scale, is_causal, warmup, iters)
    results.extend([{"kernel": "sdpa", "iter": i, "latency_ms": lat} 
                    for i, lat in enumerate(sdpa_latencies)])
    
    sdpa_mean = sum(sdpa_latencies) / len(sdpa_latencies)
    print(f"  SDPA: {sdpa_mean:.4f} ms (mean)")
    
    # Benchmark V2
    print("\nBenchmarking V2 (Tensor Cores)...")
    def v2_fn(Q, K, V, scale, causal):
        return v2_forward(Q, K, V, softmax_scale=scale, is_causal=causal)
    
    v2_latencies = benchmark_kernel(v2_fn, Q, K, V, softmax_scale, is_causal, warmup, iters)
    results.extend([{"kernel": "v2", "iter": i, "latency_ms": lat} 
                    for i, lat in enumerate(v2_latencies)])
    
    v2_mean = sum(v2_latencies) / len(v2_latencies)
    print(f"  V2: {v2_mean:.4f} ms (mean)")
    
    # Speedup
    speedup = sdpa_mean / v2_mean
    print(f"\nSpeedup: V2 is {speedup:.2f}× vs SDPA")
    
    # Create DataFrame
    df = pd.DataFrame(results)
    
    # Summary statistics
    print(f"\n{'=' * 80}")
    print("Summary Statistics (ms)")
    print(f"{'=' * 80}")
    summary = df.groupby("kernel")["latency_ms"].agg(["mean", "std", "min", "max"])
    print(summary)
    
    # Save if requested
    if csv_out:
        out_path = Path(csv_out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out_path, index=False)
        print(f"\nSaved to: {out_path}")
    
    return df


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--B", type=int, default=2)
    parser.add_argument("--S", type=int, default=512)
    parser.add_argument("--H", type=int, default=8)
    parser.add_argument("--D", type=int, default=64)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--csv", type=str, default="artifacts/bench/v2_vs_sdpa.csv")
    
    args = parser.parse_args()
    
    run_bench(args.B, args.S, args.H, args.D, args.warmup, args.iters, args.csv)

