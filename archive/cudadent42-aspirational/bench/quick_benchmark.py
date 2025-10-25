#!/usr/bin/env python3
"""
Quick benchmark: V2 vs V3 performance comparison
"""

import torch
import torch.nn.functional as F
from pathlib import Path
import sys
import time
import pandas as pd

# Import kernels
sys.path.insert(0, str(Path(__file__).parent))
from build_v3_release import build_v3_release

# Import V2 (if available)
try:
    from fa_inverted_v2_tensor_cores import flash_attention_inverted_forward as v2_forward
except ImportError:
    print("Warning: V2 kernel not found, will only benchmark V3 vs SDPA")
    v2_forward = None


def benchmark_kernel(kernel_fn, Q, K, V, softmax_scale, is_causal, warmup=20, iters=100):
    """
    Benchmark a single kernel
    
    Returns:
        list of latencies in milliseconds
    """
    
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


def run_benchmark(B, H, S, D, is_causal=False, warmup=20, iters=100):
    """
    Benchmark V2, V3, and PyTorch SDPA
    
    Returns:
        DataFrame with results
    """
    
    assert S == 512, "V3 kernel specialized for S=512"
    assert D == 64, "V3 kernel specialized for D=64"
    
    device = "cuda"
    dtype = torch.float16
    
    print(f"\n{'=' * 80}")
    print(f"Benchmark: B={B}, H={H}, S={S}, D={D}, causal={is_causal}")
    print(f"Warmup: {warmup}, Iterations: {iters}")
    print(f"{'=' * 80}\n")
    
    # Create inputs
    torch.manual_seed(42)
    Q = torch.randn(B, H, S, D, device=device, dtype=dtype)
    K = torch.randn(B, H, S, D, device=device, dtype=dtype)
    V = torch.randn(B, H, S, D, device=device, dtype=dtype)
    
    softmax_scale = 1.0 / (D ** 0.5)
    
    results = []
    
    # Benchmark PyTorch SDPA
    print("Benchmarking PyTorch SDPA...")
    def sdpa_fn(Q, K, V, scale, causal):
        return F.scaled_dot_product_attention(Q, K, V, is_causal=causal, scale=scale)
    
    sdpa_latencies = benchmark_kernel(sdpa_fn, Q, K, V, softmax_scale, is_causal, warmup, iters)
    results.extend([{"kernel": "sdpa", "iter": i, "latency_ms": lat} 
                    for i, lat in enumerate(sdpa_latencies)])
    
    print(f"  SDPA: {sum(sdpa_latencies)/len(sdpa_latencies):.4f} ms (mean)")
    
    # Benchmark V2 (if available)
    if v2_forward is not None:
        print("\nBenchmarking V2 (Tensor Cores)...")
        def v2_fn(Q, K, V, scale, causal):
            return v2_forward(Q, K, V, softmax_scale=scale, is_causal=causal)
        
        v2_latencies = benchmark_kernel(v2_fn, Q, K, V, softmax_scale, is_causal, warmup, iters)
        results.extend([{"kernel": "v2", "iter": i, "latency_ms": lat} 
                        for i, lat in enumerate(v2_latencies)])
        
        print(f"  V2: {sum(v2_latencies)/len(v2_latencies):.4f} ms (mean)")
    
    # Benchmark V3
    print("\nBenchmarking V3 (Memory-Optimized)...")
    module = build_v3_release()
    config_id = 1  # Config 1: BLOCK_M=32, BLOCK_N=64, NUM_WARPS=4
    
    def v3_fn(Q, K, V, scale, causal):
        return module.forward(Q, K, V, scale, causal, config_id)
    
    v3_latencies = benchmark_kernel(v3_fn, Q, K, V, softmax_scale, is_causal, warmup, iters)
    results.extend([{"kernel": "v3", "iter": i, "latency_ms": lat} 
                    for i, lat in enumerate(v3_latencies)])
    
    print(f"  V3: {sum(v3_latencies)/len(v3_latencies):.4f} ms (mean)")
    
    # Create DataFrame
    df = pd.DataFrame(results)
    
    # Summary statistics
    print(f"\n{'=' * 80}")
    print("Summary Statistics (ms)")
    print(f"{'=' * 80}")
    summary = df.groupby("kernel")["latency_ms"].agg(["mean", "std", "min", "max"])
    print(summary)
    
    return df


if __name__ == "__main__":
    from pathlib import Path
    
    # Run benchmark
    df = run_benchmark(
        B=2, H=8, S=512, D=64,
        is_causal=False,
        warmup=20,
        iters=100
    )
    
    # Save results
    out_dir = Path(__file__).parent.parent.parent / "artifacts" / "bench"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "bs2_s512_h8_d64.csv"
    
    df.to_csv(out_file, index=False)
    
    print(f"\n{'=' * 80}")
    print(f"Results saved to: {out_file}")
    print(f"{'=' * 80}")

