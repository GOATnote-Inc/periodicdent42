#!/usr/bin/env python3
"""
Enhanced Integrated Test with Advanced Statistics

Extends integrated_test.py with:
- Environment locking for reproducibility
- Advanced statistical analysis (Hedges' g, Cliff's Delta)
- Memory tracking
- Environment fingerprinting

Usage:
    python integrated_test_enhanced.py --batch 32 --heads 8 --seq 512 --dim 64
"""

import argparse
import json
import statistics
import time
from pathlib import Path
from typing import Dict, Any

import torch
import torch.nn.functional as F
import numpy as np

# Import new modules
from cudadent42.bench.common.env_lock import lock_environment, env_fingerprint, write_env
from cudadent42.bench.common.stats import compare_distributions, bootstrap_ci
from cudadent42.bench.common.memory_tracker import MemoryTracker, get_gpu_memory_info


def parse_args():
    parser = argparse.ArgumentParser(description="Enhanced Integrated CUDA Kernel Test")
    parser.add_argument("--output", type=Path, help="Output JSON file path")
    parser.add_argument("--batch", type=int, default=32, help="Batch size")
    parser.add_argument("--heads", type=int, default=8, help="Number of heads")
    parser.add_argument("--seq", type=int, default=512, help="Sequence length")
    parser.add_argument("--dim", type=int, default=64, help="Head dimension")
    parser.add_argument("--dtype", type=str, default="float16", choices=["float16", "bfloat16"])
    parser.add_argument("--iterations", type=int, default=100, help="Benchmark iterations")
    parser.add_argument("--warmup", type=int, default=20, help="Warmup iterations")
    parser.add_argument("--lock-env", action="store_true", help="Lock environment for reproducibility")
    return parser.parse_args()


def run_pytorch_sdpa_benchmark(B, H, S, d, dtype, iterations, warmup):
    """Benchmark PyTorch SDPA with memory tracking and advanced stats"""
    
    Q = torch.randn(B, H, S, d, device="cuda", dtype=dtype)
    K = torch.randn(B, H, S, d, device="cuda", dtype=dtype)
    V = torch.randn(B, H, S, d, device="cuda", dtype=dtype)
    
    # Warmup
    for _ in range(warmup):
        _ = F.scaled_dot_product_attention(Q, K, V)
    torch.cuda.synchronize()
    
    times = []
    with MemoryTracker() as mem_tracker:
        for _ in range(iterations):
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            
            start_event.record()
            _ = F.scaled_dot_product_attention(Q, K, V)
            end_event.record()
            torch.cuda.synchronize()
            times.append(start_event.elapsed_time(end_event))
    
    times_np = np.array(times)
    
    # Basic statistics
    mean_time_ms = float(np.mean(times_np))
    std_dev_ms = float(np.std(times_np, ddof=1))
    median_ms = float(np.median(times_np))
    
    # Bootstrap 95% CI
    ci_lower, ci_upper = bootstrap_ci(times_np, confidence=0.95, n_bootstrap=10000, seed=42)
    
    # Memory stats
    memory_stats = mem_tracker.get_stats()
    
    # Simplified GFLOPS calculation
    flops = 2 * B * H * S * d * S + 2 * B * H * S * S * d
    gflops = (flops / (mean_time_ms / 1000.0)) / 1e9 if mean_time_ms > 0 else 0
    
    # Bandwidth calculation
    bytes_accessed = 4 * B * H * S * d * 2  # 4 tensors, 2 bytes (FP16)
    bandwidth_gb_s = (bytes_accessed / (mean_time_ms / 1000.0)) / 1e9 if mean_time_ms > 0 else 0
    
    return {
        'statistics': {
            'median_ms': median_ms,
            'mean_ms': mean_time_ms,
            'std_ms': std_dev_ms,
            'ci_95_lower': float(ci_lower),
            'ci_95_upper': float(ci_upper),
            'iterations': iterations
        },
        'performance': {
            'throughput_gflops': gflops,
            'bandwidth_gb_s': bandwidth_gb_s
        },
        'memory': memory_stats.to_dict(),
        'raw_latencies': times_np.tolist()
    }


def main():
    args = parse_args()
    
    # Lock environment if requested
    if args.lock_env:
        print("🔒 Locking environment for reproducibility...")
        lock_environment()
        print()
    
    # GPU info
    print(f"🔧 GPU: {torch.cuda.get_device_name(0)}")
    gpu_mem = get_gpu_memory_info()
    print(f"💾 Memory: {gpu_mem['total']:.1f} MB total, {gpu_mem['free']:.1f} MB free")
    print(f"📊 Config: B={args.batch}, H={args.heads}, S={args.seq}, D={args.dim}")
    print()
    
    dtype = torch.float16 if args.dtype == "float16" else torch.bfloat16
    
    print(f"🚀 Running benchmark ({args.iterations} iterations, {args.warmup} warmup)...")
    bench_result = run_pytorch_sdpa_benchmark(
        args.batch, args.heads, args.seq, args.dim,
        dtype, args.iterations, args.warmup
    )
    
    print("\n✅ Results:")
    print(f"   Latency: {bench_result['statistics']['median_ms']:.4f} ms")
    print(f"   95% CI: [{bench_result['statistics']['ci_95_lower']:.4f}, "
          f"{bench_result['statistics']['ci_95_upper']:.4f}]")
    print(f"   Throughput: {bench_result['performance']['throughput_gflops']:.1f} GFLOPS")
    print(f"   Bandwidth: {bench_result['performance']['bandwidth_gb_s']:.1f} GB/s")
    print(f"   Peak Memory: {bench_result['memory']['peak_allocated_mb']:.2f} MB")
    
    # Save results
    if args.output:
        combined_result = {
            'config': {
                'batch_size': args.batch,
                'num_heads': args.heads,
                'seq_len': args.seq,
                'head_dim': args.dim,
                'dtype': args.dtype,
                'iterations': args.iterations,
                'warmup': args.warmup
            },
            'results': bench_result,
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'environment': env_fingerprint() if args.lock_env else None
        }
        
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, 'w') as f:
            json.dump(combined_result, f, indent=2)
        print(f"\n💾 Results saved to {args.output}")
        
        # Save environment fingerprint if locked
        if args.lock_env:
            env_path = args.output.parent / "env.json"
            write_env(str(env_path))


if __name__ == "__main__":
    main()

