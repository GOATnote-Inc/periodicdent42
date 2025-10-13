#!/usr/bin/env python3
"""
JSON-first benchmark for agentic parsing.
Based on 2025 best practices for autonomous optimization.
"""
import torch
import time
import statistics
import math
import json
import sys
import argparse
from datetime import datetime

def get_gpu_info():
    """Extract GPU architecture information."""
    if not torch.cuda.is_available():
        return None
    
    capability = torch.cuda.get_device_capability(0)
    return {
        "name": torch.cuda.get_device_name(0),
        "arch": f"SM_{capability[0]}{capability[1]}",
        "sm_count": torch.cuda.get_device_properties(0).multi_processor_count,
        "total_memory_gb": torch.cuda.get_device_properties(0).total_memory / 1e9,
        "cuda_version": torch.version.cuda,
        "pytorch_version": torch.__version__,
    }

def benchmark_kernel(fa_module, Q, K, V, scale, causal=True, warmup=20, runs=100):
    """Benchmark our kernel with proper warmup and statistics."""
    # Warmup
    for _ in range(warmup):
        _ = fa_module.flash_attention_forward(Q, K, V, causal, scale)
    torch.cuda.synchronize()
    
    # Measure
    times = []
    for _ in range(runs):
        torch.cuda.synchronize()
        start = time.perf_counter()
        O = fa_module.flash_attention_forward(Q, K, V, causal, scale)
        torch.cuda.synchronize()
        times.append((time.perf_counter() - start) * 1000)
    
    return {
        "mean_ms": statistics.mean(times),
        "std_ms": statistics.stdev(times) if len(times) > 1 else 0,
        "min_ms": min(times),
        "max_ms": max(times),
        "output": O,
    }

def benchmark_pytorch(Q, K, V, scale, warmup=20, runs=100):
    """Benchmark PyTorch SDPA."""
    S = Q.size(2)
    causal_mask = torch.triu(torch.ones(S, S, device='cuda', dtype=torch.bool), diagonal=1)
    
    # Warmup
    for _ in range(warmup):
        _ = torch.nn.functional.scaled_dot_product_attention(
            Q, K, V, attn_mask=~causal_mask, scale=scale
        )
    torch.cuda.synchronize()
    
    # Measure
    times = []
    for _ in range(runs):
        torch.cuda.synchronize()
        start = time.perf_counter()
        O = torch.nn.functional.scaled_dot_product_attention(
            Q, K, V, attn_mask=~causal_mask, scale=scale
        )
        torch.cuda.synchronize()
        times.append((time.perf_counter() - start) * 1000)
    
    return {
        "mean_ms": statistics.mean(times),
        "std_ms": statistics.stdev(times) if len(times) > 1 else 0,
        "min_ms": min(times),
        "max_ms": max(times),
        "output": O,
    }

def compute_metrics(B, H, S, D, result_ours, result_pytorch):
    """Compute performance metrics."""
    # FLOP count for attention: 4*B*H*S^2*D (Q@K^T + softmax@V)
    flops = 4 * B * H * S * S * D
    
    gflops_ours = (flops / 1e9) / (result_ours["mean_ms"] / 1000)
    gflops_pytorch = (flops / 1e9) / (result_pytorch["mean_ms"] / 1000)
    
    # Correctness
    max_diff = (result_ours["output"] - result_pytorch["output"]).abs().max().item()
    mean_diff = (result_ours["output"] - result_pytorch["output"]).abs().mean().item()
    
    # CTA estimate (assumes TILE_SIZE_M=64)
    TILE_SIZE_M = 64
    num_query_tiles = (S + TILE_SIZE_M - 1) // TILE_SIZE_M
    ctas = B * H * num_query_tiles
    
    return {
        "speedup_vs_pytorch": result_pytorch["mean_ms"] / result_ours["mean_ms"],
        "latency_ours_ms": result_ours["mean_ms"],
        "latency_pytorch_ms": result_pytorch["mean_ms"],
        "gflops_ours": gflops_ours,
        "gflops_pytorch": gflops_pytorch,
        "max_error": max_diff,
        "mean_error": mean_diff,
        "ctas_estimated": ctas,
    }

def main():
    parser = argparse.ArgumentParser(description="JSON-first CUDA kernel benchmark")
    parser.add_argument('--json', action='store_true', help='Output JSON only')
    parser.add_argument('--quick', action='store_true', help='Fast mode (10 runs)')
    parser.add_argument('--config', default='8,8,128,64', help='B,H,S,D configuration')
    args = parser.parse_args()
    
    # Parse config
    B, H, S, D = map(int, args.config.split(','))
    runs = 10 if args.quick else 100
    
    # GPU info
    gpu_info = get_gpu_info()
    if gpu_info is None:
        print(json.dumps({"error": "CUDA not available"}))
        sys.exit(1)
    
    # Load our kernel
    try:
        import flashmoe_science._C as fa
    except ImportError as e:
        print(json.dumps({"error": f"Failed to import kernel: {e}"}))
        sys.exit(1)
    
    # Create inputs
    Q = torch.randn(B, H, S, D, dtype=torch.float16, device='cuda')
    K, V = Q.clone(), Q.clone()
    scale = 1.0 / math.sqrt(D)
    
    # Benchmark both
    result_ours = benchmark_kernel(fa, Q, K, V, scale, runs=runs)
    result_pytorch = benchmark_pytorch(Q, K, V, scale, runs=runs)
    
    # Compute metrics
    metrics = compute_metrics(B, H, S, D, result_ours, result_pytorch)
    
    # Build output
    output = {
        "timestamp": datetime.now().isoformat(),
        "config": {"B": B, "H": H, "S": S, "D": D},
        "gpu": gpu_info,
        "metrics": metrics,
        "correctness": {
            "max_error": metrics["max_error"],
            "mean_error": metrics["mean_error"],
            "passed": metrics["max_error"] < 0.01,
        },
        "performance": {
            "speedup_vs_pytorch": metrics["speedup_vs_pytorch"],
            "ours": {
                "mean_ms": result_ours["mean_ms"],
                "std_ms": result_ours["std_ms"],
                "gflops": metrics["gflops_ours"],
            },
            "pytorch": {
                "mean_ms": result_pytorch["mean_ms"],
                "std_ms": result_pytorch["std_ms"],
                "gflops": metrics["gflops_pytorch"],
            },
        },
        "parallelism": {
            "ctas_estimated": metrics["ctas_estimated"],
            "sm_count": gpu_info["sm_count"],
            "utilization_pct": (metrics["ctas_estimated"] / gpu_info["sm_count"]) * 100,
        },
    }
    
    if args.json:
        # JSON-only output (for agentic parsing)
        print(json.dumps(output, indent=2))
    else:
        # Human-readable output
        print("=" * 60)
        print(f"Benchmark Results: B={B}, H={H}, S={S}, D={D}")
        print("=" * 60)
        print(f"GPU: {gpu_info['name']} ({gpu_info['arch']}, {gpu_info['sm_count']} SMs)")
        print()
        print("Performance:")
        print(f"  Our kernel:   {result_ours['mean_ms']:.3f} ± {result_ours['std_ms']:.3f} ms")
        print(f"  PyTorch SDPA: {result_pytorch['mean_ms']:.3f} ± {result_pytorch['std_ms']:.3f} ms")
        print(f"  Speedup:      {metrics['speedup_vs_pytorch']:.3f}×")
        print()
        print("Throughput:")
        print(f"  Our kernel:   {metrics['gflops_ours']:.1f} GFLOP/s")
        print(f"  PyTorch SDPA: {metrics['gflops_pytorch']:.1f} GFLOP/s")
        print()
        print("Correctness:")
        print(f"  Max error:    {metrics['max_error']:.6f}")
        print(f"  Mean error:   {metrics['mean_error']:.6f}")
        print(f"  Status:       {'PASS' if output['correctness']['passed'] else 'FAIL'}")
        print()
        print("Parallelism:")
        print(f"  CTAs:         {metrics['ctas_estimated']}")
        print(f"  Utilization:  {output['parallelism']['utilization_pct']:.1f}%")
        print()
        print("JSON output available with --json flag")

if __name__ == "__main__":
    main()

