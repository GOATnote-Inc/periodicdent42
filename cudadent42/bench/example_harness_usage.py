#!/usr/bin/env python3
"""
Example: Using the CUDA Benchmark Harness
==========================================
Demonstrates how to integrate the production harness with PyTorch SDPA.
"""

import torch
from pathlib import Path
from benchmark_harness import CUDABenchmarkHarness, BenchmarkConfig


def benchmark_pytorch_sdpa():
    """Example: Benchmark PyTorch SDPA using the harness"""
    
    # Configuration
    B, H, S, d = 32, 8, 128, 64
    dtype = torch.float16
    causal = False
    scale = 1.0 / (d ** 0.5)
    
    # Create inputs
    Q = torch.randn(B, H, S, d, dtype=dtype, device='cuda')
    K = torch.randn(B, H, S, d, dtype=dtype, device='cuda')
    V = torch.randn(B, H, S, d, dtype=dtype, device='cuda')
    
    # Kernel wrapper that returns timing in milliseconds
    def sdpa_wrapper():
        """Wrapper for PyTorch SDPA with CUDA event timing"""
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        
        start_event.record()
        O = torch.nn.functional.scaled_dot_product_attention(
            Q, K, V, scale=scale, is_causal=causal
        )
        end_event.record()
        torch.cuda.synchronize()
        
        return start_event.elapsed_time(end_event)
    
    # Calculate theoretical values for metrics
    # FlashAttention FLOP count: 4*B*H*S^2*d (Q@K^T + softmax@V)
    flop_count = 4 * B * H * S * S * d
    
    # Memory access: Q, K, V read (3*B*H*S*d), O write (B*H*S*d)
    # Total: 4*B*H*S*d elements, each 2 bytes (FP16)
    memory_bytes = 4 * B * H * S * d * 2
    
    # Configure harness
    config = BenchmarkConfig(
        warmup_iterations=200,      # Match paper methodology
        benchmark_iterations=500,   # Match paper methodology
        flush_l2_cache=False,       # Requires custom kernel
        lock_clocks=False,          # Requires sudo
        exclude_memory_transfers=True
    )
    
    # Create harness and run benchmark
    harness = CUDABenchmarkHarness(config)
    
    result = harness.benchmark_kernel(
        kernel_func=sdpa_wrapper,
        kernel_name=f"pytorch_sdpa_B{B}_H{H}_S{S}_d{d}",
        flop_count=flop_count,
        memory_bytes=memory_bytes,
        # Parameters for documentation
        batch_size=B,
        num_heads=H,
        seq_len=S,
        head_dim=d,
        dtype=str(dtype),
        causal=causal
    )
    
    # Save results
    output_dir = Path(__file__).parent / "out" / "harness_results"
    output_path = output_dir / f"sdpa_B{B}_H{H}_S{S}_d{d}.json"
    harness.save_results(result, output_path)
    
    # Compare to baseline if exists
    baseline_path = output_dir / "baseline.json"
    if baseline_path.exists():
        comparison = harness.compare_results(baseline_path, result)
        print(f"\nSpeedup vs baseline: {comparison['speedup']:.3f}x")
    
    return result


def benchmark_sweep():
    """Run benchmark sweep across multiple configurations"""
    configs = [
        (1, 1, 128, 64),    # Small: single query
        (8, 4, 128, 64),    # Medium: small batch
        (32, 8, 128, 64),   # Large: production batch
        (32, 8, 256, 64),   # XL: long sequence
    ]
    
    results = []
    for B, H, S, d in configs:
        print(f"\n{'='*70}")
        print(f"Configuration: B={B}, H={H}, S={S}, d={d}")
        print(f"{'='*70}")
        
        # Create inputs
        Q = torch.randn(B, H, S, d, dtype=torch.float16, device='cuda')
        K = torch.randn(B, H, S, d, dtype=torch.float16, device='cuda')
        V = torch.randn(B, H, S, d, dtype=torch.float16, device='cuda')
        scale = 1.0 / (d ** 0.5)
        
        def sdpa_wrapper():
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            _ = torch.nn.functional.scaled_dot_product_attention(Q, K, V, scale=scale)
            end.record()
            torch.cuda.synchronize()
            return start.elapsed_time(end)
        
        # Benchmark
        config = BenchmarkConfig(warmup_iterations=50, benchmark_iterations=200)
        harness = CUDABenchmarkHarness(config)
        
        result = harness.benchmark_kernel(
            kernel_func=sdpa_wrapper,
            kernel_name=f"sdpa_B{B}_H{H}_S{S}",
            flop_count=4 * B * H * S * S * d,
            memory_bytes=4 * B * H * S * d * 2,
            B=B, H=H, S=S, d=d
        )
        
        results.append(result)
    
    # Summary table
    print(f"\n{'='*70}")
    print("SUMMARY: PyTorch SDPA Performance on L4")
    print(f"{'='*70}")
    print(f"{'Config':<20} {'Mean (ms)':<12} {'GFLOPS':<12} {'BW (GB/s)':<12}")
    print("-" * 70)
    
    for result in results:
        params = result.parameters
        config_str = f"B={params['B']} H={params['H']} S={params['S']}"
        m = result.metrics
        print(f"{config_str:<20} {m.mean_time_ms:>10.4f}   "
              f"{m.throughput_gflops:>10.2f}   {m.bandwidth_gb_s:>10.2f}")
    
    return results


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "--sweep":
        # Run full configuration sweep
        results = benchmark_sweep()
    else:
        # Run single benchmark
        result = benchmark_pytorch_sdpa()
        
        print("\n" + "="*70)
        print("Benchmark complete. Key metrics:")
        print("="*70)
        print(f"Mean latency:  {result.metrics.mean_time_ms:.4f} ms")
        print(f"Throughput:    {result.metrics.throughput_gflops:.2f} GFLOPS")
        print(f"Bandwidth:     {result.metrics.bandwidth_gb_s:.2f} GB/s")
        print(f"Std deviation: {result.metrics.std_dev_ms:.4f} ms")
        print(f"95th percentile: {result.metrics.percentile_95_ms:.4f} ms")

