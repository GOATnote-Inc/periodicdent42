#!/usr/bin/env python3
"""
Integrated Correctness + Benchmark Test
========================================
Combines correctness checking and benchmarking for complete validation.

Run this NOW on GPU to see both tools working together.
"""

import torch
import sys
from pathlib import Path

# Add bench directory to path
sys.path.insert(0, str(Path(__file__).parent))

from correctness_checker import CUDACorrectnessChecker, CorrectnessConfig, ToleranceMode
from benchmark_harness import CUDABenchmarkHarness, BenchmarkConfig
from roofline_analyzer import RooflineAnalyzer


def main():
    """Run integrated correctness + benchmark test"""
    import argparse
    import json
    
    parser = argparse.ArgumentParser(description='Integrated correctness and benchmark test')
    parser.add_argument('--output', type=Path, help='Output JSON file path')
    parser.add_argument('--batch', type=int, default=32, help='Batch size')
    parser.add_argument('--heads', type=int, default=8, help='Number of heads')
    parser.add_argument('--seq', type=int, default=128, help='Sequence length')
    parser.add_argument('--dim', type=int, default=64, help='Head dimension')
    args = parser.parse_args()
    
    if not torch.cuda.is_available():
        print("CUDA not available")
        sys.exit(1)
    
    print("="*70)
    print("INTEGRATED TEST: PyTorch SDPA")
    print("="*70)
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"CUDA: {torch.version.cuda}")
    print(f"PyTorch: {torch.__version__}")
    
    # Configuration
    B, H, S, d = args.batch, args.heads, args.seq, args.dim
    dtype = torch.float16
    scale = 1.0 / (d ** 0.5)
    
    # Create test inputs
    torch.manual_seed(42)
    Q = torch.randn(B, H, S, d, dtype=dtype, device='cuda')
    K = torch.randn(B, H, S, d, dtype=dtype, device='cuda')
    V = torch.randn(B, H, S, d, dtype=dtype, device='cuda')
    
    # ========================================================================
    # PHASE 1: CORRECTNESS CHECK
    # ========================================================================
    
    print("\n" + "="*70)
    print("PHASE 1: CORRECTNESS")
    print("="*70)
    
    # Reference: PyTorch SDPA (CPU)
    Q_cpu = Q.cpu().float()
    K_cpu = K.cpu().float()
    V_cpu = V.cpu().float()
    
    with torch.no_grad():
        ref_output = torch.nn.functional.scaled_dot_product_attention(
            Q_cpu, K_cpu, V_cpu, scale=scale
        )
    
    # CUDA: PyTorch SDPA (GPU)
    with torch.no_grad():
        cuda_output = torch.nn.functional.scaled_dot_product_attention(
            Q, K, V, scale=scale
        )
    
    # Check correctness
    config = CorrectnessConfig(
        atol=1e-3,  # FP16 precision
        rtol=1e-3,
        mode=ToleranceMode.MIXED
    )
    
    checker = CUDACorrectnessChecker(config)
    result = checker.check(
        reference_output=ref_output.numpy(),
        cuda_output=cuda_output.cpu().numpy(),
        kernel_name="pytorch_sdpa_fp16"
    )
    
    if not result.passed:
        checker.print_detailed_report(
            result, 
            ref_output.numpy(), 
            cuda_output.cpu().numpy()
        )
    
    # ========================================================================
    # PHASE 2: BENCHMARK
    # ========================================================================
    
    print("\n" + "="*70)
    print("PHASE 2: BENCHMARK")
    print("="*70)
    
    def sdpa_wrapper(**kwargs):
        """CUDA event timing wrapper"""
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        
        start.record()
        with torch.no_grad():
            _ = torch.nn.functional.scaled_dot_product_attention(Q, K, V, scale=scale)
        end.record()
        torch.cuda.synchronize()
        
        return start.elapsed_time(end)
    
    # Calculate metrics
    flop_count = 4 * B * H * S * S * d
    memory_bytes = 4 * B * H * S * d * 2
    
    # Benchmark
    bench_config = BenchmarkConfig(
        warmup_iterations=50,
        benchmark_iterations=200
    )
    
    harness = CUDABenchmarkHarness(bench_config)
    
    bench_result = harness.benchmark_kernel(
        kernel_func=sdpa_wrapper,
        kernel_name="pytorch_sdpa",
        flop_count=flop_count,
        memory_bytes=memory_bytes,
        B=B, H=H, S=S, d=d
    )
    
    # Save results (default or specified path)
    if args.output:
        output_path = args.output
    else:
        output_dir = Path(__file__).parent / "out"
        output_path = output_dir / "integrated_test_result.json"
    
    harness.save_results(bench_result, output_path)
    
    # ========================================================================
    # PHASE 3: ROOFLINE ANALYSIS
    # ========================================================================
    
    print("\n" + "="*70)
    print("PHASE 3: ROOFLINE ANALYSIS")
    print("="*70)
    
    # Detect GPU
    gpu_name = torch.cuda.get_device_name(0)
    if "L4" in gpu_name:
        gpu_name = "L4"
    elif "A100" in gpu_name:
        gpu_name = "A100-SXM4-80GB"
    elif "H100" in gpu_name:
        gpu_name = "H100-SXM5-80GB"
    elif "V100" in gpu_name:
        gpu_name = "V100-SXM2-32GB"
    
    analyzer = RooflineAnalyzer(gpu_name=gpu_name, dtype="fp16")
    roofline_result = analyzer.analyze(
        flop_count=flop_count,
        memory_bytes=memory_bytes,
        time_ms=bench_result.metrics.mean_time_ms
    )
    analyzer.print_analysis(roofline_result)
    
    # ========================================================================
    # PHASE 4: SUMMARY
    # ========================================================================
    
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    
    print(f"\nCorrectness:")
    print(f"  Status:          {'PASS' if result.passed else 'FAIL'}")
    print(f"  Max Abs Error:   {result.max_abs_error:.2e}")
    print(f"  Mean Abs Error:  {result.mean_abs_error:.2e}")
    corr_str = f"{result.correlation:.6f}" if result.correlation is not None else "N/A"
    print(f"  Correlation:     {corr_str}")
    
    print(f"\nPerformance:")
    print(f"  Mean Latency:    {bench_result.metrics.mean_time_ms:.4f} ms")
    print(f"  Std Dev:         {bench_result.metrics.std_dev_ms:.4f} ms")
    print(f"  Throughput:      {bench_result.metrics.throughput_gflops:.2f} GFLOPS")
    print(f"  Bandwidth:       {bench_result.metrics.bandwidth_gb_s:.2f} GB/s")
    
    print(f"\nConfiguration:")
    print(f"  Batch Size:      {B}")
    print(f"  Num Heads:       {H}")
    print(f"  Sequence Length: {S}")
    print(f"  Head Dimension:  {d}")
    print(f"  Precision:       FP16")
    
    # Export structured JSON for CI
    if args.output:
        combined_result = {
            'correctness': {
                'passed': bool(result.passed),
                'max_abs_error': float(result.max_abs_error),
                'mean_abs_error': float(result.mean_abs_error),
                'correlation': float(result.correlation) if result.correlation is not None else None
            },
            'performance': {
                'mean_time_ms': float(bench_result.metrics.mean_time_ms),
                'std_dev_ms': float(bench_result.metrics.std_dev_ms),
                'throughput_gflops': float(bench_result.metrics.throughput_gflops) if bench_result.metrics.throughput_gflops else None,
                'bandwidth_gb_s': float(bench_result.metrics.bandwidth_gb_s) if bench_result.metrics.bandwidth_gb_s else None
            },
            'roofline': {
                'arithmetic_intensity': float(roofline_result.arithmetic_intensity),
                'bottleneck': roofline_result.bottleneck,
                'efficiency_pct': float(roofline_result.efficiency_percent),
                'recommendations': roofline_result.recommendations[:3] if len(roofline_result.recommendations) > 0 else []
            },
            'config': {
                'batch_size': B,
                'num_heads': H,
                'seq_len': S,
                'head_dim': d,
                'dtype': 'float16'
            },
            'kernel_name': bench_result.kernel_name,
            'timestamp': bench_result.timestamp
        }
        
        with open(args.output, 'w') as f:
            json.dump(combined_result, f, indent=2)
    
    if result.passed:
        print("\nResult: Both correctness and performance validated successfully.")
        return 0
    else:
        print("\nResult: Correctness check failed.")
        return 1


if __name__ == "__main__":
    sys.exit(main())

