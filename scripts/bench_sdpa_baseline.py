#!/usr/bin/env python3
"""
SDPA Baseline Benchmark - Phase 2: Performance Baselines
Benchmarks PyTorch SDPA and our kernel with statistical rigor.

Usage:
    python scripts/bench_sdpa_baseline.py --shapes canonical
    python scripts/bench_sdpa_baseline.py --shapes all --output benchmarks/l4/2025-10-14/
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "cudadent42"))

import argparse
import json
import time
from datetime import datetime
from typing import List, Dict, Any

import torch
import torch.nn.functional as F
import numpy as np

# Import robust-kbench
from third_party.robust_kbench import BenchmarkRunner, ShapeConfig, BenchmarkReporter, RBKConfig

# Import our kernel
try:
    from bench.fa_inverted_prod import flash_attention_inverted_forward as our_kernel
    KERNEL_AVAILABLE = True
except ImportError:
    KERNEL_AVAILABLE = False
    print("⚠️  Warning: Production kernel not available, will only benchmark SDPA")


def sdpa_kernel(Q, K, V, causal=False):
    """PyTorch SDPA wrapper for benchmarking"""
    return F.scaled_dot_product_attention(Q, K, V, is_causal=causal)


def our_kernel_wrapper(Q, K, V, causal=False):
    """Our kernel wrapper for benchmarking"""
    return our_kernel(Q, K, V, is_causal=causal)


def main():
    parser = argparse.ArgumentParser(description="SDPA Baseline Benchmark")
    parser.add_argument(
        "--shapes",
        type=str,
        default="canonical",
        choices=["canonical", "all", "stress"],
        help="Shape grid to benchmark",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output directory (default: benchmarks/l4/{today}/)",
    )
    parser.add_argument(
        "--warmups",
        type=int,
        default=20,
        help="Number of warmup iterations",
    )
    parser.add_argument(
        "--iters",
        type=int,
        default=100,
        help="Number of benchmark iterations",
    )
    args = parser.parse_args()
    
    # Verify CUDA
    if not torch.cuda.is_available():
        print("❌ CUDA not available")
        return 1
    
    gpu_name = torch.cuda.get_device_name(0)
    print("=" * 70)
    print(f"SDPA Baseline Benchmark - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)
    print(f"GPU: {gpu_name}")
    print(f"Shapes: {args.shapes}")
    print(f"Warmups: {args.warmups}, Iterations: {args.iters}")
    print()
    
    # Setup output directory
    if args.output is None:
        today = datetime.now().strftime("%Y-%m-%d")
        output_dir = Path(f"benchmarks/l4/{today}")
    else:
        output_dir = Path(args.output)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output: {output_dir}")
    print()
    
    # Create shape config
    if args.shapes == "canonical":
        config = RBKConfig.canonical_shapes()
    elif args.shapes == "stress":
        config = RBKConfig(shapes=[
            ShapeConfig("stress_4k", 1, 8, 4096, 128, True, "float16"),
            ShapeConfig("stress_8k", 1, 8, 8192, 128, True, "float16"),
        ])
    else:
        config = RBKConfig.default_l4_grid()
    
    config.warmups = args.warmups
    config.iterations = args.iters
    config.output_dir = str(output_dir)
    
    # Create runner and reporter
    runner = BenchmarkRunner(warmups=args.warmups, iterations=args.iters)
    reporter = BenchmarkReporter(output_dir)
    
    # Benchmark SDPA
    print("Benchmarking PyTorch SDPA...")
    print("-" * 70)
    sdpa_results = runner.benchmark_shapes(
        sdpa_kernel,
        config.shapes,
        kernel_name="pytorch_sdpa",
    )
    
    # Save SDPA results
    sdpa_json = reporter.save_json(sdpa_results, "baseline_sdpa.json")
    sdpa_csv = reporter.save_csv(sdpa_results, "baseline_sdpa.csv")
    sdpa_md = reporter.save_markdown(sdpa_results, "baseline_sdpa.md")
    
    print()
    print(f"✅ SDPA results saved:")
    print(f"   JSON: {sdpa_json}")
    print(f"   CSV:  {sdpa_csv}")
    print(f"   MD:   {sdpa_md}")
    print()
    
    # Benchmark our kernel if available
    if KERNEL_AVAILABLE:
        print("Benchmarking our kernel...")
        print("-" * 70)
        our_results = runner.benchmark_shapes(
            our_kernel_wrapper,
            config.shapes,
            kernel_name="fa_inverted_prod",
        )
        
        # Save our results
        our_json = reporter.save_json(our_results, "baseline_ours.json")
        our_csv = reporter.save_csv(our_results, "baseline_ours.csv")
        our_md = reporter.save_markdown(our_results, "baseline_ours.md")
        
        print()
        print(f"✅ Our kernel results saved:")
        print(f"   JSON: {our_json}")
        print(f"   CSV:  {our_csv}")
        print(f"   MD:   {our_md}")
        print()
        
        # Generate comparison
        print("=" * 70)
        print("Speedup Analysis (Ours vs SDPA)")
        print("=" * 70)
        
        for ours, sdpa in zip(our_results, sdpa_results):
            speedup = sdpa.p50_latency_ms / ours.p50_latency_ms
            symbol = "🚀" if speedup > 1.0 else "🐢"
            
            print(f"{symbol} {ours.shape}: {speedup:.3f}× "
                  f"({ours.p50_latency_ms:.3f} ms vs {sdpa.p50_latency_ms:.3f} ms)")
        
        # Save comparison report
        comparison = []
        for ours, sdpa in zip(our_results, sdpa_results):
            comparison.append({
                "shape": str(ours.shape),
                "sdpa_p50_ms": sdpa.p50_latency_ms,
                "ours_p50_ms": ours.p50_latency_ms,
                "speedup": sdpa.p50_latency_ms / ours.p50_latency_ms,
                "sdpa_tflops": sdpa.tflops,
                "ours_tflops": ours.tflops,
            })
        
        comparison_file = output_dir / "comparison.json"
        with open(comparison_file, 'w') as f:
            json.dump(comparison, f, indent=2)
        
        print()
        print(f"✅ Comparison saved: {comparison_file}")
    
    print()
    print("=" * 70)
    print("✅ Phase 2 Baselines Complete")
    print("=" * 70)
    print()
    print("Next: Phase 3 - Wire robust-kbench for optimization loop")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

