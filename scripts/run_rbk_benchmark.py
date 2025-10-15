#!/usr/bin/env python3
"""
robust-kbench Runner - Phase 3: Micro-benchmarking Integration
Runs comprehensive benchmarks using robust-kbench framework.

Usage:
    python scripts/run_rbk_benchmark.py --config rbk_config.yaml
    python scripts/run_rbk_benchmark.py --config rbk_config.yaml --kernels sdpa,v3
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "cudadent42"))

import argparse
import json
from datetime import datetime
from typing import List, Dict, Any

import torch
import torch.nn.functional as F

# Import robust-kbench
from third_party.robust_kbench import BenchmarkRunner, RBKConfig, BenchmarkReporter

# Import our kernels
KERNELS_AVAILABLE = {}

try:
    from bench.fa_inverted_prod import flash_attention_inverted_forward
    KERNELS_AVAILABLE["fa_inverted_prod"] = flash_attention_inverted_forward
except ImportError:
    print("⚠️  Warning: fa_inverted_prod not available")

try:
    from bench.fa_s512_v3 import flash_attention_s512_v3_forward
    KERNELS_AVAILABLE["v3"] = flash_attention_s512_v3_forward
except ImportError:
    print("⚠️  Warning: fa_s512_v3 not available")

try:
    from bench.fa_inverted_v2_tensor_cores import flash_attention_inverted_forward as fa_v2
    KERNELS_AVAILABLE["v2"] = fa_v2
except ImportError:
    print("⚠️  Warning: fa_inverted_v2 not available")


def sdpa_kernel(Q, K, V, causal=False):
    """PyTorch SDPA wrapper"""
    return F.scaled_dot_product_attention(Q, K, V, is_causal=causal)


def create_kernel_wrapper(kernel_fn, name: str):
    """Create a wrapper that handles different kernel APIs"""
    def wrapper(Q, K, V, causal=False):
        if name == "v3":
            # V3 kernel uses different API
            return kernel_fn(Q, K, V, is_causal=causal, config_id=1)
        else:
            return kernel_fn(Q, K, V, is_causal=causal)
    return wrapper


def main():
    parser = argparse.ArgumentParser(description="robust-kbench Runner")
    parser.add_argument(
        "--config",
        type=Path,
        default="rbk_config.yaml",
        help="Path to RBK config YAML",
    )
    parser.add_argument(
        "--kernels",
        type=str,
        default="sdpa,all",
        help="Comma-separated list of kernels to benchmark (sdpa,v2,v3,fa_inverted_prod,all)",
    )
    parser.add_argument(
        "--output-suffix",
        type=str,
        default="",
        help="Suffix for output files (e.g., '_fix_a')",
    )
    args = parser.parse_args()
    
    # Verify CUDA
    if not torch.cuda.is_available():
        print("❌ CUDA not available")
        return 1
    
    gpu_name = torch.cuda.get_device_name(0)
    compute_cap = torch.cuda.get_device_capability(0)
    
    print("=" * 80)
    print(f"robust-kbench Runner - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)
    print(f"GPU: {gpu_name} (Compute {compute_cap[0]}.{compute_cap[1]})")
    print(f"Config: {args.config}")
    print()
    
    # Load config
    if not args.config.exists():
        print(f"❌ Config file not found: {args.config}")
        return 1
    
    config = RBKConfig.from_yaml(args.config)
    print(f"Shapes: {len(config.shapes)}")
    print(f"Warmups: {config.warmups}, Iterations: {config.iterations}")
    print(f"Output: {config.output_dir}")
    print()
    
    # Parse kernel list
    kernel_list = args.kernels.lower().split(',')
    if "all" in kernel_list:
        kernel_list = ["sdpa"] + list(KERNELS_AVAILABLE.keys())
    
    # Create runner and reporter
    runner = BenchmarkRunner(warmups=config.warmups, iterations=config.iterations)
    reporter = BenchmarkReporter(Path(config.output_dir))
    
    all_results = {}
    
    # Benchmark each kernel
    for kernel_name in kernel_list:
        print(f"\n{'='*80}")
        print(f"Benchmarking: {kernel_name}")
        print(f"{'='*80}")
        
        if kernel_name == "sdpa":
            kernel_fn = sdpa_kernel
        elif kernel_name in KERNELS_AVAILABLE:
            kernel_fn = create_kernel_wrapper(KERNELS_AVAILABLE[kernel_name], kernel_name)
        else:
            print(f"⚠️  Kernel '{kernel_name}' not available, skipping")
            continue
        
        try:
            results = runner.benchmark_shapes(
                kernel_fn,
                config.shapes,
                kernel_name=kernel_name,
            )
            
            # Save results
            suffix = args.output_suffix
            json_file = reporter.save_json(results, f"rbk_{kernel_name}{suffix}.json")
            csv_file = reporter.save_csv(results, f"rbk_{kernel_name}{suffix}.csv")
            md_file = reporter.save_markdown(results, f"rbk_{kernel_name}{suffix}.md")
            
            print()
            print(f"✅ Results saved:")
            print(f"   JSON: {json_file}")
            print(f"   CSV:  {csv_file}")
            print(f"   MD:   {md_file}")
            
            all_results[kernel_name] = results
            
        except Exception as e:
            print(f"❌ Benchmarking failed: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # Generate comparison report if multiple kernels
    if len(all_results) > 1 and "sdpa" in all_results:
        print()
        print("=" * 80)
        print("Speedup Analysis (vs SDPA)")
        print("=" * 80)
        
        sdpa_results = all_results["sdpa"]
        comparison_data = []
        
        for kernel_name, results in all_results.items():
            if kernel_name == "sdpa":
                continue
            
            print(f"\n{kernel_name}:")
            print("-" * 40)
            
            for ours, sdpa in zip(results, sdpa_results):
                speedup = sdpa.p50_latency_ms / ours.p50_latency_ms
                symbol = "🚀" if speedup >= 1.0 else "🐢"
                
                print(f"{symbol} {ours.shape.name:25s}: {speedup:6.3f}× "
                      f"({ours.p50_latency_ms:7.3f} ms vs {sdpa.p50_latency_ms:7.3f} ms)")
                
                comparison_data.append({
                    "kernel": kernel_name,
                    "shape": ours.shape.name,
                    "ours_p50_ms": ours.p50_latency_ms,
                    "ours_p90_ms": ours.p90_latency_ms,
                    "sdpa_p50_ms": sdpa.p50_latency_ms,
                    "sdpa_p90_ms": sdpa.p90_latency_ms,
                    "speedup_p50": speedup,
                    "speedup_p90": sdpa.p90_latency_ms / ours.p90_latency_ms,
                    "ours_tflops": ours.tflops,
                    "sdpa_tflops": sdpa.tflops,
                })
        
        # Save comparison
        comparison_file = Path(config.output_dir) / f"comparison{args.output_suffix}.json"
        with open(comparison_file, 'w') as f:
            json.dump(comparison_data, f, indent=2)
        
        print()
        print(f"✅ Comparison saved: {comparison_file}")
        
        # Summary statistics
        print()
        print("=" * 80)
        print("Summary Statistics")
        print("=" * 80)
        
        for kernel_name in all_results.keys():
            if kernel_name == "sdpa":
                continue
            
            kernel_comparisons = [c for c in comparison_data if c["kernel"] == kernel_name]
            if kernel_comparisons:
                speedups = [c["speedup_p50"] for c in kernel_comparisons]
                mean_speedup = sum(speedups) / len(speedups)
                min_speedup = min(speedups)
                max_speedup = max(speedups)
                
                wins = sum(1 for s in speedups if s >= 1.0)
                losses = len(speedups) - wins
                
                print(f"\n{kernel_name}:")
                print(f"  Mean speedup: {mean_speedup:.3f}×")
                print(f"  Range: {min_speedup:.3f}× to {max_speedup:.3f}×")
                print(f"  Wins: {wins}/{len(speedups)} shapes")
                print(f"  Losses: {losses}/{len(speedups)} shapes")
    
    print()
    print("=" * 80)
    print("✅ Phase 3: robust-kbench Integration Complete")
    print("=" * 80)
    print()
    print("Next: Phase 4 - EvoEngineer guided optimization loop")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

