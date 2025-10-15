#!/usr/bin/env python3
"""
Comprehensive SDPA Baseline Benchmarking
Measures p50/p90 latency and TFLOP/s for PyTorch SDPA vs custom CUDA kernel.

Output formats:
- JSON: benchmarks/l4/<date>/baseline_{sdpa,ours}.json
- CSV: benchmarks/l4/<date>/baseline_{sdpa,ours}.csv
- MD: benchmarks/l4/<date>/baseline_comparison.md
"""

import torch
import torch.nn.functional as F
import time
import json
import csv
import os
import sys
import argparse
from datetime import datetime
from typing import Dict, List, Optional, Tuple

# Add repo root to path
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, REPO_ROOT)

# ============================================================================
# Configuration
# ============================================================================

# Dynamically load our kernel
OUR_KERNEL_MODULE = os.environ.get("OUR_KERNEL_MODULE", "cudadent42.bench.fa_s512_v3")
OUR_KERNEL_FUNCTION = os.environ.get("OUR_KERNEL_FUNCTION", "flash_attention_s512_v3_forward")

try:
    module = __import__(OUR_KERNEL_MODULE, fromlist=[OUR_KERNEL_FUNCTION])
    our_kernel = getattr(module, OUR_KERNEL_FUNCTION)
    print(f"✅ Loaded kernel: {OUR_KERNEL_MODULE}.{OUR_KERNEL_FUNCTION}")
except (ImportError, AttributeError) as e:
    print(f"⚠️  Could not load kernel: {e}")
    our_kernel = None

# Check CUDA
if not torch.cuda.is_available():
    print("❌ CUDA not available")
    sys.exit(1)

# Fixed seed
SEED = 42
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

# ============================================================================
# Benchmark Configuration
# ============================================================================

DEFAULT_WARMUPS = 20
DEFAULT_ITERS = 100

# Canonical shapes (must benchmark)
CANONICAL_SHAPES = [
    {"B": 4, "H": 16, "S": 2048, "D": 128, "causal": True, "dtype": "float16", "name": "canonical_large_causal"},
    {"B": 1, "H": 8, "S": 4096, "D": 128, "causal": True, "dtype": "float16", "name": "canonical_long_seq"},
    {"B": 8, "H": 16, "S": 1024, "D": 64, "causal": False, "dtype": "float16", "name": "canonical_std_noncausal"},
]

# V3 specialized shapes (S=512, D=64)
V3_SHAPES = [
    {"B": 1, "H": 8, "S": 512, "D": 64, "causal": False, "dtype": "float16", "name": "v3_small"},
    {"B": 1, "H": 8, "S": 512, "D": 64, "causal": True, "dtype": "float16", "name": "v3_small_causal"},
    {"B": 4, "H": 16, "S": 512, "D": 64, "causal": False, "dtype": "float16", "name": "v3_medium"},
    {"B": 4, "H": 16, "S": 512, "D": 64, "causal": True, "dtype": "float16", "name": "v3_medium_causal"},
    {"B": 8, "H": 16, "S": 512, "D": 64, "causal": False, "dtype": "float16", "name": "v3_large"},
    {"B": 8, "H": 16, "S": 512, "D": 64, "causal": True, "dtype": "float16", "name": "v3_large_causal"},
]

# ============================================================================
# Helper Functions
# ============================================================================

def str_to_dtype(dtype_str: str) -> torch.dtype:
    """Convert string to torch dtype."""
    if dtype_str == "float16":
        return torch.float16
    elif dtype_str == "bfloat16":
        return torch.bfloat16
    elif dtype_str == "float32":
        return torch.float32
    else:
        raise ValueError(f"Unsupported dtype: {dtype_str}")


def compute_flops(B: int, H: int, S: int, D: int) -> int:
    """
    Compute FLOPs for attention.
    
    Attention: O = softmax(QK^T / sqrt(D)) V
    - QK^T: B * H * S * S * D (matmul)
    - Softmax: ~B * H * S * S (scale + exp + sum + divide)
    - OV: B * H * S * S * D (matmul)
    
    Total ≈ 2 * B * H * S^2 * D + B * H * S^2
        ≈ B * H * S^2 * (2D + 1)
        ≈ 4 * B * H * S^2 * D (for D=64)
    """
    return 4 * B * H * S * S * D


def benchmark_kernel(
    kernel_func,
    shape_config: Dict,
    warmups: int,
    iterations: int,
    kernel_name: str = "kernel"
) -> Optional[Dict]:
    """
    Benchmark a kernel function.
    
    Returns:
        Dict with latencies, p50, p90, TFLOP/s, or None if failed
    """
    B = shape_config["B"]
    H = shape_config["H"]
    S = shape_config["S"]
    D = shape_config["D"]
    causal = shape_config["causal"]
    dtype = str_to_dtype(shape_config["dtype"])
    name = shape_config["name"]
    
    # Check BF16 support
    if dtype == torch.bfloat16 and not torch.cuda.is_bf16_supported():
        print(f"  ⊘ BF16 not supported, skipping {name}")
        return None
    
    # Check V3 kernel constraints
    if kernel_name == "ours" and (S != 512 or D != 64):
        print(f"  ⊘ V3 kernel specialized for S=512, D=64 (got S={S}, D={D}), skipping")
        return None
    
    try:
        # Generate inputs
        Q = torch.randn(B, H, S, D, device='cuda', dtype=dtype)
        K = torch.randn(B, H, S, D, device='cuda', dtype=dtype)
        V = torch.randn(B, H, S, D, device='cuda', dtype=dtype)
        
        # Warmup
        for _ in range(warmups):
            if kernel_name == "ours":
                kernel_func(Q, K, V, is_causal=causal, config_id=1)
            else:
                kernel_func(Q, K, V, is_causal=causal)
        
        # Benchmark
        latencies = []
        for _ in range(iterations):
            torch.cuda.synchronize()
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            
            start_event.record()
            if kernel_name == "ours":
                output = kernel_func(Q, K, V, is_causal=causal, config_id=1)
            else:
                output = kernel_func(Q, K, V, is_causal=causal)
            end_event.record()
            
            torch.cuda.synchronize()
            latencies.append(start_event.elapsed_time(end_event))  # milliseconds
        
        # Compute statistics
        latencies_sorted = sorted(latencies)
        p50_idx = len(latencies) // 2
        p90_idx = int(len(latencies) * 0.9)
        
        p50_ms = latencies_sorted[p50_idx]
        p90_ms = latencies_sorted[p90_idx]
        min_ms = latencies_sorted[0]
        max_ms = latencies_sorted[-1]
        mean_ms = sum(latencies) / len(latencies)
        
        # Compute TFLOP/s
        flops = compute_flops(B, H, S, D)
        tflops_p50 = (flops / (p50_ms * 1e-3)) / 1e12  # TFLOP/s
        
        return {
            "shape": name,
            "config": shape_config,
            "latencies_ms": latencies,
            "p50_ms": p50_ms,
            "p90_ms": p90_ms,
            "min_ms": min_ms,
            "max_ms": max_ms,
            "mean_ms": mean_ms,
            "tflops_p50": tflops_p50,
            "warmups": warmups,
            "iterations": iterations,
        }
        
    except Exception as e:
        print(f"  ✗ Error benchmarking {name}: {e}")
        return None


def save_results(
    kernel_name: str,
    results: List[Dict],
    output_dir: str,
    timestamp: str
):
    """Save results to JSON, CSV, and MD formats."""
    os.makedirs(output_dir, exist_ok=True)
    
    # JSON
    json_path = os.path.join(output_dir, f"baseline_{kernel_name}.json")
    with open(json_path, 'w') as f:
        # Remove raw latencies from JSON for brevity
        results_summary = []
        for res in results:
            res_copy = res.copy()
            res_copy.pop("latencies_ms", None)
            results_summary.append(res_copy)
        json.dump(results_summary, f, indent=2)
    print(f"   JSON: {json_path}")
    
    # CSV
    csv_path = os.path.join(output_dir, f"baseline_{kernel_name}.csv")
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["name", "B", "H", "S", "D", "dtype", "causal", "p50_ms", "p90_ms", "tflops_p50"])
        for res in results:
            writer.writerow([
                res["shape"],
                res["config"]["B"],
                res["config"]["H"],
                res["config"]["S"],
                res["config"]["D"],
                res["config"]["dtype"],
                res["config"]["causal"],
                f"{res['p50_ms']:.3f}",
                f"{res['p90_ms']:.3f}",
                f"{res['tflops_p50']:.2f}",
            ])
    print(f"   CSV:  {csv_path}")
    
    # Markdown
    md_path = os.path.join(output_dir, f"baseline_{kernel_name}.md")
    with open(md_path, 'w') as f:
        f.write(f"# Baseline Benchmark Report: {kernel_name}\n\n")
        f.write(f"**Date**: {timestamp}\n")
        f.write(f"**Warmups**: {results[0]['warmups'] if results else 'N/A'}\n")
        f.write(f"**Iterations**: {results[0]['iterations'] if results else 'N/A'}\n\n")
        f.write("| Name | B | H | S | D | Dtype | Causal | p50 (ms) | p90 (ms) | TFLOP/s |\n")
        f.write("|------|---|---|---|---|-------|--------|----------|----------|---------|\n")
        for res in results:
            f.write(f"| {res['shape']} | {res['config']['B']} | {res['config']['H']} | {res['config']['S']} | {res['config']['D']} | {res['config']['dtype']} | {res['config']['causal']} | {res['p50_ms']:.3f} | {res['p90_ms']:.3f} | {res['tflops_p50']:.2f} |\n")
    print(f"   MD:   {md_path}")


def generate_comparison_report(
    sdpa_results: List[Dict],
    ours_results: List[Dict],
    output_dir: str,
    timestamp: str
):
    """Generate comparison report between SDPA and our kernel."""
    md_path = os.path.join(output_dir, "baseline_comparison.md")
    
    # Create lookup for easy comparison
    sdpa_map = {res["shape"]: res for res in sdpa_results}
    ours_map = {res["shape"]: res for res in ours_results}
    
    with open(md_path, 'w') as f:
        f.write("# Baseline Benchmark Comparison\n\n")
        f.write(f"**Date**: {timestamp}\n")
        f.write(f"**GPU**: {torch.cuda.get_device_name(0)}\n")
        f.write(f"**CUDA**: {torch.version.cuda}\n\n")
        
        f.write("## Speedup Analysis (Ours vs SDPA)\n\n")
        f.write("| Shape | SDPA p50 (ms) | Ours p50 (ms) | Speedup | SDPA p90 (ms) | Ours p90 (ms) | Status |\n")
        f.write("|-------|---------------|---------------|---------|---------------|---------------|---------|\n")
        
        for shape_name in sorted(set(sdpa_map.keys()) & set(ours_map.keys())):
            sdpa = sdpa_map[shape_name]
            ours = ours_map[shape_name]
            
            speedup = sdpa["p50_ms"] / ours["p50_ms"]
            status = "🚀" if speedup >= 1.10 else "✓" if speedup >= 1.0 else "🐢"
            
            f.write(f"| {shape_name} | {sdpa['p50_ms']:.3f} | {ours['p50_ms']:.3f} | {speedup:.3f}× | {sdpa['p90_ms']:.3f} | {ours['p90_ms']:.3f} | {status} |\n")
        
        f.write("\n## Legend\n\n")
        f.write("- 🚀: Speedup ≥ 1.10× (10%+ faster)\n")
        f.write("- ✓: Speedup ≥ 1.00× (faster or equal)\n")
        f.write("- 🐢: Speedup < 1.00× (slower)\n")
    
    print(f"\n✅ Comparison report: {md_path}")


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Comprehensive SDPA baseline benchmarking")
    parser.add_argument("--warmups", type=int, default=DEFAULT_WARMUPS, help="Number of warmup iterations")
    parser.add_argument("--iters", type=int, default=DEFAULT_ITERS, help="Number of benchmark iterations")
    parser.add_argument("--shapes", choices=["canonical", "v3", "all"], default="v3",
                        help="Which shape set to benchmark")
    args = parser.parse_args()
    
    # Determine shapes to test
    if args.shapes == "canonical":
        shapes = CANONICAL_SHAPES
    elif args.shapes == "v3":
        shapes = V3_SHAPES
    else:  # all
        shapes = CANONICAL_SHAPES + V3_SHAPES
    
    # Create output directory
    date_str = datetime.now().strftime("%Y-%m-%d")
    output_dir = os.path.join(REPO_ROOT, "benchmarks", "l4", date_str)
    os.makedirs(output_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    print("=" * 80)
    print("Comprehensive SDPA Baseline Benchmarking")
    print("=" * 80)
    print(f"Date: {timestamp}")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"CUDA: {torch.version.cuda}")
    print(f"Compute Cap: {torch.cuda.get_device_capability(0)}")
    print(f"Warmups: {args.warmups}, Iterations: {args.iters}")
    print(f"Shapes: {args.shapes} ({len(shapes)} configurations)")
    print(f"Output: {output_dir}")
    print("=" * 80)
    
    # Benchmark SDPA
    print("\n" + "=" * 80)
    print("Benchmarking PyTorch SDPA")
    print("=" * 80)
    sdpa_results = []
    for i, shape in enumerate(shapes):
        print(f"[{i+1}/{len(shapes)}] {shape['name']}...", end=" ", flush=True)
        result = benchmark_kernel(F.scaled_dot_product_attention, shape, args.warmups, args.iters, "sdpa")
        if result:
            print(f"✓ {result['p50_ms']:.3f} ms (p50), {result['tflops_p50']:.2f} TFLOP/s")
            sdpa_results.append(result)
        else:
            print("✗ Failed or skipped")
    
    if sdpa_results:
        save_results("sdpa", sdpa_results, output_dir, timestamp)
    
    # Benchmark our kernel
    if our_kernel:
        print("\n" + "=" * 80)
        print("Benchmarking Our Kernel")
        print("=" * 80)
        ours_results = []
        for i, shape in enumerate(shapes):
            print(f"[{i+1}/{len(shapes)}] {shape['name']}...", end=" ", flush=True)
            result = benchmark_kernel(our_kernel, shape, args.warmups, args.iters, "ours")
            if result:
                print(f"✓ {result['p50_ms']:.3f} ms (p50), {result['tflops_p50']:.2f} TFLOP/s")
                ours_results.append(result)
            else:
                print("✗ Failed or skipped")
        
        if ours_results:
            save_results("ours", ours_results, output_dir, timestamp)
            
            # Generate comparison
            if sdpa_results:
                generate_comparison_report(sdpa_results, ours_results, output_dir, timestamp)
    else:
        print("\n⚠️  Our kernel not available, skipping")
    
    print("\n" + "=" * 80)
    print("✅ Phase 2: Baseline Benchmarking Complete")
    print("=" * 80)


if __name__ == "__main__":
    main()

