#!/usr/bin/env python3
"""
Enhanced Integrated Benchmark with Publication-Grade Statistics

Integrates:
- Environment locking (env_lock)
- Bootstrap confidence intervals (stats)
- GPU memory tracking (memory_tracker)
- Statistical comparison (effect sizes, significance testing)

Author: Brandon Dent (b@thegoatnote.com)
License: Apache 2.0
"""

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Optional, List, Dict, Any
import numpy as np
import torch
import torch.nn.functional as F

# Import our enhanced modules
sys.path.insert(0, str(Path(__file__).parent))
from common.env_lock import lock_environment, write_env
from common.stats import bootstrap_ci, compare_distributions
from common.memory_tracker import MemoryTracker


def benchmark_pytorch_sdpa(
    batch: int,
    heads: int,
    seq: int,
    dim: int,
    iterations: int = 100,
    warmup: int = 20,
    lock_env: bool = True
) -> Dict[str, Any]:
    """
    Benchmark PyTorch SDPA with publication-grade statistics
    
    Args:
        batch: Batch size
        heads: Number of attention heads
        seq: Sequence length
        dim: Head dimension
        iterations: Number of measurement iterations
        warmup: Number of warmup iterations
        lock_env: Whether to lock environment for reproducibility
    
    Returns:
        Dict with comprehensive results including bootstrap CIs
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float16
    
    if lock_env:
        lock_environment()
    
    # Create inputs
    Q = torch.randn(batch, heads, seq, dim, device=device, dtype=dtype)
    K = torch.randn(batch, heads, seq, dim, device=device, dtype=dtype)
    V = torch.randn(batch, heads, seq, dim, device=device, dtype=dtype)
    
    # Warmup
    for _ in range(warmup):
        with torch.backends.cuda.sdp_kernel(enable_flash=True, enable_math=False, enable_mem_efficient=True):
            _ = F.scaled_dot_product_attention(Q, K, V)
    
    torch.cuda.synchronize()
    
    # Benchmark with memory tracking
    times = []
    with MemoryTracker() as mem_tracker:
        for _ in range(iterations):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            
            start.record()
            with torch.backends.cuda.sdp_kernel(enable_flash=True, enable_math=False, enable_mem_efficient=True):
                output = F.scaled_dot_product_attention(Q, K, V)
            end.record()
            
            torch.cuda.synchronize()
            elapsed_ms = start.elapsed_time(end)
            times.append(elapsed_ms)
    
    times_array = np.array(times)
    
    # Compute descriptive statistics
    median_ms = float(np.median(times_array))
    mean_ms = float(np.mean(times_array))
    std_ms = float(np.std(times_array, ddof=1))
    
    # Bootstrap 95% confidence interval
    ci_lower, ci_upper = bootstrap_ci(times_array, statistic=np.median, confidence=0.95, n_bootstrap=10000, seed=42)
    
    # Compute throughput (GFLOPS)
    flops = 4 * batch * heads * seq * seq * dim
    throughput_gflops = float((flops / (median_ms / 1000)) / 1e9)
    
    # Compute bandwidth (GB/s)
    bytes_transferred = 3 * batch * heads * seq * dim * 2 + batch * heads * seq * dim * 2
    bandwidth_gb_s = float((bytes_transferred / (median_ms / 1000)) / 1e9)
    
    return {
        "config": {
            "batch": batch,
            "heads": heads,
            "seq": seq,
            "dim": dim,
            "dtype": str(dtype),
            "iterations": iterations,
            "warmup": warmup
        },
        "statistics": {
            "median_ms": median_ms,
            "mean_ms": mean_ms,
            "std_ms": std_ms,
            "ci_95_lower": float(ci_lower),
            "ci_95_upper": float(ci_upper),
            "n_samples": len(times)
        },
        "performance": {
            "throughput_gflops": throughput_gflops,
            "bandwidth_gb_s": bandwidth_gb_s
        },
        "memory": {
            "peak_mb": mem_tracker.peak_mb,
            "allocated_mb": mem_tracker.allocated_mb,
            "reserved_mb": mem_tracker.reserved_mb
        },
        "raw_latencies": times  # For post-hoc analysis
    }


def main():
    parser = argparse.ArgumentParser(
        description="Enhanced integrated benchmark with publication-grade statistics"
    )
    parser.add_argument(
        "--batch",
        type=int,
        default=32,
        help="Batch size (default: 32)"
    )
    parser.add_argument(
        "--heads",
        type=int,
        default=8,
        help="Number of attention heads (default: 8)"
    )
    parser.add_argument(
        "--seq",
        type=int,
        nargs="+",
        default=[512],
        help="Sequence length(s) to test (default: 512). Can specify multiple: --seq 128 256 512"
    )
    parser.add_argument(
        "--dim",
        type=int,
        default=64,
        help="Head dimension (default: 64)"
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=100,
        help="Number of benchmark iterations (default: 100)"
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=20,
        help="Number of warmup iterations (default: 20)"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("cudadent42/bench/artifacts"),
        help="Output directory for results (default: cudadent42/bench/artifacts)"
    )
    parser.add_argument(
        "--lock-env",
        action="store_true",
        default=True,
        help="Lock environment for reproducibility (default: True)"
    )
    parser.add_argument(
        "--compare",
        action="store_true",
        help="If multiple --seq values provided, compare them statistically"
    )
    
    args = parser.parse_args()
    
    # Check CUDA
    if not torch.cuda.is_available():
        print("❌ CUDA not available")
        return 1
    
    print("=" * 70)
    print("ENHANCED INTEGRATED BENCHMARK")
    print("Publication-Grade Statistics + Environment Locking")
    print("=" * 70)
    print()
    
    print(f"🔧 GPU: {torch.cuda.get_device_name(0)}")
    print(f"📊 Config: B={args.batch}, H={args.heads}, D={args.dim}")
    print(f"📏 Sequence lengths: {args.seq}")
    print(f"⏱️  Iterations: {args.iterations} (warmup: {args.warmup})")
    if args.lock_env:
        print("🔒 Environment locking: ENABLED")
    print()
    
    # Run benchmarks for each sequence length
    results = {}
    for seq in args.seq:
        print(f"🚀 Benchmarking S={seq}...")
        
        result = benchmark_pytorch_sdpa(
            batch=args.batch,
            heads=args.heads,
            seq=seq,
            dim=args.dim,
            iterations=args.iterations,
            warmup=args.warmup,
            lock_env=args.lock_env
        )
        
        results[f"s{seq}"] = result
        
        # Print results
        stats = result["statistics"]
        perf = result["performance"]
        mem = result["memory"]
        
        print(f"✅ Complete:")
        print(f"   Median:     {stats['median_ms']:.4f} ms")
        print(f"   95% CI:     [{stats['ci_95_lower']:.4f}, {stats['ci_95_upper']:.4f}] ms")
        print(f"   Throughput: {perf['throughput_gflops']:.1f} GFLOPS")
        print(f"   Bandwidth:  {perf['bandwidth_gb_s']:.1f} GB/s")
        print(f"   Peak GPU:   {mem['peak_mb']:.2f} MB")
        print()
    
    # Statistical comparison if requested
    if args.compare and len(args.seq) == 2:
        print("=" * 70)
        print("STATISTICAL COMPARISON")
        print("=" * 70)
        print()
        
        seq_keys = [f"s{s}" for s in sorted(args.seq)]
        baseline_key = seq_keys[1]  # Larger sequence (slower)
        candidate_key = seq_keys[0]  # Smaller sequence (faster)
        
        baseline_latencies = np.array(results[baseline_key]["raw_latencies"])
        candidate_latencies = np.array(results[candidate_key]["raw_latencies"])
        
        comparison = compare_distributions(
            baseline=baseline_latencies,
            candidate=candidate_latencies,
            confidence=0.95,
            n_bootstrap=10000,
            seed=42
        )
        
        print(f"Baseline:  S={args.seq[1]} (slower)")
        print(f"Candidate: S={args.seq[0]} (faster)")
        print()
        
        print("📊 Results:")
        print(f"   Speedup:        {comparison['speedup']:.3f}×")
        print(f"   Improvement:    {comparison['improvement_pct']:.1f}%")
        print(f"   Hedges' g:      {comparison['hedges_g']:.3f} ({comparison['hedges_interpretation']})")
        print(f"   Cliff's Delta:  {comparison['cliffs_delta']:.3f} ({comparison['cliffs_interpretation']})")
        print(f"   CIs Overlap:    {comparison['cis_overlap']}")
        print(f"   Significant:    {comparison['is_significant']} (p={comparison['mann_whitney_p']:.4f})")
        print()
        
        print("📝 Publication-Ready Statement:")
        print(comparison['verdict'])
        print()
        
        # Save comparison
        comparison_path = args.output_dir / "comparison.json"
        comparison_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Convert numpy types to native Python for JSON serialization
        comparison_serializable = {
            k: float(v) if isinstance(v, (np.floating, np.integer)) else v
            for k, v in comparison.items()
        }
        
        with open(comparison_path, 'w') as f:
            json.dump(comparison_serializable, f, indent=2)
        
        print(f"💾 Comparison saved to {comparison_path}")
    
    # Save individual results
    print("=" * 70)
    print("SAVING RESULTS")
    print("=" * 70)
    print()
    
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    for name, result in results.items():
        # Add metadata
        result["timestamp"] = time.strftime("%Y-%m-%d %H:%M:%S")
        result["gpu"] = torch.cuda.get_device_name(0)
        result["pytorch_version"] = torch.__version__
        
        output_path = args.output_dir / f"enhanced_{name}.json"
        with open(output_path, 'w') as f:
            json.dump(result, f, indent=2)
        
        print(f"✓ Saved {output_path}")
    
    # Save environment fingerprint
    env_path = args.output_dir / "env.json"
    write_env(str(env_path))
    print(f"✓ Saved {env_path}")
    print()
    
    print("=" * 70)
    print("✅ BENCHMARK COMPLETE")
    print("=" * 70)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
