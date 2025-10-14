#!/usr/bin/env python3
"""
Comprehensive Baseline Characterization: PyTorch SDPA

Thorough scientific characterization of PyTorch SDPA (FlashAttention-2)
baseline performance for publication-quality results.

This establishes the reference for all future custom kernel comparisons.

Author: GOATnote Autonomous Research Lab Initiative
Date: 2025-10-14
"""

import sys
import time
import json
from pathlib import Path
from typing import Dict, List
import numpy as np
import torch
import torch.nn.functional as F

# Add project to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from cudadent42.bench.common.env_lock import lock_environment
from cudadent42.bench.common.stats import bootstrap_ci
from cudadent42.bench.common.memory_tracker import MemoryTracker


def benchmark_sdpa_config(
    batch: int,
    heads: int,
    seq: int,
    dim: int,
    backend: str = 'auto',
    iterations: int = 100,
    warmup: int = 20
) -> Dict:
    """
    Benchmark single SDPA configuration
    
    Returns dict with statistics and metadata
    """
    device = torch.device("cuda")
    dtype = torch.float16
    
    # Create inputs
    Q = torch.randn(batch, heads, seq, dim, device=device, dtype=dtype)
    K = torch.randn(batch, heads, seq, dim, device=device, dtype=dtype)
    V = torch.randn(batch, heads, seq, dim, device=device, dtype=dtype)
    
    # Warmup
    with torch.backends.cuda.sdp_kernel(
        enable_flash=(backend in ['auto', 'flash']),
        enable_math=False,
        enable_mem_efficient=(backend in ['auto', 'memory_efficient'])
    ):
        for _ in range(warmup):
            _ = F.scaled_dot_product_attention(Q, K, V)
    
    torch.cuda.synchronize()
    
    # Benchmark
    latencies = []
    with MemoryTracker() as mem_tracker:
        with torch.backends.cuda.sdp_kernel(
            enable_flash=(backend in ['auto', 'flash']),
            enable_math=False,
            enable_mem_efficient=(backend in ['auto', 'memory_efficient'])
        ):
            for _ in range(iterations):
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                
                start.record()
                output = F.scaled_dot_product_attention(Q, K, V)
                end.record()
                
                torch.cuda.synchronize()
                elapsed_ms = start.elapsed_time(end)
                latencies.append(elapsed_ms)
    
    latencies = np.array(latencies)
    
    # Statistics
    median_ms = np.median(latencies)
    mean_ms = np.mean(latencies)
    std_ms = np.std(latencies, ddof=1)
    ci_lower, ci_upper = bootstrap_ci(latencies, statistic=np.median, 
                                     confidence=0.95, n_bootstrap=10000, seed=42)
    
    # Memory stats
    mem_stats = mem_tracker.get_stats()
    
    # Calculate throughput
    # FLOPs for attention: 4 * B * H * S^2 * D (Q@K^T + softmax + attention@V)
    flops = 4 * batch * heads * seq * seq * dim
    throughput_gflops = flops / (median_ms * 1e-3) / 1e9
    
    # Memory bandwidth (approximate)
    # Read Q, K, V, write O: (3 + 1) * B * H * S * D * 2 bytes (FP16)
    bytes_transferred = 4 * batch * heads * seq * dim * 2
    bandwidth_gb_s = bytes_transferred / (median_ms * 1e-3) / 1e9
    
    return {
        'config': {
            'batch': batch,
            'heads': heads,
            'seq': seq,
            'dim': dim,
            'backend': backend,
            'dtype': 'float16',
            'iterations': iterations,
            'warmup': warmup
        },
        'statistics': {
            'median_ms': float(median_ms),
            'mean_ms': float(mean_ms),
            'std_ms': float(std_ms),
            'ci_95_lower': float(ci_lower),
            'ci_95_upper': float(ci_upper),
            'min_ms': float(np.min(latencies)),
            'max_ms': float(np.max(latencies)),
            'n_samples': len(latencies)
        },
        'performance': {
            'throughput_gflops': float(throughput_gflops),
            'bandwidth_gb_s': float(bandwidth_gb_s)
        },
        'memory': {
            'peak_mb': float(mem_stats.peak_mb),
            'allocated_mb': float(mem_stats.allocated_mb),
            'reserved_mb': float(mem_stats.reserved_mb)
        },
        'raw_latencies': latencies.tolist()
    }


def run_comprehensive_baseline(
    output_dir: str = "cudadent42/bench/artifacts/baseline_comprehensive"
) -> Dict:
    """
    Run comprehensive baseline characterization
    
    Tests multiple configurations to understand PyTorch SDPA performance
    characteristics on L4.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*70)
    print("COMPREHENSIVE BASELINE CHARACTERIZATION")
    print("PyTorch SDPA (FlashAttention-2) on NVIDIA L4")
    print("="*70)
    print()
    
    # Lock environment
    lock_environment()
    assert torch.backends.cuda.matmul.allow_tf32 == False, "TF32 not disabled!"
    
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"CUDA: {torch.version.cuda}")
    print(f"PyTorch: {torch.__version__}")
    print(f"Environment: FP16, TF32 off, deterministic")
    print()
    
    # Test configurations
    configs = [
        # Fixed B=32, H=8, D=64, vary S
        {'batch': 32, 'heads': 8, 'seq': 128, 'dim': 64},
        {'batch': 32, 'heads': 8, 'seq': 256, 'dim': 64},
        {'batch': 32, 'heads': 8, 'seq': 512, 'dim': 64},  # Primary target
        {'batch': 32, 'heads': 8, 'seq': 1024, 'dim': 64},
        {'batch': 32, 'heads': 8, 'seq': 2048, 'dim': 64},
        
        # Vary batch size at S=512
        {'batch': 8, 'heads': 8, 'seq': 512, 'dim': 64},
        {'batch': 16, 'heads': 8, 'seq': 512, 'dim': 64},
        {'batch': 64, 'heads': 8, 'seq': 512, 'dim': 64},
        
        # Vary heads at S=512
        {'batch': 32, 'heads': 4, 'seq': 512, 'dim': 64},
        {'batch': 32, 'heads': 16, 'seq': 512, 'dim': 64},
    ]
    
    results = []
    total_configs = len(configs)
    
    start_time = time.time()
    
    for i, config in enumerate(configs, 1):
        print(f"[{i}/{total_configs}] Testing B={config['batch']}, H={config['heads']}, "
              f"S={config['seq']}, D={config['dim']}...")
        
        try:
            result = benchmark_sdpa_config(**config, backend='auto', iterations=100, warmup=20)
            results.append(result)
            
            median_ms = result['statistics']['median_ms']
            ci = result['statistics']
            throughput = result['performance']['throughput_gflops']
            bandwidth = result['performance']['bandwidth_gb_s']
            
            print(f"  ✅ Median: {median_ms:.4f} ms "
                  f"(95% CI: [{ci['ci_95_lower']:.4f}, {ci['ci_95_upper']:.4f}])")
            print(f"     Throughput: {throughput:.1f} GFLOPS, "
                  f"Bandwidth: {bandwidth:.1f} GB/s")
            print()
            
            # Save individual result
            result_file = output_dir / f"config_{i:02d}.json"
            with open(result_file, 'w') as f:
                json.dump(result, f, indent=2)
        
        except Exception as e:
            print(f"  ❌ FAILED: {e}")
            print()
    
    elapsed_time = time.time() - start_time
    
    # Summary
    print("="*70)
    print("BASELINE CHARACTERIZATION COMPLETE")
    print("="*70)
    print(f"Configurations tested: {len(results)}/{total_configs}")
    print(f"Time elapsed: {elapsed_time/60:.1f} minutes")
    print()
    
    # Find best/worst
    if results:
        results_sorted = sorted(results, key=lambda x: x['statistics']['median_ms'])
        best = results_sorted[0]
        worst = results_sorted[-1]
        
        print("Fastest configuration:")
        print(f"  B={best['config']['batch']}, H={best['config']['heads']}, "
              f"S={best['config']['seq']}, D={best['config']['dim']}")
        print(f"  Median: {best['statistics']['median_ms']:.4f} ms")
        print()
        
        print("Slowest configuration:")
        print(f"  B={worst['config']['batch']}, H={worst['config']['heads']}, "
              f"S={worst['config']['seq']}, D={worst['config']['dim']}")
        print(f"  Median: {worst['statistics']['median_ms']:.4f} ms")
        print()
        
        speedup = worst['statistics']['median_ms'] / best['statistics']['median_ms']
        print(f"Speedup (best vs worst): {speedup:.2f}×")
        print()
    
    # Save combined results
    summary = {
        'metadata': {
            'date': time.strftime('%Y-%m-%d %H:%M:%S'),
            'gpu': torch.cuda.get_device_name(0),
            'cuda_version': torch.version.cuda,
            'pytorch_version': torch.__version__,
            'environment': 'FP16, TF32 disabled, deterministic',
            'total_time_minutes': elapsed_time / 60,
            'configs_tested': len(results),
            'configs_total': total_configs
        },
        'results': results
    }
    
    summary_file = output_dir / "summary.json"
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"✅ Results saved to: {output_dir}")
    print()
    
    return summary


def main():
    """Run comprehensive baseline characterization"""
    try:
        summary = run_comprehensive_baseline()
        
        print("="*70)
        print("✅ BASELINE CHARACTERIZATION SUCCESS")
        print("="*70)
        print()
        print("Next steps:")
        print("  1. Review results in cudadent42/bench/artifacts/baseline_comprehensive/")
        print("  2. Generate publication report")
        print("  3. Use as reference for custom kernel comparisons")
        print()
        
        return 0
    
    except Exception as e:
        print(f"❌ FAILED: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())

