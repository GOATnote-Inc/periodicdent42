#!/usr/bin/env python3
"""
Stage 2 Redux: Block Size Tuning for H100

Goal: Find optimal BLOCK_M, BLOCK_N for 110+ TFLOPS
Method: Systematic sweep of block sizes
Expected: 10-20% gain from better block sizing

Attribution:
- Triton autotuning methodology
- H100 optimization best practices (NVIDIA)
"""

import torch
import numpy as np
from flashcore.fast.attention_stage5_warpspec import (
    _attention_fwd_stage5, 
    attention_stage5
)
import triton

def benchmark_block_config(B, H, S, D, block_m, block_n, warmup=10, iters=50):
    """Benchmark a specific block configuration"""
    
    # Create inputs
    query = torch.randn(B, H, S, D, device='cuda', dtype=torch.float16)
    key = torch.randn(B, H, S, D, device='cuda', dtype=torch.float16)
    value = torch.randn(B, H, S, D, device='cuda', dtype=torch.float16)
    
    # Warmup
    for _ in range(warmup):
        _ = attention_stage5(query, key, value, is_causal=True, 
                            block_m=block_m, block_n=block_n)
    torch.cuda.synchronize()
    
    # Benchmark
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    
    times = []
    for _ in range(iters):
        start.record()
        _ = attention_stage5(query, key, value, is_causal=True,
                            block_m=block_m, block_n=block_n)
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end))
    
    times = torch.tensor(times)
    median_ms = torch.quantile(times, 0.5).item()
    
    # Compute TFLOPS
    flops = 4 * B * H * S * S * D
    tflops = flops / (median_ms / 1000) / 1e12
    
    return {
        'block_m': block_m,
        'block_n': block_n,
        'median_ms': median_ms,
        'tflops': tflops,
        'std_ms': times.std().item()
    }


def sweep_block_sizes():
    """Sweep block sizes to find optimal configuration"""
    
    # Test configuration
    B, H, S, D = 16, 16, 2048, 64
    
    print("="*80)
    print("STAGE 2 REDUX: BLOCK SIZE TUNING FOR H100")
    print("="*80)
    print(f"\nTest config: B={B}, H={H}, S={S}, D={D}")
    print(f"Baseline: 94.5 TFLOPS (block_m=64, block_n=64)")
    print(f"Target: 110+ TFLOPS")
    print()
    
    # Block size candidates (powers of 2, divisible by 16 for tensor cores)
    block_sizes = [32, 64, 128]
    
    results = []
    
    print("Testing block size combinations...")
    print("-"*80)
    
    for block_m in block_sizes:
        for block_n in block_sizes:
            # Skip if blocks are too large
            if block_m * D > 8192 or block_n * D > 8192:  # Register limit
                continue
            
            try:
                result = benchmark_block_config(B, H, S, D, block_m, block_n)
                results.append(result)
                
                status = "✅" if result['tflops'] >= 110 else "⚠️" if result['tflops'] >= 100 else "❌"
                print(f"{status} M={block_m:3d}, N={block_n:3d}: "
                      f"{result['tflops']:6.1f} TFLOPS "
                      f"({result['median_ms']:.3f} ms, std={result['std_ms']:.3f})")
                
            except Exception as e:
                print(f"❌ M={block_m:3d}, N={block_n:3d}: FAILED ({e})")
    
    print()
    print("="*80)
    print("RESULTS SUMMARY")
    print("="*80)
    
    # Sort by TFLOPS
    results.sort(key=lambda x: x['tflops'], reverse=True)
    
    print("\nTop 5 Configurations:")
    for i, r in enumerate(results[:5], 1):
        improvement = (r['tflops'] / 94.5 - 1) * 100
        status = "✅ TARGET MET" if r['tflops'] >= 110 else "⚠️ CLOSE" if r['tflops'] >= 105 else "❌ BELOW"
        print(f"{i}. M={r['block_m']:3d}, N={r['block_n']:3d}: "
              f"{r['tflops']:6.1f} TFLOPS ({improvement:+.1f}%) - {status}")
    
    # Best configuration
    best = results[0]
    print()
    print("="*80)
    print(f"BEST CONFIGURATION: M={best['block_m']}, N={best['block_n']}")
    print(f"  TFLOPS:      {best['tflops']:.1f}")
    print(f"  vs Baseline: {(best['tflops']/94.5 - 1)*100:+.1f}%")
    print(f"  Latency:     {best['median_ms']:.3f} ms")
    print(f"  Stability:   {best['std_ms']:.3f} ms std")
    print("="*80)
    
    # Check if we hit target
    if best['tflops'] >= 110:
        print("\n🎉 SUCCESS: Target of 110 TFLOPS achieved!")
        print(f"   Block size tuning gave us {(best['tflops']/94.5 - 1)*100:.1f}% improvement")
    elif best['tflops'] >= 100:
        print("\n⚠️  CLOSE: Got to {:.1f} TFLOPS (need 110)".format(best['tflops']))
        print("   Block tuning helped, but need more optimizations")
    else:
        print("\n❌ MISS: Best was {:.1f} TFLOPS (need 110)".format(best['tflops']))
        print("   Block size not the bottleneck - try other approaches")
    
    return results


if __name__ == "__main__":
    results = sweep_block_sizes()

