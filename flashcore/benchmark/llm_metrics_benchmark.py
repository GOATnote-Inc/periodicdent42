#!/usr/bin/env python3
"""
Real-World LLM Benchmark: FlashCore vs FA3

Measures what matters for LLM inference:
1. Tokens/second (generation throughput)
2. Sequences/second (batching efficiency)
3. Max sequence length (scaling)
4. VRAM usage (memory efficiency)

Goal: Clear picture of practical performance vs FA3
"""

import torch
import torch.nn.functional as F
from torch.backends.cuda import sdp_kernel
import time
import numpy as np
from dataclasses import dataclass
from typing import List, Dict
import sys

# Try to import FlashCore (if available)
try:
    from flashcore.fast.attention_production import attention_with_kv_cache
    FLASHCORE_AVAILABLE = True
except ImportError:
    FLASHCORE_AVAILABLE = False
    print("⚠️  FlashCore not available, comparing Triton vs SDPA only")


@dataclass
class BenchmarkConfig:
    """Configuration for LLM inference scenario"""
    batch_size: int
    num_heads: int
    head_dim: int
    seq_length: int
    name: str
    
    @property
    def total_tokens(self):
        """Total tokens processed per forward pass"""
        return self.batch_size * self.seq_length
    
    @property
    def hidden_size(self):
        return self.num_heads * self.head_dim


# Real-world LLM configurations
CONFIGS = [
    # Single sequence (latency-critical)
    BenchmarkConfig(1, 32, 128, 512, "LLaMA-7B (B=1, S=512) - Low Latency"),
    BenchmarkConfig(1, 32, 128, 2048, "LLaMA-7B (B=1, S=2K) - Medium Context"),
    BenchmarkConfig(1, 32, 128, 8192, "LLaMA-7B (B=1, S=8K) - Long Context"),
    
    # Batched inference (throughput-critical)
    BenchmarkConfig(8, 32, 128, 512, "LLaMA-7B (B=8, S=512) - Batched"),
    BenchmarkConfig(32, 32, 128, 512, "LLaMA-7B (B=32, S=512) - High Throughput"),
    
    # Larger models
    BenchmarkConfig(1, 40, 128, 2048, "LLaMA-13B (B=1, S=2K)"),
    BenchmarkConfig(8, 40, 128, 2048, "LLaMA-13B (B=8, S=2K)"),
    
    # GPT-4 class (96 heads)
    BenchmarkConfig(1, 96, 128, 2048, "GPT-4 class (B=1, S=2K)"),
    BenchmarkConfig(8, 96, 128, 2048, "GPT-4 class (B=8, S=2K)"),
]


def measure_memory_usage(func, *args, **kwargs):
    """Measure peak VRAM usage during function execution"""
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    
    _ = func(*args, **kwargs)
    torch.cuda.synchronize()
    
    peak_memory_mb = torch.cuda.max_memory_allocated() / 1024 / 1024
    return peak_memory_mb


def benchmark_kernel(
    kernel_fn,
    config: BenchmarkConfig,
    warmup: int = 20,
    iters: int = 100,
    kernel_name: str = "Unknown"
) -> Dict:
    """Benchmark a kernel with real-world LLM metrics"""
    
    B, H, S, D = config.batch_size, config.num_heads, config.seq_length, config.head_dim
    
    # Create inputs
    query = torch.randn(B, H, S, D, device='cuda', dtype=torch.float16)
    key = torch.randn(B, H, S, D, device='cuda', dtype=torch.float16)
    value = torch.randn(B, H, S, D, device='cuda', dtype=torch.float16)
    
    # Warmup
    for _ in range(warmup):
        _ = kernel_fn(query, key, value, is_causal=True)
    torch.cuda.synchronize()
    
    # Measure latency
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    
    times = []
    for _ in range(iters):
        start.record()
        _ = kernel_fn(query, key, value, is_causal=True)
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end))
    
    times = np.array(times)
    median_ms = np.median(times)
    p99_ms = np.percentile(times, 99)
    
    # Measure memory
    peak_memory_mb = measure_memory_usage(kernel_fn, query, key, value, is_causal=True)
    
    # Compute practical metrics
    tokens_per_second = (config.total_tokens / (median_ms / 1000))
    sequences_per_second = (config.batch_size / (median_ms / 1000))
    
    # Compute TFLOPS (for reference)
    flops = 4 * B * H * S * S * D
    tflops = flops / (median_ms / 1000) / 1e12
    
    return {
        'kernel': kernel_name,
        'config': config.name,
        'latency_p50_ms': median_ms,
        'latency_p99_ms': p99_ms,
        'tokens_per_sec': tokens_per_second,
        'sequences_per_sec': sequences_per_second,
        'vram_mb': peak_memory_mb,
        'tflops': tflops,
        'batch_size': B,
        'seq_length': S,
        'total_tokens': config.total_tokens,
    }


def pytorch_sdpa(query, key, value, is_causal=False):
    """Wrapper for PyTorch SDPA (FA2/FA3 backend)"""
    return F.scaled_dot_product_attention(query, key, value, is_causal=is_causal)


def triton_baseline(query, key, value, is_causal=False):
    """Wrapper for FlashCore Triton baseline"""
    if not FLASHCORE_AVAILABLE:
        return pytorch_sdpa(query, key, value, is_causal)
    
    # Call FlashCore Triton kernel
    B, H, S, D = query.shape
    return attention_with_kv_cache(
        query, key, value,
        past_key_value=None,
        is_causal=is_causal,
        update_cache=False,
        num_query_heads=H,
        num_kv_heads=H
    )[0]


def compare_kernels(config: BenchmarkConfig):
    """Compare PyTorch SDPA vs FlashCore Triton"""
    print(f"\n{'='*80}")
    print(f"CONFIG: {config.name}")
    print(f"{'='*80}")
    print(f"Batch: {config.batch_size}, Heads: {config.num_heads}, "
          f"Seq: {config.seq_length}, Dim: {config.head_dim}")
    print(f"Total tokens: {config.total_tokens:,}")
    print()
    
    results = []
    
    # Benchmark PyTorch SDPA (FA2/FA3 backend)
    print("[1/2] Benchmarking PyTorch SDPA (FA2/FA3)...")
    try:
        result_sdpa = benchmark_kernel(pytorch_sdpa, config, kernel_name="PyTorch SDPA (FA3)")
        results.append(result_sdpa)
        print(f"  ✅ Latency: {result_sdpa['latency_p50_ms']:.2f} ms")
        print(f"  ✅ Throughput: {result_sdpa['tokens_per_sec']:,.0f} tokens/sec")
        print(f"  ✅ VRAM: {result_sdpa['vram_mb']:.1f} MB")
    except Exception as e:
        print(f"  ❌ Failed: {e}")
    
    # Benchmark FlashCore Triton
    print("[2/2] Benchmarking FlashCore Triton...")
    try:
        result_triton = benchmark_kernel(triton_baseline, config, kernel_name="FlashCore Triton")
        results.append(result_triton)
        print(f"  ✅ Latency: {result_triton['latency_p50_ms']:.2f} ms")
        print(f"  ✅ Throughput: {result_triton['tokens_per_sec']:,.0f} tokens/sec")
        print(f"  ✅ VRAM: {result_triton['vram_mb']:.1f} MB")
    except Exception as e:
        print(f"  ❌ Failed: {e}")
    
    # Comparison
    if len(results) == 2:
        sdpa, triton = results
        print()
        print(f"{'='*80}")
        print("COMPARISON")
        print(f"{'='*80}")
        
        speedup = sdpa['latency_p50_ms'] / triton['latency_p50_ms']
        throughput_gain = triton['tokens_per_sec'] / sdpa['tokens_per_sec']
        memory_ratio = triton['vram_mb'] / sdpa['vram_mb']
        
        print(f"\nLatency:")
        print(f"  SDPA:     {sdpa['latency_p50_ms']:.2f} ms")
        print(f"  FlashCore: {triton['latency_p50_ms']:.2f} ms")
        print(f"  Speedup:   {speedup:.2f}× {'✅' if speedup > 1.0 else '❌'}")
        
        print(f"\nThroughput (tokens/sec):")
        print(f"  SDPA:     {sdpa['tokens_per_sec']:,.0f}")
        print(f"  FlashCore: {triton['tokens_per_sec']:,.0f}")
        print(f"  Gain:      {(throughput_gain - 1) * 100:+.1f}% {'✅' if throughput_gain > 1.0 else '❌'}")
        
        print(f"\nVRAM Usage:")
        print(f"  SDPA:     {sdpa['vram_mb']:.1f} MB")
        print(f"  FlashCore: {triton['vram_mb']:.1f} MB")
        print(f"  Ratio:     {memory_ratio:.2f}× {'✅' if memory_ratio < 1.0 else '⚠️'}")
        
        print(f"\nTFLOPS (reference):")
        print(f"  SDPA:     {sdpa['tflops']:.1f}")
        print(f"  FlashCore: {triton['tflops']:.1f}")
    
    return results


def main():
    print("="*80)
    print("FLASHCORE: REAL-WORLD LLM BENCHMARK")
    print("="*80)
    print()
    print("Measuring what matters:")
    print("  1. Tokens/second (generation throughput)")
    print("  2. Sequences/second (batching efficiency)")
    print("  3. Latency (p50, p99)")
    print("  4. VRAM usage (memory efficiency)")
    print()
    print(f"GPU: {torch.cuda.get_device_name()}")
    print(f"CUDA: {torch.version.cuda}")
    print(f"PyTorch: {torch.__version__}")
    print()
    
    # Run all configurations
    all_results = []
    for config in CONFIGS:
        results = compare_kernels(config)
        all_results.extend(results)
    
    # Summary
    print("\n" + "="*80)
    print("SUMMARY: VALUE PROPOSITION")
    print("="*80)
    print()
    
    # Group by kernel
    sdpa_results = [r for r in all_results if 'SDPA' in r['kernel']]
    triton_results = [r for r in all_results if 'Triton' in r['kernel']]
    
    if sdpa_results and triton_results:
        # Average speedup across configs
        speedups = []
        for sdpa, triton in zip(sdpa_results, triton_results):
            if sdpa['config'] == triton['config']:
                speedup = sdpa['latency_p50_ms'] / triton['latency_p50_ms']
                speedups.append(speedup)
        
        avg_speedup = np.mean(speedups)
        
        # Average throughput gain
        throughput_gains = []
        for sdpa, triton in zip(sdpa_results, triton_results):
            if sdpa['config'] == triton['config']:
                gain = triton['tokens_per_sec'] / sdpa['tokens_per_sec']
                throughput_gains.append(gain)
        
        avg_throughput_gain = np.mean(throughput_gains)
        
        # Memory efficiency
        memory_ratios = []
        for sdpa, triton in zip(sdpa_results, triton_results):
            if sdpa['config'] == triton['config']:
                ratio = triton['vram_mb'] / sdpa['vram_mb']
                memory_ratios.append(ratio)
        
        avg_memory_ratio = np.mean(memory_ratios)
        
        print(f"Average Speedup:      {avg_speedup:.2f}× {'✅' if avg_speedup > 1.0 else '❌'}")
        print(f"Throughput Gain:      {(avg_throughput_gain - 1) * 100:+.1f}%")
        print(f"Memory Efficiency:    {avg_memory_ratio:.2f}× {'✅' if avg_memory_ratio < 1.0 else '⚠️'}")
        print()
        
        if avg_speedup >= 1.1:
            print("✅ VALUE DELIVERED: FlashCore beats FA3 by 10%+")
        elif avg_speedup >= 1.0:
            print("⚠️  COMPETITIVE: FlashCore matches FA3")
        else:
            print("❌ NEEDS WORK: FlashCore slower than FA3")
            print(f"   Gap: {(1.0 - avg_speedup) * 100:.1f}% slower")
            print("   Next: Implement Phase 2 (WGMMA + TMA) for Hopper")
    
    print("="*80)


if __name__ == "__main__":
    main()

