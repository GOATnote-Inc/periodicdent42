"""Profiling utilities for BlackwellSparseK kernels."""

import time
import torch
from typing import Callable, Dict, Any, Optional


def benchmark_latency(
    kernel_fn: Callable,
    *args,
    num_warmup: int = 10,
    num_iters: int = 100,
    **kwargs
) -> Dict[str, float]:
    """
    Benchmark kernel latency with CUDA events.
    
    Args:
        kernel_fn: Kernel function to benchmark
        *args: Positional arguments to kernel
        num_warmup: Number of warmup iterations
        num_iters: Number of measurement iterations
        **kwargs: Keyword arguments to kernel
    
    Returns:
        Dictionary with timing statistics (μs)
    
    Example:
        >>> from blackwell_sparsek import attention_forward
        >>> from blackwell_sparsek.utils import benchmark_latency
        >>> 
        >>> Q = torch.randn(1, 8, 512, 64, dtype=torch.float16, device='cuda')
        >>> K = torch.randn(1, 8, 512, 64, dtype=torch.float16, device='cuda')
        >>> V = torch.randn(1, 8, 512, 64, dtype=torch.float16, device='cuda')
        >>> 
        >>> stats = benchmark_latency(attention_forward, Q, K, V)
        >>> print(f"Latency: {stats['mean_us']:.2f} μs")
    """
    # Warmup
    for _ in range(num_warmup):
        _ = kernel_fn(*args, **kwargs)
    
    torch.cuda.synchronize()
    
    # Measure with CUDA events
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    
    timings = []
    for _ in range(num_iters):
        start_event.record()
        _ = kernel_fn(*args, **kwargs)
        end_event.record()
        torch.cuda.synchronize()
        timings.append(start_event.elapsed_time(end_event) * 1000)  # Convert to μs
    
    # Statistics
    import statistics
    return {
        "mean_us": statistics.mean(timings),
        "median_us": statistics.median(timings),
        "min_us": min(timings),
        "max_us": max(timings),
        "std_us": statistics.stdev(timings) if len(timings) > 1 else 0.0,
        "num_iters": num_iters,
    }


def profile_kernel(
    kernel_fn: Callable,
    *args,
    num_runs: int = 10,
    return_output: bool = False,
    **kwargs
) -> Dict[str, Any]:
    """
    Profile kernel with detailed metrics.
    
    Args:
        kernel_fn: Kernel function to profile
        *args: Positional arguments to kernel
        num_runs: Number of profiling runs
        return_output: Whether to return kernel output
        **kwargs: Keyword arguments to kernel
    
    Returns:
        Dictionary with profiling results
    """
    # Get input shape info
    input_info = {}
    for i, arg in enumerate(args):
        if isinstance(arg, torch.Tensor):
            input_info[f"arg{i}_shape"] = tuple(arg.shape)
            input_info[f"arg{i}_dtype"] = str(arg.dtype)
            input_info[f"arg{i}_device"] = str(arg.device)
    
    # Benchmark
    timing_stats = benchmark_latency(kernel_fn, *args, num_iters=num_runs, **kwargs)
    
    # Memory info
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        memory_stats = {
            "allocated_mb": torch.cuda.memory_allocated() / 1024 / 1024,
            "reserved_mb": torch.cuda.memory_reserved() / 1024 / 1024,
            "max_allocated_mb": torch.cuda.max_memory_allocated() / 1024 / 1024,
        }
    else:
        memory_stats = {}
    
    result = {
        "input_info": input_info,
        "timing": timing_stats,
        "memory": memory_stats,
    }
    
    if return_output:
        output = kernel_fn(*args, **kwargs)
        result["output"] = output
    
    return result


def print_profile_summary(profile_result: Dict[str, Any]):
    """Pretty-print profiling results."""
    print("=" * 80)
    print("BlackwellSparseK Kernel Profile")
    print("=" * 80)
    
    # Input info
    print("\nInput Tensors:")
    for key, value in profile_result["input_info"].items():
        print(f"  {key}: {value}")
    
    # Timing
    print("\nTiming (μs):")
    timing = profile_result["timing"]
    print(f"  Mean:   {timing['mean_us']:8.2f} μs")
    print(f"  Median: {timing['median_us']:8.2f} μs")
    print(f"  Min:    {timing['min_us']:8.2f} μs")
    print(f"  Max:    {timing['max_us']:8.2f} μs")
    print(f"  Std:    {timing['std_us']:8.2f} μs")
    print(f"  Iters:  {timing['num_iters']}")
    
    # Memory
    if profile_result["memory"]:
        print("\nMemory (MB):")
        mem = profile_result["memory"]
        print(f"  Allocated:     {mem['allocated_mb']:8.2f} MB")
        print(f"  Reserved:      {mem['reserved_mb']:8.2f} MB")
        print(f"  Max Allocated: {mem['max_allocated_mb']:8.2f} MB")
    
    print("=" * 80)

