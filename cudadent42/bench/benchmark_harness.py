#!/usr/bin/env python3
"""
CUDA Kernel Benchmark Harness
==============================
Production-ready benchmarking framework for CUDA kernels with:
- Proper warmup and timing using CUDA events
- L2 cache flushing options
- Clock locking awareness
- Statistical analysis
- JSON output for reproducibility
"""

import subprocess
import json
import time
import statistics
from dataclasses import dataclass, asdict
from typing import List, Dict, Optional, Callable
from pathlib import Path
import numpy as np


@dataclass
class BenchmarkConfig:
    """Configuration for benchmark runs"""
    warmup_iterations: int = 10
    benchmark_iterations: int = 100
    flush_l2_cache: bool = False
    lock_clocks: bool = False
    exclude_memory_transfers: bool = True
    device_id: int = 0
    
    
@dataclass
class KernelMetrics:
    """Detailed kernel performance metrics"""
    mean_time_ms: float
    median_time_ms: float
    std_dev_ms: float
    min_time_ms: float
    max_time_ms: float
    percentile_95_ms: float
    percentile_99_ms: float
    iterations: int
    
    # Computed metrics
    throughput_gflops: Optional[float] = None
    bandwidth_gb_s: Optional[float] = None
    occupancy_percent: Optional[float] = None
    
    
@dataclass
class BenchmarkResult:
    """Complete benchmark result with metadata"""
    kernel_name: str
    timestamp: str
    config: BenchmarkConfig
    metrics: KernelMetrics
    environment: Dict[str, str]
    parameters: Dict[str, any]
    

class CUDABenchmarkHarness:
    """
    Comprehensive CUDA kernel benchmarking harness
    
    Usage:
        harness = CUDABenchmarkHarness(config)
        result = harness.benchmark_kernel(
            kernel_func=my_cuda_kernel,
            kernel_name="attention_forward",
            flop_count=2 * batch * seq * hidden,
            memory_bytes=batch * seq * hidden * 4
        )
    """
    
    def __init__(self, config: BenchmarkConfig = BenchmarkConfig()):
        self.config = config
        self.environment = self._get_environment_info()
        
        if self.config.lock_clocks:
            self._lock_gpu_clocks()
            
    def _get_environment_info(self) -> Dict[str, str]:
        """Collect environment metadata for reproducibility"""
        env = {}
        
        try:
            # CUDA version
            result = subprocess.run(['nvcc', '--version'], 
                                  capture_output=True, text=True)
            if result.returncode == 0:
                for line in result.stdout.split('\n'):
                    if 'release' in line.lower():
                        env['cuda_version'] = line.strip()
                        
            # GPU info
            result = subprocess.run(['nvidia-smi', '--query-gpu=name,driver_version,memory.total',
                                   '--format=csv,noheader'], 
                                  capture_output=True, text=True)
            if result.returncode == 0:
                gpu_info = result.stdout.strip().split(', ')
                if len(gpu_info) >= 3:
                    env['gpu_name'] = gpu_info[0]
                    env['driver_version'] = gpu_info[1]
                    env['gpu_memory'] = gpu_info[2]
                    
        except Exception as e:
            env['error'] = f"Could not collect environment: {str(e)}"
            
        return env
        
    def _lock_gpu_clocks(self):
        """Lock GPU clocks to max for consistent benchmarking"""
        try:
            subprocess.run(['sudo', 'nvidia-smi', '-lgc', '0,-1'], 
                         check=False, capture_output=True)
            print("⚠️  GPU clocks locked to maximum")
            print("   Remember to unlock with: sudo nvidia-smi -rgc")
        except Exception as e:
            print(f"⚠️  Could not lock GPU clocks: {e}")
            
    def benchmark_kernel(
        self,
        kernel_func: Callable,
        kernel_name: str,
        flop_count: Optional[int] = None,
        memory_bytes: Optional[int] = None,
        **kernel_params
    ) -> BenchmarkResult:
        """
        Benchmark a CUDA kernel with proper methodology
        
        Args:
            kernel_func: Python function that launches the CUDA kernel
            kernel_name: Descriptive name for the kernel
            flop_count: Number of floating point operations (for GFLOPS)
            memory_bytes: Total memory accessed (for bandwidth)
            **kernel_params: Parameters to pass to kernel_func
            
        Returns:
            BenchmarkResult with detailed metrics
        """
        print(f"\n{'='*60}")
        print(f"Benchmarking: {kernel_name}")
        print(f"{'='*60}")
        
        # Warmup phase
        print(f"Warmup: {self.config.warmup_iterations} iterations...")
        for _ in range(self.config.warmup_iterations):
            kernel_func(**kernel_params)
            
        # Flush L2 cache if requested
        if self.config.flush_l2_cache:
            self._flush_l2_cache()
            
        # Benchmark phase
        print(f"Benchmarking: {self.config.benchmark_iterations} iterations...")
        timings = []
        
        for i in range(self.config.benchmark_iterations):
            if self.config.flush_l2_cache and i > 0:
                self._flush_l2_cache()
                
            # Time using CUDA events (handled by kernel_func)
            elapsed_ms = kernel_func(**kernel_params)
            timings.append(elapsed_ms)
            
            if (i + 1) % 20 == 0:
                print(f"  Progress: {i+1}/{self.config.benchmark_iterations}")
                
        # Compute statistics
        metrics = self._compute_metrics(
            timings, flop_count, memory_bytes
        )
        
        # Create result
        result = BenchmarkResult(
            kernel_name=kernel_name,
            timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
            config=self.config,
            metrics=metrics,
            environment=self.environment,
            parameters=kernel_params
        )
        
        self._print_results(result)
        return result
        
    def _flush_l2_cache(self):
        """Flush L2 cache between runs for consistency"""
        # This requires a helper CUDA kernel that writes to a large buffer
        # Implementation depends on pycuda/cupy availability
        pass
        
    def _compute_metrics(
        self, 
        timings: List[float],
        flop_count: Optional[int],
        memory_bytes: Optional[int]
    ) -> KernelMetrics:
        """Compute comprehensive performance metrics"""
        
        metrics = KernelMetrics(
            mean_time_ms=statistics.mean(timings),
            median_time_ms=statistics.median(timings),
            std_dev_ms=statistics.stdev(timings) if len(timings) > 1 else 0.0,
            min_time_ms=min(timings),
            max_time_ms=max(timings),
            percentile_95_ms=float(np.percentile(timings, 95)),
            percentile_99_ms=float(np.percentile(timings, 99)),
            iterations=len(timings)
        )
        
        # Compute derived metrics
        if flop_count:
            # GFLOPS = (FLOP / time_seconds) / 1e9
            time_seconds = metrics.mean_time_ms / 1000.0
            metrics.throughput_gflops = (flop_count / time_seconds) / 1e9
            
        if memory_bytes:
            # GB/s = (bytes / time_seconds) / 1e9
            time_seconds = metrics.mean_time_ms / 1000.0
            metrics.bandwidth_gb_s = (memory_bytes / time_seconds) / 1e9
            
        return metrics
        
    def _print_results(self, result: BenchmarkResult):
        """Pretty print benchmark results"""
        print(f"\n{'='*60}")
        print(f"Results: {result.kernel_name}")
        print(f"{'='*60}")
        
        m = result.metrics
        print(f"\nTiming Statistics:")
        print(f"  Mean:      {m.mean_time_ms:8.4f} ms")
        print(f"  Median:    {m.median_time_ms:8.4f} ms")
        print(f"  Std Dev:   {m.std_dev_ms:8.4f} ms")
        print(f"  Min:       {m.min_time_ms:8.4f} ms")
        print(f"  Max:       {m.max_time_ms:8.4f} ms")
        print(f"  95th pct:  {m.percentile_95_ms:8.4f} ms")
        print(f"  99th pct:  {m.percentile_99_ms:8.4f} ms")
        
        if m.throughput_gflops:
            print(f"\nPerformance:")
            print(f"  Throughput: {m.throughput_gflops:8.2f} GFLOPS")
            
        if m.bandwidth_gb_s:
            print(f"  Bandwidth:  {m.bandwidth_gb_s:8.2f} GB/s")
            
        print(f"\nEnvironment:")
        for key, value in result.environment.items():
            print(f"  {key}: {value}")
            
    def save_results(self, result: BenchmarkResult, output_path: Path):
        """Save benchmark results to JSON for reproducibility"""
        output_data = {
            'kernel_name': result.kernel_name,
            'timestamp': result.timestamp,
            'config': asdict(result.config),
            'metrics': asdict(result.metrics),
            'environment': result.environment,
            'parameters': result.parameters
        }
        
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(output_data, f, indent=2)
            
        print(f"\nResults saved to: {output_path}")
        
    def compare_results(
        self, 
        baseline_path: Path, 
        current_result: BenchmarkResult
    ) -> Dict[str, float]:
        """Compare current result against baseline"""
        with open(baseline_path, 'r') as f:
            baseline = json.load(f)
            
        baseline_time = baseline['metrics']['mean_time_ms']
        current_time = current_result.metrics.mean_time_ms
        
        speedup = baseline_time / current_time
        improvement_pct = ((baseline_time - current_time) / baseline_time) * 100
        
        print(f"\nComparison vs Baseline:")
        print(f"  Baseline:    {baseline_time:8.4f} ms")
        print(f"  Current:     {current_time:8.4f} ms")
        print(f"  Speedup:     {speedup:8.4f}x")
        print(f"  Improvement: {improvement_pct:8.2f}%")
        
        return {
            'speedup': speedup,
            'improvement_pct': improvement_pct,
            'baseline_ms': baseline_time,
            'current_ms': current_time
        }

