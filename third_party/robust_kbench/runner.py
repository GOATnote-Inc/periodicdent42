"""
robust-kbench Runner: Execute benchmarks with statistical rigor
"""
import time
import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Optional, Callable, Any
from dataclasses import dataclass, asdict
from .config import ShapeConfig


@dataclass
class BenchmarkResult:
    """Single benchmark run result"""
    shape: ShapeConfig
    kernel_name: str
    
    # Latency statistics (ms)
    p50_latency_ms: float
    p90_latency_ms: float
    p95_latency_ms: float
    p99_latency_ms: float
    mean_latency_ms: float
    std_latency_ms: float
    
    # Throughput
    tflops: float
    
    # All raw measurements
    raw_latencies_ms: List[float]
    
    def to_dict(self) -> Dict[str, Any]:
        result = asdict(self)
        result["shape"] = self.shape.to_dict()
        return result


class BenchmarkRunner:
    """Runs statistically-rigorous kernel benchmarks"""
    
    def __init__(self, warmups: int = 20, iterations: int = 100):
        self.warmups = warmups
        self.iterations = iterations
    
    def benchmark_kernel(
        self,
        kernel_fn: Callable,
        shape: ShapeConfig,
        kernel_name: str = "custom",
    ) -> BenchmarkResult:
        """Benchmark a single kernel on one shape"""
        
        # Prepare inputs
        B, H, S, D = shape.batch_size, shape.num_heads, shape.seq_len, shape.head_dim
        
        dtype = torch.float16 if shape.dtype == "float16" else torch.bfloat16
        device = torch.device("cuda")
        
        Q = torch.randn(B, H, S, D, device=device, dtype=dtype)
        K = torch.randn(B, H, S, D, device=device, dtype=dtype)
        V = torch.randn(B, H, S, D, device=device, dtype=dtype)
        
        # Warmup
        for _ in range(self.warmups):
            _ = kernel_fn(Q, K, V, causal=shape.causal)
            torch.cuda.synchronize()
        
        # Benchmark
        latencies_ms = []
        for _ in range(self.iterations):
            start = time.perf_counter()
            _ = kernel_fn(Q, K, V, causal=shape.causal)
            torch.cuda.synchronize()
            end = time.perf_counter()
            latencies_ms.append((end - start) * 1000)
        
        # Calculate statistics
        latencies_np = np.array(latencies_ms)
        p50 = float(np.percentile(latencies_np, 50))
        p90 = float(np.percentile(latencies_np, 90))
        p95 = float(np.percentile(latencies_np, 95))
        p99 = float(np.percentile(latencies_np, 99))
        mean = float(np.mean(latencies_np))
        std = float(np.std(latencies_np))
        
        # Calculate TFLOP/s
        # Attention FLOPS: 4 * B * H * S^2 * D (QK^T + softmax + PV)
        flops = 4 * B * H * S * S * D
        tflops = (flops / (p50 * 1e-3)) / 1e12
        
        return BenchmarkResult(
            shape=shape,
            kernel_name=kernel_name,
            p50_latency_ms=p50,
            p90_latency_ms=p90,
            p95_latency_ms=p95,
            p99_latency_ms=p99,
            mean_latency_ms=mean,
            std_latency_ms=std,
            tflops=tflops,
            raw_latencies_ms=latencies_ms,
        )
    
    def benchmark_sdpa(self, shape: ShapeConfig) -> BenchmarkResult:
        """Benchmark PyTorch SDPA as reference"""
        
        def sdpa_kernel(Q, K, V, causal=False):
            return F.scaled_dot_product_attention(
                Q, K, V,
                is_causal=causal,
            )
        
        return self.benchmark_kernel(sdpa_kernel, shape, kernel_name="pytorch_sdpa")
    
    def benchmark_shapes(
        self,
        kernel_fn: Callable,
        shapes: List[ShapeConfig],
        kernel_name: str = "custom",
        baseline_fn: Optional[Callable] = None,
    ) -> List[BenchmarkResult]:
        """Benchmark kernel on multiple shapes"""
        
        results = []
        
        for i, shape in enumerate(shapes):
            print(f"[{i+1}/{len(shapes)}] Benchmarking {shape}... ", end="", flush=True)
            
            try:
                result = self.benchmark_kernel(kernel_fn, shape, kernel_name)
                results.append(result)
                print(f"✓ {result.p50_latency_ms:.3f} ms (p50)")
            except Exception as e:
                print(f"✗ {e}")
                continue
        
        return results

