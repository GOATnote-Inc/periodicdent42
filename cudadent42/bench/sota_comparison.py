#!/usr/bin/env python3
"""
SOTA Benchmark Comparison - Publication-Grade Artifact

Compares PyTorch SDPA (optimized config from Option B) against:
- flash-attn (Dao-AILab's reference implementation)
- xFormers (Meta's memory-efficient attention)
- PyTorch SDPA (baseline reference)

Generates publication-ready tables, statistical analysis, and reproducibility guide.

Author: Brandon Dent (b@thegoatnote.com)
License: Apache 2.0
"""

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, asdict
import torch
import torch.nn.functional as F
import numpy as np


@dataclass
class BenchmarkConfig:
    """Single benchmark configuration"""
    name: str
    batch: int
    heads: int
    seq: int
    dim: int
    dtype: torch.dtype
    causal: bool = False


@dataclass
class BenchmarkResult:
    """Result for one implementation on one config"""
    implementation: str
    config_name: str
    median_ms: float
    mean_ms: float
    std_ms: float
    ci_95_low: float
    ci_95_high: float
    throughput_gflops: float
    bandwidth_gb_s: float
    memory_mb: float
    iterations: int
    error: Optional[str] = None


class SOTABenchmark:
    """
    SOTA comparison benchmark suite
    
    Tests multiple implementations across representative configs.
    Generates publication-grade statistics and reproducibility guide.
    """
    
    def __init__(
        self,
        iterations: int = 100,
        warmup: int = 20,
        output_dir: Path = Path("artifacts")
    ):
        self.iterations = iterations
        self.warmup = warmup
        self.output_dir = output_dir
        self.results: List[BenchmarkResult] = []
        
        # Check CUDA
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA not available")
        
        self.device = torch.device("cuda")
        self.gpu_name = torch.cuda.get_device_name(0)
        self.gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
        
        print(f"🔧 GPU: {self.gpu_name}")
        print(f"💾 Memory: {self.gpu_memory_gb:.1f} GB")
        print(f"⏱️  Iterations: {self.iterations} (warmup: {self.warmup})")
        print()
        
        # Check for optional dependencies
        self.has_flash_attn = self._check_flash_attn()
        self.has_xformers = self._check_xformers()
        
        print(f"📦 Available implementations:")
        print(f"  ✅ PyTorch SDPA (built-in)")
        print(f"  {'✅' if self.has_flash_attn else '❌'} flash-attn")
        print(f"  {'✅' if self.has_xformers else '❌'} xFormers")
        print()
    
    def _check_flash_attn(self) -> bool:
        """Check if flash-attn is available"""
        try:
            import flash_attn
            return True
        except ImportError:
            return False
    
    def _check_xformers(self) -> bool:
        """Check if xFormers is available"""
        try:
            import xformers
            import xformers.ops
            return True
        except ImportError:
            return False
    
    def _benchmark_pytorch_sdpa(
        self,
        config: BenchmarkConfig,
        backend: str = "auto"
    ) -> Optional[BenchmarkResult]:
        """Benchmark PyTorch SDPA"""
        try:
            # Create inputs
            Q = torch.randn(config.batch, config.heads, config.seq, config.dim,
                          device=self.device, dtype=config.dtype)
            K = torch.randn(config.batch, config.heads, config.seq, config.dim,
                          device=self.device, dtype=config.dtype)
            V = torch.randn(config.batch, config.heads, config.seq, config.dim,
                          device=self.device, dtype=config.dtype)
            
            # Configure backend
            if backend == "flash":
                enable_flash, enable_math, enable_mem_efficient = True, False, False
            elif backend == "memory_efficient":
                enable_flash, enable_math, enable_mem_efficient = False, False, True
            elif backend == "math":
                enable_flash, enable_math, enable_mem_efficient = False, True, False
            else:  # auto
                enable_flash, enable_math, enable_mem_efficient = True, False, True
            
            # Warmup
            with torch.backends.cuda.sdp_kernel(
                enable_flash=enable_flash,
                enable_math=enable_math,
                enable_mem_efficient=enable_mem_efficient
            ):
                for _ in range(self.warmup):
                    _ = F.scaled_dot_product_attention(Q, K, V, is_causal=config.causal)
            
            torch.cuda.synchronize()
            
            # Measure memory
            torch.cuda.reset_peak_memory_stats()
            
            # Benchmark
            times = []
            with torch.backends.cuda.sdp_kernel(
                enable_flash=enable_flash,
                enable_math=enable_math,
                enable_mem_efficient=enable_mem_efficient
            ):
                for _ in range(self.iterations):
                    start = torch.cuda.Event(enable_timing=True)
                    end = torch.cuda.Event(enable_timing=True)
                    
                    start.record()
                    output = F.scaled_dot_product_attention(Q, K, V, is_causal=config.causal)
                    end.record()
                    
                    torch.cuda.synchronize()
                    elapsed_ms = start.elapsed_time(end)
                    times.append(elapsed_ms)
            
            memory_mb = torch.cuda.max_memory_allocated() / 1e6
            
            # Statistics
            times_np = np.array(times)
            mean_ms = float(np.mean(times_np))
            std_ms = float(np.std(times_np))
            median_ms = float(np.median(times_np))
            
            # Bootstrap 95% CI
            from scipy import stats
            ci = stats.bootstrap(
                (times_np,),
                np.median,
                confidence_level=0.95,
                n_resamples=1000,
                random_state=42
            )
            ci_95_low = float(ci.confidence_interval.low)
            ci_95_high = float(ci.confidence_interval.high)
            
            # Compute metrics
            flops = 4 * config.batch * config.heads * config.seq * config.seq * config.dim
            throughput_gflops = (flops / (mean_ms / 1000)) / 1e9
            
            bytes_transferred = (3 * config.batch * config.heads * config.seq * config.dim * 2 +
                               config.batch * config.heads * config.seq * config.dim * 2)
            bandwidth_gb_s = (bytes_transferred / (mean_ms / 1000)) / 1e9
            
            impl_name = f"PyTorch SDPA ({backend})" if backend != "auto" else "PyTorch SDPA"
            
            return BenchmarkResult(
                implementation=impl_name,
                config_name=config.name,
                median_ms=median_ms,
                mean_ms=mean_ms,
                std_ms=std_ms,
                ci_95_low=ci_95_low,
                ci_95_high=ci_95_high,
                throughput_gflops=throughput_gflops,
                bandwidth_gb_s=bandwidth_gb_s,
                memory_mb=memory_mb,
                iterations=len(times)
            )
        
        except Exception as e:
            return BenchmarkResult(
                implementation=f"PyTorch SDPA ({backend})",
                config_name=config.name,
                median_ms=0, mean_ms=0, std_ms=0,
                ci_95_low=0, ci_95_high=0,
                throughput_gflops=0, bandwidth_gb_s=0,
                memory_mb=0, iterations=0,
                error=str(e)
            )
    
    def _benchmark_flash_attn(
        self,
        config: BenchmarkConfig
    ) -> Optional[BenchmarkResult]:
        """Benchmark flash-attn (if available)"""
        if not self.has_flash_attn:
            return None
        
        try:
            from flash_attn import flash_attn_func
            
            # Create inputs (flash-attn uses different layout: B, S, H, D)
            Q = torch.randn(config.batch, config.seq, config.heads, config.dim,
                          device=self.device, dtype=config.dtype)
            K = torch.randn(config.batch, config.seq, config.heads, config.dim,
                          device=self.device, dtype=config.dtype)
            V = torch.randn(config.batch, config.seq, config.heads, config.dim,
                          device=self.device, dtype=config.dtype)
            
            # Warmup
            for _ in range(self.warmup):
                _ = flash_attn_func(Q, K, V, causal=config.causal)
            
            torch.cuda.synchronize()
            
            # Measure memory
            torch.cuda.reset_peak_memory_stats()
            
            # Benchmark
            times = []
            for _ in range(self.iterations):
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                
                start.record()
                output = flash_attn_func(Q, K, V, causal=config.causal)
                end.record()
                
                torch.cuda.synchronize()
                elapsed_ms = start.elapsed_time(end)
                times.append(elapsed_ms)
            
            memory_mb = torch.cuda.max_memory_allocated() / 1e6
            
            # Statistics (same as PyTorch SDPA)
            times_np = np.array(times)
            mean_ms = float(np.mean(times_np))
            std_ms = float(np.std(times_np))
            median_ms = float(np.median(times_np))
            
            from scipy import stats
            ci = stats.bootstrap(
                (times_np,),
                np.median,
                confidence_level=0.95,
                n_resamples=1000,
                random_state=42
            )
            ci_95_low = float(ci.confidence_interval.low)
            ci_95_high = float(ci.confidence_interval.high)
            
            # Compute metrics
            flops = 4 * config.batch * config.heads * config.seq * config.seq * config.dim
            throughput_gflops = (flops / (mean_ms / 1000)) / 1e9
            
            bytes_transferred = (3 * config.batch * config.heads * config.seq * config.dim * 2 +
                               config.batch * config.heads * config.seq * config.dim * 2)
            bandwidth_gb_s = (bytes_transferred / (mean_ms / 1000)) / 1e9
            
            return BenchmarkResult(
                implementation="flash-attn",
                config_name=config.name,
                median_ms=median_ms,
                mean_ms=mean_ms,
                std_ms=std_ms,
                ci_95_low=ci_95_low,
                ci_95_high=ci_95_high,
                throughput_gflops=throughput_gflops,
                bandwidth_gb_s=bandwidth_gb_s,
                memory_mb=memory_mb,
                iterations=len(times)
            )
        
        except Exception as e:
            return BenchmarkResult(
                implementation="flash-attn",
                config_name=config.name,
                median_ms=0, mean_ms=0, std_ms=0,
                ci_95_low=0, ci_95_high=0,
                throughput_gflops=0, bandwidth_gb_s=0,
                memory_mb=0, iterations=0,
                error=str(e)
            )
    
    def _benchmark_xformers(
        self,
        config: BenchmarkConfig
    ) -> Optional[BenchmarkResult]:
        """Benchmark xFormers (if available)"""
        if not self.has_xformers:
            return None
        
        try:
            import xformers.ops as xops
            
            # Create inputs (xFormers uses B, S, H, D layout)
            Q = torch.randn(config.batch, config.seq, config.heads, config.dim,
                          device=self.device, dtype=config.dtype)
            K = torch.randn(config.batch, config.seq, config.heads, config.dim,
                          device=self.device, dtype=config.dtype)
            V = torch.randn(config.batch, config.seq, config.heads, config.dim,
                          device=self.device, dtype=config.dtype)
            
            # Warmup
            for _ in range(self.warmup):
                _ = xops.memory_efficient_attention(Q, K, V)
            
            torch.cuda.synchronize()
            
            # Measure memory
            torch.cuda.reset_peak_memory_stats()
            
            # Benchmark
            times = []
            for _ in range(self.iterations):
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                
                start.record()
                output = xops.memory_efficient_attention(Q, K, V)
                end.record()
                
                torch.cuda.synchronize()
                elapsed_ms = start.elapsed_time(end)
                times.append(elapsed_ms)
            
            memory_mb = torch.cuda.max_memory_allocated() / 1e6
            
            # Statistics
            times_np = np.array(times)
            mean_ms = float(np.mean(times_np))
            std_ms = float(np.std(times_np))
            median_ms = float(np.median(times_np))
            
            from scipy import stats
            ci = stats.bootstrap(
                (times_np,),
                np.median,
                confidence_level=0.95,
                n_resamples=1000,
                random_state=42
            )
            ci_95_low = float(ci.confidence_interval.low)
            ci_95_high = float(ci.confidence_interval.high)
            
            # Compute metrics
            flops = 4 * config.batch * config.heads * config.seq * config.seq * config.dim
            throughput_gflops = (flops / (mean_ms / 1000)) / 1e9
            
            bytes_transferred = (3 * config.batch * config.heads * config.seq * config.dim * 2 +
                               config.batch * config.heads * config.seq * config.dim * 2)
            bandwidth_gb_s = (bytes_transferred / (mean_ms / 1000)) / 1e9
            
            return BenchmarkResult(
                implementation="xFormers",
                config_name=config.name,
                median_ms=median_ms,
                mean_ms=mean_ms,
                std_ms=std_ms,
                ci_95_low=ci_95_low,
                ci_95_high=ci_95_high,
                throughput_gflops=throughput_gflops,
                bandwidth_gb_s=bandwidth_gb_s,
                memory_mb=memory_mb,
                iterations=len(times)
            )
        
        except Exception as e:
            return BenchmarkResult(
                implementation="xFormers",
                config_name=config.name,
                median_ms=0, mean_ms=0, std_ms=0,
                ci_95_low=0, ci_95_high=0,
                throughput_gflops=0, bandwidth_gb_s=0,
                memory_mb=0, iterations=0,
                error=str(e)
            )
    
    def run(self, configs: List[BenchmarkConfig]) -> List[BenchmarkResult]:
        """Run benchmark suite across all configs and implementations"""
        print(f"🚀 Starting SOTA comparison")
        print(f"📊 Configs: {len(configs)}")
        print()
        
        for i, config in enumerate(configs, 1):
            print(f"[{i}/{len(configs)}] Config: {config.name}")
            print(f"  B={config.batch}, H={config.heads}, S={config.seq}, D={config.dim}")
            
            # PyTorch SDPA
            print(f"  Testing PyTorch SDPA...")
            result = self._benchmark_pytorch_sdpa(config, backend="auto")
            if result and not result.error:
                self.results.append(result)
                print(f"    ✅ {result.median_ms:.4f} ms ({result.bandwidth_gb_s:.1f} GB/s)")
            else:
                print(f"    ❌ Failed: {result.error if result else 'Unknown error'}")
            
            # flash-attn
            if self.has_flash_attn:
                print(f"  Testing flash-attn...")
                result = self._benchmark_flash_attn(config)
                if result and not result.error:
                    self.results.append(result)
                    print(f"    ✅ {result.median_ms:.4f} ms ({result.bandwidth_gb_s:.1f} GB/s)")
                else:
                    print(f"    ❌ Failed: {result.error if result else 'Unknown error'}")
            
            # xFormers
            if self.has_xformers:
                print(f"  Testing xFormers...")
                result = self._benchmark_xformers(config)
                if result and not result.error:
                    self.results.append(result)
                    print(f"    ✅ {result.median_ms:.4f} ms ({result.bandwidth_gb_s:.1f} GB/s)")
                else:
                    print(f"    ❌ Failed: {result.error if result else 'Unknown error'}")
            
            print()
        
        print(f"✅ Benchmark complete: {len(self.results)} results")
        return self.results
    
    def generate_report(self):
        """Generate publication-grade report"""
        if not self.results:
            print("❌ No results to report")
            return
        
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate markdown report
        report_path = self.output_dir / "sota_comparison_report.md"
        self._generate_markdown_report(report_path)
        
        # Generate JSON data
        json_path = self.output_dir / "sota_comparison_data.json"
        self._generate_json_data(json_path)
        
        print(f"\n📄 Report: {report_path}")
        print(f"📊 Data: {json_path}")
    
    def _generate_markdown_report(self, output_path: Path):
        """Generate markdown report"""
        lines = [
            "# SOTA Attention Benchmark Comparison\n",
            f"**GPU**: {self.gpu_name} ({self.gpu_memory_gb:.1f} GB)",
            f"**Date**: {time.strftime('%Y-%m-%d %H:%M:%S')}",
            f"**Iterations**: {self.iterations} (warmup: {self.warmup})",
            f"**Statistical Method**: Bootstrap 95% CI (N=1000 resamples)\n",
            "---\n",
            "## Executive Summary\n",
        ]
        
        # Group by config
        configs = sorted(set(r.config_name for r in self.results))
        
        lines.append(f"Tested **{len(configs)} configurations** across **{len(set(r.implementation for r in self.results))} implementations**:\n")
        
        implementations = sorted(set(r.implementation for r in self.results))
        for impl in implementations:
            lines.append(f"- {impl}")
        
        lines.append("\n## Results by Configuration\n")
        
        for config_name in configs:
            config_results = [r for r in self.results if r.config_name == config_name]
            
            lines.append(f"### {config_name}\n")
            lines.append("| Implementation | Latency (ms) | 95% CI | Bandwidth (GB/s) | Memory (MB) |")
            lines.append("|----------------|--------------|--------|------------------|-------------|")
            
            # Sort by latency (fastest first)
            config_results_sorted = sorted(config_results, key=lambda r: r.median_ms)
            
            for r in config_results_sorted:
                lines.append(
                    f"| {r.implementation} | **{r.median_ms:.4f}** | "
                    f"[{r.ci_95_low:.4f}, {r.ci_95_high:.4f}] | "
                    f"{r.bandwidth_gb_s:.1f} | {r.memory_mb:.0f} |"
                )
            
            # Winner
            winner = config_results_sorted[0]
            lines.append(f"\n**Winner**: {winner.implementation} ({winner.median_ms:.4f} ms)\n")
        
        lines.append("## Statistical Analysis\n")
        lines.append("All results use **bootstrap 95% confidence intervals** (N=1000 resamples).")
        lines.append("Implementations with **non-overlapping CIs** are statistically significantly different.\n")
        
        lines.append("## Reproducibility\n")
        lines.append("### Environment")
        lines.append(f"- GPU: {self.gpu_name}")
        lines.append(f"- PyTorch: {torch.__version__}")
        lines.append(f"- CUDA: {torch.version.cuda}")
        if self.has_flash_attn:
            import flash_attn
            lines.append(f"- flash-attn: {flash_attn.__version__}")
        if self.has_xformers:
            import xformers
            lines.append(f"- xFormers: {xformers.__version__}")
        
        lines.append("\n### Run Command")
        lines.append("```bash")
        lines.append("python sota_comparison.py --iterations 100 --warmup 20")
        lines.append("```\n")
        
        lines.append("## Raw Data\n")
        lines.append("```json")
        lines.append(json.dumps([asdict(r) for r in self.results], indent=2))
        lines.append("```\n")
        
        output_path.write_text("\n".join(lines))
    
    def _generate_json_data(self, output_path: Path):
        """Generate JSON data export"""
        data = {
            "metadata": {
                "gpu": self.gpu_name,
                "gpu_memory_gb": self.gpu_memory_gb,
                "date": time.strftime("%Y-%m-%d %H:%M:%S"),
                "iterations": self.iterations,
                "warmup": self.warmup,
                "pytorch_version": torch.__version__,
                "cuda_version": torch.version.cuda,
            },
            "results": [asdict(r) for r in self.results]
        }
        
        with open(output_path, 'w') as f:
            json.dump(data, f, indent=2)


def main():
    parser = argparse.ArgumentParser(
        description="SOTA attention benchmark comparison"
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=100,
        help="Benchmark iterations per config (default: 100)"
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=20,
        help="Warmup iterations per config (default: 20)"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts"),
        help="Output directory (default: artifacts/)"
    )
    
    args = parser.parse_args()
    
    # Define benchmark configs (focusing on optimized S=128 from Option B)
    configs = [
        # Optimized config from Option B
        BenchmarkConfig(
            name="Optimized (B=32, H=8, S=128, D=64)",
            batch=32, heads=8, seq=128, dim=64,
            dtype=torch.float16
        ),
        # Baseline from Option A
        BenchmarkConfig(
            name="Baseline (B=32, H=8, S=512, D=64)",
            batch=32, heads=8, seq=512, dim=64,
            dtype=torch.float16
        ),
        # Additional representative configs
        BenchmarkConfig(
            name="Small (B=4, H=8, S=256, D=64)",
            batch=4, heads=8, seq=256, dim=64,
            dtype=torch.float16
        ),
        BenchmarkConfig(
            name="Large (B=16, H=16, S=1024, D=64)",
            batch=16, heads=16, seq=1024, dim=64,
            dtype=torch.float16
        ),
    ]
    
    # Run benchmark
    try:
        benchmark = SOTABenchmark(
            iterations=args.iterations,
            warmup=args.warmup,
            output_dir=args.output
        )
        
        results = benchmark.run(configs)
        
        if not results:
            print("❌ No successful benchmark runs", file=sys.stderr)
            return 1
        
        # Generate report
        benchmark.generate_report()
        
        print("\n✅ SOTA comparison complete")
        return 0
    
    except KeyboardInterrupt:
        print("\n⚠️  Benchmark interrupted")
        return 130
    except Exception as e:
        print(f"\n❌ Benchmark failed: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())

