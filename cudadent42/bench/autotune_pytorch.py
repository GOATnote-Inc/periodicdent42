#!/usr/bin/env python3
"""
PyTorch SDPA Autotuner - Find optimal attention configurations

Since custom CUDA kernel has build issues, this autotuner finds
optimal PyTorch SDPA configurations for the target GPU.

Tunes:
- SDPA backend (flash, memory_efficient, math)
- Batch size
- Number of heads
- Sequence length
- Head dimension
- Data layout

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
import itertools
import torch
import torch.nn.functional as F


@dataclass
class ConfigResult:
    """Result for one configuration"""
    config: Dict[str, any]
    median_ms: float
    mean_ms: float
    std_ms: float
    throughput_gflops: float
    bandwidth_gb_s: float
    speedup_vs_baseline: float
    iterations: int


class PyTorchSDPAAutotuner:
    """
    Find optimal PyTorch SDPA configurations for target GPU
    
    Strategy:
    1. Fix one reference config (e.g., B=32, H=8, S=512, D=64)
    2. Vary one dimension at a time to find optimal ranges
    3. Test different SDPA backends
    4. Report best configurations
    """
    
    def __init__(
        self,
        time_budget_minutes: int = 20,
        iterations_per_config: int = 50,
        warmup: int = 10
    ):
        self.time_budget = time_budget_minutes * 60
        self.iterations = iterations_per_config
        self.warmup = warmup
        self.results: List[ConfigResult] = []
        self.baseline_ms: Optional[float] = None
        
        # Check CUDA
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA not available")
        
        self.device = torch.device("cuda")
        self.gpu_name = torch.cuda.get_device_name(0)
        
        print(f"🔧 GPU: {self.gpu_name}")
    
    def _benchmark_config(
        self,
        batch: int,
        heads: int,
        seq: int,
        dim: int,
        backend: str = "auto",
        dtype: torch.dtype = torch.float16
    ) -> Optional[ConfigResult]:
        """Benchmark one configuration"""
        
        # Create inputs
        try:
            Q = torch.randn(batch, heads, seq, dim, device=self.device, dtype=dtype)
            K = torch.randn(batch, heads, seq, dim, device=self.device, dtype=dtype)
            V = torch.randn(batch, heads, seq, dim, device=self.device, dtype=dtype)
        except RuntimeError as e:
            # OOM or other error
            return None
        
        # Configure backend flags
        if backend == "flash":
            enable_flash = True
            enable_math = False
            enable_mem_efficient = False
        elif backend == "memory_efficient":
            enable_flash = False
            enable_math = False
            enable_mem_efficient = True
        elif backend == "math":
            enable_flash = False
            enable_math = True
            enable_mem_efficient = False
        else:  # auto
            enable_flash = True
            enable_math = False
            enable_mem_efficient = True
        
        # Warmup
        with torch.backends.cuda.sdp_kernel(
            enable_flash=enable_flash,
            enable_math=enable_math,
            enable_mem_efficient=enable_mem_efficient
        ):
            for _ in range(self.warmup):
                _ = F.scaled_dot_product_attention(Q, K, V)
        
        torch.cuda.synchronize()
        
        # Benchmark
        times = []
        try:
            with torch.backends.cuda.sdp_kernel(
                enable_flash=enable_flash,
                enable_math=enable_math,
                enable_mem_efficient=enable_mem_efficient
            ):
                for _ in range(self.iterations):
                    start = torch.cuda.Event(enable_timing=True)
                    end = torch.cuda.Event(enable_timing=True)
                    
                    start.record()
                    output = F.scaled_dot_product_attention(Q, K, V)
                    end.record()
                    
                    torch.cuda.synchronize()
                    elapsed_ms = start.elapsed_time(end)
                    times.append(elapsed_ms)
        except RuntimeError as e:
            return None
        
        # Statistics
        import statistics
        mean_ms = statistics.mean(times)
        std_ms = statistics.stdev(times) if len(times) > 1 else 0.0
        median_ms = statistics.median(times)
        
        # Compute metrics
        flops = 4 * batch * heads * seq * seq * dim
        throughput_gflops = (flops / (mean_ms / 1000)) / 1e9
        
        bytes_transferred = 3 * batch * heads * seq * dim * 2 + batch * heads * seq * dim * 2
        bandwidth_gb_s = (bytes_transferred / (mean_ms / 1000)) / 1e9
        
        config = {
            "batch": batch,
            "heads": heads,
            "seq": seq,
            "dim": dim,
            "backend": backend,
            "dtype": str(dtype)
        }
        
        speedup = self.baseline_ms / median_ms if self.baseline_ms else 1.0
        
        return ConfigResult(
            config=config,
            median_ms=median_ms,
            mean_ms=mean_ms,
            std_ms=std_ms,
            throughput_gflops=throughput_gflops,
            bandwidth_gb_s=bandwidth_gb_s,
            speedup_vs_baseline=speedup,
            iterations=len(times)
        )
    
    def run(self) -> List[ConfigResult]:
        """
        Run autotuning within time budget
        
        Strategy:
        1. Establish baseline (B=32, H=8, S=512, D=64, auto backend)
        2. Test different backends for baseline config
        3. Vary batch size (keeping H,S,D fixed)
        4. Vary sequence length (keeping B,H,D fixed)
        5. Vary number of heads (keeping B,S,D fixed)
        """
        
        print(f"⏱️  Time budget: {self.time_budget / 60:.1f} minutes")
        print(f"🔬 Iterations per config: {self.iterations} (warmup: {self.warmup})")
        print()
        
        start_time = time.time()
        
        # 1. Baseline
        print("📊 Phase 1: Baseline")
        baseline_config = self._benchmark_config(
            batch=32, heads=8, seq=512, dim=64, backend="auto"
        )
        if baseline_config is None:
            raise RuntimeError("Failed to benchmark baseline")
        
        self.baseline_ms = baseline_config.median_ms
        self.results.append(baseline_config)
        
        print(f"✅ Baseline: {baseline_config.median_ms:.4f} ms")
        print()
        
        # 2. Backend sweep (same config, different backends)
        print("📊 Phase 2: Backend Sweep (B=32, H=8, S=512, D=64)")
        for backend in ["flash", "memory_efficient", "math"]:
            if time.time() - start_time > self.time_budget:
                break
            
            print(f"  Testing backend: {backend}")
            result = self._benchmark_config(
                batch=32, heads=8, seq=512, dim=64, backend=backend
            )
            if result:
                self.results.append(result)
                status = "✅" if result.speedup_vs_baseline > 1.0 else "⚪"
                print(f"  {status} {result.median_ms:.4f} ms (speedup: {result.speedup_vs_baseline:.3f}×)")
        print()
        
        # 3. Batch size sweep
        print("📊 Phase 3: Batch Size Sweep (H=8, S=512, D=64, backend=auto)")
        for batch in [4, 8, 16, 32, 64]:
            if time.time() - start_time > self.time_budget:
                break
            
            print(f"  Testing batch: {batch}")
            result = self._benchmark_config(
                batch=batch, heads=8, seq=512, dim=64, backend="auto"
            )
            if result:
                self.results.append(result)
                print(f"  ✅ {result.median_ms:.4f} ms ({result.throughput_gflops:.0f} GFLOPS)")
        print()
        
        # 4. Sequence length sweep
        print("📊 Phase 4: Sequence Length Sweep (B=32, H=8, D=64, backend=auto)")
        for seq in [128, 256, 512, 1024, 2048]:
            if time.time() - start_time > self.time_budget:
                break
            
            print(f"  Testing seq: {seq}")
            result = self._benchmark_config(
                batch=32, heads=8, seq=seq, dim=64, backend="auto"
            )
            if result:
                self.results.append(result)
                print(f"  ✅ {result.median_ms:.4f} ms ({result.bandwidth_gb_s:.1f} GB/s)")
        print()
        
        # 5. Head count sweep
        print("📊 Phase 5: Head Count Sweep (B=32, S=512, D=64, backend=auto)")
        for heads in [4, 8, 12, 16]:
            if time.time() - start_time > self.time_budget:
                break
            
            print(f"  Testing heads: {heads}")
            result = self._benchmark_config(
                batch=32, heads=heads, seq=512, dim=64, backend="auto"
            )
            if result:
                self.results.append(result)
                print(f"  ✅ {result.median_ms:.4f} ms")
        print()
        
        elapsed = time.time() - start_time
        print(f"✅ Tested {len(self.results)} configurations in {elapsed/60:.1f} min")
        
        return self.results
    
    def generate_report(self, output_path: Path):
        """Generate markdown report with recommendations"""
        if not self.results:
            return
        
        # Find best overall config
        best_overall = min(self.results, key=lambda r: r.median_ms)
        
        # Find best per category
        backends = [r for r in self.results if r.config["batch"] == 32 and r.config["heads"] == 8 
                    and r.config["seq"] == 512 and r.config["dim"] == 64]
        best_backend = min(backends, key=lambda r: r.median_ms) if backends else None
        
        lines = [
            f"# PyTorch SDPA Autotune Report\n",
            f"**GPU**: {self.gpu_name}",
            f"**Date**: {time.strftime('%Y-%m-%d %H:%M:%S')}",
            f"**Iterations per config**: {self.iterations}",
            f"**Baseline**: {self.baseline_ms:.4f} ms (B=32, H=8, S=512, D=64, auto backend)\n",
            "---\n",
            "## 🏆 Best Configuration Found\n",
            f"**Config**: B={best_overall.config['batch']}, H={best_overall.config['heads']}, "
            f"S={best_overall.config['seq']}, D={best_overall.config['dim']}, "
            f"backend={best_overall.config['backend']}",
            f"**Latency**: {best_overall.median_ms:.4f} ms (±{best_overall.std_ms:.4f} ms)",
            f"**Speedup vs Baseline**: {best_overall.speedup_vs_baseline:.3f}×",
            f"**Throughput**: {best_overall.throughput_gflops:.0f} GFLOPS",
            f"**Bandwidth**: {best_overall.bandwidth_gb_s:.1f} GB/s\n",
        ]
        
        if best_backend:
            lines.extend([
                "## 🔧 Best Backend (for B=32, H=8, S=512, D=64)\n",
                f"**Backend**: `{best_backend.config['backend']}`",
                f"**Latency**: {best_backend.median_ms:.4f} ms",
                f"**Speedup vs Auto**: {self.baseline_ms / best_backend.median_ms:.3f}×\n",
                "### How to Use",
                "```python",
                f"with torch.backends.cuda.sdp_kernel(",
                f"    enable_flash={best_backend.config['backend'] == 'flash'},",
                f"    enable_memory_efficient={best_backend.config['backend'] == 'memory_efficient'},",
                f"    enable_math={best_backend.config['backend'] == 'math'}",
                "):",
                "    output = F.scaled_dot_product_attention(Q, K, V)",
                "```\n"
            ])
        
        # Batch size recommendations
        batch_results = [r for r in self.results if r.config["heads"] == 8 
                         and r.config["seq"] == 512 and r.config["dim"] == 64]
        if len(batch_results) > 1:
            lines.append("## 📊 Batch Size Analysis (H=8, S=512, D=64)\n")
            lines.append("| Batch | Latency (ms) | Throughput (GFLOPS) | Samples/sec |")
            lines.append("|-------|--------------|---------------------|-------------|")
            for r in sorted(batch_results, key=lambda x: x.config["batch"]):
                samples_per_sec = r.config["batch"] * 1000 / r.median_ms
                lines.append(
                    f"| {r.config['batch']} | {r.median_ms:.4f} | "
                    f"{r.throughput_gflops:.0f} | {samples_per_sec:.0f} |"
                )
            lines.append("")
        
        # Sequence length recommendations
        seq_results = [r for r in self.results if r.config["batch"] == 32 
                       and r.config["heads"] == 8 and r.config["dim"] == 64]
        if len(seq_results) > 1:
            lines.append("## 📏 Sequence Length Analysis (B=32, H=8, D=64)\n")
            lines.append("| Seq Length | Latency (ms) | Bandwidth (GB/s) | Efficiency (%) |")
            lines.append("|------------|--------------|------------------|----------------|")
            peak_bw = 242.0  # L4 peak bandwidth
            for r in sorted(seq_results, key=lambda x: x.config["seq"]):
                efficiency = (r.bandwidth_gb_s / peak_bw) * 100
                lines.append(
                    f"| {r.config['seq']} | {r.median_ms:.4f} | "
                    f"{r.bandwidth_gb_s:.1f} | {efficiency:.1f}% |"
                )
            lines.append("")
        
        # Key insights
        lines.extend([
            "## 💡 Key Insights\n",
        ])
        
        # Backend insight
        if best_backend and best_backend.config['backend'] != 'auto':
            improvement = (self.baseline_ms / best_backend.median_ms - 1.0) * 100
            lines.append(
                f"1. **Backend Selection**: Using `{best_backend.config['backend']}` backend "
                f"gives {improvement:.1f}% speedup over auto-selection for the baseline config."
            )
        
        # Batch size insight
        if len(batch_results) > 2:
            best_batch = min(batch_results, key=lambda r: r.median_ms / r.config["batch"])
            lines.append(
                f"2. **Batch Size**: B={best_batch.config['batch']} achieves best "
                f"samples/second ({best_batch.config['batch'] * 1000 / best_batch.median_ms:.0f} samples/s)."
            )
        
        # Sequence length insight
        if len(seq_results) > 2:
            best_efficiency = max(seq_results, key=lambda r: r.bandwidth_gb_s)
            lines.append(
                f"3. **Sequence Length**: S={best_efficiency.config['seq']} achieves best "
                f"memory bandwidth utilization ({best_efficiency.bandwidth_gb_s:.1f} GB/s)."
            )
        
        lines.extend([
            "",
            "## 📈 All Results\n",
            "```json",
            json.dumps([asdict(r) for r in self.results], indent=2),
            "```\n"
        ])
        
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text("\n".join(lines))
        print(f"\n✅ Report written to {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Autotune PyTorch SDPA configurations for target GPU"
    )
    parser.add_argument(
        "--time-budget",
        type=int,
        default=20,
        help="Time budget in minutes (default: 20)"
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=50,
        help="Benchmark iterations per config (default: 50)"
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=10,
        help="Warmup iterations per config (default: 10)"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("tuning/pytorch_sdpa_suggestions.md"),
        help="Output path for report (default: tuning/pytorch_sdpa_suggestions.md)"
    )
    
    args = parser.parse_args()
    
    # Run autotuning
    try:
        tuner = PyTorchSDPAAutotuner(
            time_budget_minutes=args.time_budget,
            iterations_per_config=args.iterations,
            warmup=args.warmup
        )
        
        results = tuner.run()
        
        if not results:
            print("❌ No successful tuning runs", file=sys.stderr)
            return 1
        
        # Generate report
        tuner.generate_report(args.output)
        
        # Find best
        best = min(results, key=lambda r: r.median_ms)
        improvement = (tuner.baseline_ms / best.median_ms - 1.0) * 100
        
        if improvement > 5.0:
            print(f"\n🎉 Found {improvement:.1f}% speedup!")
            print(f"📋 See {args.output} for details")
            return 0
        else:
            print(f"\n✅ Autotune complete (best improvement: {improvement:.1f}%)")
            print(f"📋 See {args.output} for details")
            return 0
    
    except KeyboardInterrupt:
        print("\n⚠️  Tuning interrupted")
        return 130
    except Exception as e:
        print(f"\n❌ Tuning failed: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())

