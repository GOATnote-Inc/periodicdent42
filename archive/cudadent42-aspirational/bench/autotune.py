#!/usr/bin/env python3
"""
Lightweight autotuner for CUDA kernel parameters

Searches small parameter space (tile sizes, stages, etc.) and suggests
best settings. Designed for manual dispatch in CI - human decides to apply.

Author: Brandon Dent (b@thegoatnote.com)
License: Apache 2.0
"""

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, asdict
import itertools


@dataclass
class TuneParam:
    """Single tunable parameter"""
    name: str
    values: List[any]
    current_default: any


@dataclass
class TuneResult:
    """Result of one parameter combination"""
    params: Dict[str, any]
    median_ms: float
    throughput_gflops: float
    speedup_vs_baseline: float
    build_time_s: float


class KernelAutotuner:
    """
    Lightweight autotuner - exhaustive search over small parameter space
    
    Design:
    - Small search space (typically <100 combinations)
    - Exhaustive (not Bayesian) to avoid dependencies
    - Compiles + benchmarks each combination
    - Outputs best params + markdown report
    """
    
    def __init__(
        self,
        config_name: str,
        time_budget_minutes: int = 20,
        min_iterations: int = 3  # Iterations per param combo
    ):
        self.config_name = config_name
        self.time_budget = time_budget_minutes * 60
        self.min_iterations = min_iterations
        self.results: List[TuneResult] = []
        self.baseline_ms: Optional[float] = None
        
        # Define tunable parameters (kernel-specific)
        self.params = self._get_tunable_params()
    
    def _get_tunable_params(self) -> List[TuneParam]:
        """
        Define search space
        
        TODO: Make this configurable via JSON or kernel introspection
        For now, common FlashAttention parameters
        """
        return [
            TuneParam(
                name="BLOCK_M",
                values=[64, 128],  # Small space for demo
                current_default=64
            ),
            TuneParam(
                name="BLOCK_N",
                values=[64, 128],
                current_default=64
            ),
            TuneParam(
                name="NUM_STAGES",
                values=[1, 2],
                current_default=1
            ),
        ]
    
    def _build_with_params(self, params: Dict[str, any]) -> bool:
        """
        Rebuild kernel with specific compile-time parameters
        
        Returns:
            True if build succeeded
        """
        # Generate compile flags
        flags = []
        for name, value in params.items():
            flags.append(f"-D{name}={value}")
        
        # Build command
        cmd = [
            "python3", "setup.py", "build_ext", "--inplace",
            f"--extra-compile-args={' '.join(flags)}"
        ]
        
        try:
            start = time.time()
            result = subprocess.run(
                cmd,
                cwd=Path.cwd().parent if Path.cwd().name == "bench" else Path.cwd(),
                capture_output=True,
                timeout=300
            )
            build_time = time.time() - start
            
            if result.returncode != 0:
                print(f"  ❌ Build failed for {params}")
                return False, 0
            
            return True, build_time
        except subprocess.TimeoutExpired:
            print(f"  ⏱️  Build timeout for {params}")
            return False, 0
    
    def _benchmark_config(self) -> Optional[float]:
        """
        Run benchmark for current kernel build
        
        Returns:
            Median latency in ms, or None if failed
        """
        cmd = [
            "python3", "integrated_test.py",
            "--config", self.config_name,
            "--iterations", str(self.min_iterations),
            "--output", "results/tune_temp.json"
        ]
        
        try:
            result = subprocess.run(
                cmd,
                cwd=Path.cwd() if Path.cwd().name == "bench" else Path.cwd() / "bench",
                capture_output=True,
                timeout=60
            )
            
            if result.returncode != 0:
                return None
            
            # Parse result
            with open("results/tune_temp.json") as f:
                data = json.load(f)
            
            perf = data.get("performance", data.get("metrics", {}))
            return perf.get("mean_time_ms")
        
        except (subprocess.TimeoutExpired, FileNotFoundError, json.JSONDecodeError):
            return None
    
    def _get_baseline(self) -> float:
        """Get baseline performance (current default params)"""
        if self.baseline_ms is not None:
            return self.baseline_ms
        
        # Build with default params
        default_params = {p.name: p.current_default for p in self.params}
        success, _ = self._build_with_params(default_params)
        
        if not success:
            raise RuntimeError("Failed to build baseline")
        
        self.baseline_ms = self._benchmark_config()
        if self.baseline_ms is None:
            raise RuntimeError("Failed to benchmark baseline")
        
        print(f"✅ Baseline: {self.baseline_ms:.4f} ms")
        return self.baseline_ms
    
    def run(self) -> List[TuneResult]:
        """
        Run autotuning within time budget
        
        Returns:
            List of results, sorted by speedup
        """
        print(f"🔧 Starting autotune for {self.config_name}")
        print(f"⏱️  Time budget: {self.time_budget / 60:.1f} minutes")
        
        # Get baseline
        baseline_ms = self._get_baseline()
        
        # Generate all parameter combinations
        param_names = [p.name for p in self.params]
        param_values = [p.values for p in self.params]
        combinations = list(itertools.product(*param_values))
        
        print(f"🔍 Search space: {len(combinations)} combinations")
        
        start_time = time.time()
        tested = 0
        
        for combo in combinations:
            # Check time budget
            elapsed = time.time() - start_time
            if elapsed > self.time_budget:
                print(f"\n⏱️  Time budget exceeded ({elapsed/60:.1f} min)")
                break
            
            params = dict(zip(param_names, combo))
            
            # Skip baseline (already tested)
            if all(params[p.name] == p.current_default for p in self.params):
                continue
            
            print(f"\n[{tested+1}/{len(combinations)}] Testing {params}")
            
            # Build
            success, build_time = self._build_with_params(params)
            if not success:
                continue
            
            # Benchmark
            median_ms = self._benchmark_config()
            if median_ms is None:
                print("  ❌ Benchmark failed")
                continue
            
            speedup = baseline_ms / median_ms
            throughput = 1000.0 / median_ms  # Placeholder GFLOPS
            
            result = TuneResult(
                params=params,
                median_ms=median_ms,
                throughput_gflops=throughput,
                speedup_vs_baseline=speedup,
                build_time_s=build_time
            )
            
            self.results.append(result)
            tested += 1
            
            # Show result
            status = "✅" if speedup > 1.0 else "❌"
            print(f"  {status} {median_ms:.4f} ms (speedup: {speedup:.3f}×)")
        
        # Sort by speedup
        self.results.sort(key=lambda r: r.speedup_vs_baseline, reverse=True)
        
        print(f"\n✅ Tested {tested} combinations in {(time.time() - start_time)/60:.1f} min")
        
        return self.results
    
    def generate_report(self, output_path: Path):
        """Generate markdown report with recommendations"""
        if not self.results:
            return
        
        best = self.results[0]
        
        lines = [
            f"# Autotune Report: {self.config_name}\n",
            "## Best Configuration Found\n",
            f"**Speedup**: {best.speedup_vs_baseline:.3f}×\n",
            f"**Latency**: {best.median_ms:.4f} ms (baseline: {self.baseline_ms:.4f} ms)\n",
            f"**Parameters**:\n"
        ]
        
        for name, value in best.params.items():
            lines.append(f"- `{name} = {value}`")
        
        lines.append("\n## How to Apply\n")
        lines.append("### Option 1: Compile-time flags\n")
        lines.append("```cmake")
        for name, value in best.params.items():
            lines.append(f"target_compile_definitions(kernel PRIVATE {name}={value})")
        lines.append("```\n")
        
        lines.append("### Option 2: nvcc flags\n")
        lines.append("```bash")
        flags = " ".join(f"-D{name}={value}" for name, value in best.params.items())
        lines.append(f"nvcc {flags} flash_attention.cu")
        lines.append("```\n")
        
        lines.append("## Top 5 Results\n")
        lines.append("| Rank | Speedup | Latency (ms) | Parameters |")
        lines.append("|------|---------|--------------|------------|")
        
        for i, result in enumerate(self.results[:5], 1):
            params_str = ", ".join(f"{k}={v}" for k, v in result.params.items())
            lines.append(
                f"| {i} | {result.speedup_vs_baseline:.3f}× | "
                f"{result.median_ms:.4f} | {params_str} |"
            )
        
        lines.append("\n## All Tested Configurations\n")
        lines.append("```json")
        lines.append(json.dumps([asdict(r) for r in self.results], indent=2))
        lines.append("```\n")
        
        output_path.write_text("\n".join(lines))
        print(f"\n✅ Report written to {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Autotune CUDA kernel parameters"
    )
    parser.add_argument(
        "--config",
        required=True,
        help="Config name to tune (e.g., training_512)"
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
        default=3,
        help="Benchmark iterations per config (default: 3)"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("tuning/suggestions.md"),
        help="Output path for suggestions (default: tuning/suggestions.md)"
    )
    
    args = parser.parse_args()
    
    # Run autotuning
    tuner = KernelAutotuner(
        config_name=args.config,
        time_budget_minutes=args.time_budget,
        min_iterations=args.iterations
    )
    
    try:
        results = tuner.run()
        
        if not results:
            print("❌ No successful tuning runs", file=sys.stderr)
            return 1
        
        # Generate report
        args.output.parent.mkdir(parents=True, exist_ok=True)
        tuner.generate_report(args.output)
        
        # Show best result
        best = results[0]
        if best.speedup_vs_baseline > 1.05:
            print(f"\n🎉 Found {best.speedup_vs_baseline:.2f}× speedup!")
            print(f"📋 See {args.output} for details")
            return 0
        else:
            print(f"\n⚠️  No significant improvement found (best: {best.speedup_vs_baseline:.3f}×)")
            return 1
    
    except KeyboardInterrupt:
        print("\n⚠️  Tuning interrupted")
        return 130
    except Exception as e:
        print(f"\n❌ Tuning failed: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())

