#!/usr/bin/env python3
"""
Performance Ratcheting System - Makes CI iteratively improve CUDA kernels

Core concept: Track best-known performance per (GPU, config), fail on regression,
auto-suggest improvements when gains are found.

Author: Brandon Dent (b@thegoatnote.com)
License: Apache 2.0
"""

import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import argparse


@dataclass
class PerfResult:
    """Single performance measurement"""
    config_name: str
    median_ms: float
    ci_95_low: float
    ci_95_high: float
    throughput_gflops: Optional[float] = None
    bandwidth_gb_s: Optional[float] = None
    git_commit: Optional[str] = None
    timestamp: Optional[str] = None


@dataclass
class RatchetDecision:
    """Decision about a config's performance"""
    config_name: str
    current_ms: float
    baseline_ms: Optional[float]
    delta_pct: Optional[float]  # Positive = faster
    is_regression: bool
    is_improvement: bool
    keep_baseline: bool  # True if baseline is still best


class PerformanceRatchet:
    """
    Ratcheting system: keeps best-known results, fails on regression
    
    Design:
    - baseline.json stores best-known median per config
    - Always compare to best (not just previous commit)
    - Update baseline if current is faster
    - Fail CI if regression > threshold
    """
    
    def __init__(
        self,
        baseline_path: Path,
        regression_threshold_pct: float = -3.0,
        improvement_threshold_pct: float = 5.0
    ):
        self.baseline_path = baseline_path
        self.regression_threshold = regression_threshold_pct
        self.improvement_threshold = improvement_threshold_pct
        
        # Load existing baseline
        self.baseline = self._load_baseline()
    
    def _load_baseline(self) -> Dict[str, PerfResult]:
        """Load baseline from disk, or empty if doesn't exist"""
        if not self.baseline_path.exists():
            return {}
        
        with open(self.baseline_path) as f:
            data = json.load(f)
        
        results = {}
        for config_name, config_data in data.get("configs", {}).items():
            results[config_name] = PerfResult(
                config_name=config_name,
                median_ms=config_data["median_ms"],
                ci_95_low=config_data.get("ci_95_low", config_data["median_ms"]),
                ci_95_high=config_data.get("ci_95_high", config_data["median_ms"]),
                throughput_gflops=config_data.get("throughput_gflops"),
                bandwidth_gb_s=config_data.get("bandwidth_gb_s"),
                git_commit=config_data.get("git_commit"),
                timestamp=config_data.get("timestamp")
            )
        
        return results
    
    def _save_baseline(self):
        """Save updated baseline to disk"""
        data = {
            "configs": {
                name: {
                    "median_ms": result.median_ms,
                    "ci_95_low": result.ci_95_low,
                    "ci_95_high": result.ci_95_high,
                    "throughput_gflops": result.throughput_gflops,
                    "bandwidth_gb_s": result.bandwidth_gb_s,
                    "git_commit": result.git_commit,
                    "timestamp": result.timestamp
                }
                for name, result in self.baseline.items()
            }
        }
        
        self.baseline_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.baseline_path, 'w') as f:
            json.dump(data, f, indent=2)
    
    def evaluate(self, current: List[PerfResult]) -> Tuple[List[RatchetDecision], bool]:
        """
        Evaluate current results against baseline
        
        Returns:
            (decisions, has_regression)
        """
        decisions = []
        has_regression = False
        
        for cur in current:
            baseline = self.baseline.get(cur.config_name)
            
            if baseline is None:
                # New config - accept and add to baseline
                decision = RatchetDecision(
                    config_name=cur.config_name,
                    current_ms=cur.median_ms,
                    baseline_ms=None,
                    delta_pct=None,
                    is_regression=False,
                    is_improvement=True,  # New = improvement by definition
                    keep_baseline=False
                )
                self.baseline[cur.config_name] = cur
            else:
                # Compare to baseline
                delta_pct = (baseline.median_ms - cur.median_ms) / baseline.median_ms * 100
                
                is_regression = delta_pct < self.regression_threshold
                is_improvement = delta_pct > self.improvement_threshold
                
                # Update baseline if current is faster
                keep_baseline = cur.median_ms >= baseline.median_ms
                if not keep_baseline:
                    self.baseline[cur.config_name] = cur
                
                decision = RatchetDecision(
                    config_name=cur.config_name,
                    current_ms=cur.median_ms,
                    baseline_ms=baseline.median_ms,
                    delta_pct=delta_pct,
                    is_regression=is_regression,
                    is_improvement=is_improvement,
                    keep_baseline=keep_baseline
                )
                
                if is_regression:
                    has_regression = True
            
            decisions.append(decision)
        
        # Save updated baseline
        self._save_baseline()
        
        return decisions, has_regression
    
    def format_report(self, decisions: List[RatchetDecision]) -> str:
        """Generate human-readable report"""
        lines = ["# Performance Ratchet Report\n"]
        
        # Summary
        total = len(decisions)
        regressions = sum(1 for d in decisions if d.is_regression)
        improvements = sum(1 for d in decisions if d.is_improvement)
        unchanged = total - regressions - improvements
        
        lines.append("## Summary\n")
        lines.append(f"- **Total configs**: {total}")
        lines.append(f"- **Regressions**: {regressions} ❌")
        lines.append(f"- **Improvements**: {improvements} ✅")
        lines.append(f"- **Unchanged**: {unchanged}")
        lines.append("")
        
        if regressions > 0:
            lines.append("## ❌ Regressions (FAIL)\n")
            lines.append("| Config | Baseline | Current | Change |")
            lines.append("|--------|----------|---------|--------|")
            for d in decisions:
                if d.is_regression:
                    lines.append(
                        f"| {d.config_name} | {d.baseline_ms:.4f} ms | "
                        f"{d.current_ms:.4f} ms | **{d.delta_pct:+.1f}%** |"
                    )
            lines.append("")
        
        if improvements > 0:
            lines.append("## ✅ Improvements (Baseline Updated)\n")
            lines.append("| Config | Baseline | Current | Change |")
            lines.append("|--------|----------|---------|--------|")
            for d in decisions:
                if d.is_improvement:
                    baseline_str = f"{d.baseline_ms:.4f} ms" if d.baseline_ms else "NEW"
                    delta_str = f"{d.delta_pct:+.1f}%" if d.delta_pct else "N/A"
                    lines.append(
                        f"| {d.config_name} | {baseline_str} | "
                        f"{d.current_ms:.4f} ms | **{delta_str}** |"
                    )
            lines.append("")
        
        if unchanged > 0:
            lines.append("## ↔️ Unchanged\n")
            lines.append("| Config | Current | Change |")
            lines.append("|--------|---------|--------|")
            for d in decisions:
                if not d.is_regression and not d.is_improvement:
                    delta_str = f"{d.delta_pct:+.1f}%" if d.delta_pct else "N/A"
                    lines.append(
                        f"| {d.config_name} | {d.current_ms:.4f} ms | {delta_str} |"
                    )
            lines.append("")
        
        return "\n".join(lines)
    
    def get_profile_targets(self, decisions: List[RatchetDecision]) -> List[str]:
        """
        Return configs that should be profiled (regressions or large improvements)
        """
        targets = []
        for d in decisions:
            if d.is_regression:
                targets.append(d.config_name)
            elif d.is_improvement and d.delta_pct and d.delta_pct > 10.0:
                # Large improvement - profile to understand why
                targets.append(d.config_name)
        return targets


def load_current_results(path: Path) -> List[PerfResult]:
    """Load current benchmark results"""
    with open(path) as f:
        data = json.load(f)
    
    results = []
    
    # Handle both old and new formats
    if "configs" in data:
        configs = data["configs"]
    else:
        configs = [data]  # Single result
    
    for config in configs:
        # Extract performance data
        if "performance" in config:
            perf = config["performance"]
        elif "metrics" in config:
            perf = config["metrics"]
        else:
            perf = config
        
        results.append(PerfResult(
            config_name=config.get("name", config.get("config", {}).get("name", "unknown")),
            median_ms=perf["mean_time_ms"],  # Use mean as proxy for median if needed
            ci_95_low=perf.get("ci_95_low", perf["mean_time_ms"] - perf.get("std_dev_ms", 0)),
            ci_95_high=perf.get("ci_95_high", perf["mean_time_ms"] + perf.get("std_dev_ms", 0)),
            throughput_gflops=perf.get("throughput_gflops"),
            bandwidth_gb_s=perf.get("bandwidth_gb_s"),
            git_commit=config.get("git_commit"),
            timestamp=config.get("timestamp")
        ))
    
    return results


def main():
    parser = argparse.ArgumentParser(
        description="Performance ratcheting - track best results, fail on regression"
    )
    parser.add_argument(
        "current_results",
        type=Path,
        help="Path to current benchmark results (JSON)"
    )
    parser.add_argument(
        "--baseline",
        type=Path,
        default=Path("results/baseline.json"),
        help="Path to baseline file (default: results/baseline.json)"
    )
    parser.add_argument(
        "--regression-threshold",
        type=float,
        default=-3.0,
        help="Regression threshold in percent (default: -3.0 = 3%% slower fails)"
    )
    parser.add_argument(
        "--improvement-threshold",
        type=float,
        default=5.0,
        help="Improvement threshold in percent (default: 5.0 = 5%% faster triggers profile)"
    )
    parser.add_argument(
        "--output-report",
        type=Path,
        help="Output path for markdown report"
    )
    parser.add_argument(
        "--output-profile-targets",
        type=Path,
        help="Output path for configs to profile (newline-separated)"
    )
    
    args = parser.parse_args()
    
    # Load current results
    try:
        current_results = load_current_results(args.current_results)
    except Exception as e:
        print(f"❌ Failed to load current results: {e}", file=sys.stderr)
        return 1
    
    if not current_results:
        print("⚠️  No results found in current file", file=sys.stderr)
        return 1
    
    # Evaluate against baseline
    ratchet = PerformanceRatchet(
        baseline_path=args.baseline,
        regression_threshold_pct=args.regression_threshold,
        improvement_threshold_pct=args.improvement_threshold
    )
    
    decisions, has_regression = ratchet.evaluate(current_results)
    
    # Generate report
    report = ratchet.format_report(decisions)
    
    if args.output_report:
        args.output_report.write_text(report)
        print(f"✅ Report written to {args.output_report}")
    else:
        print(report)
    
    # Get profile targets
    profile_targets = ratchet.get_profile_targets(decisions)
    if args.output_profile_targets and profile_targets:
        args.output_profile_targets.write_text("\n".join(profile_targets))
        print(f"📊 Profile targets written to {args.output_profile_targets}")
    elif profile_targets:
        print(f"\n📊 Configs to profile: {', '.join(profile_targets)}")
    
    # Exit with failure if regressions detected
    if has_regression:
        print("\n❌ Performance regression detected - CI should FAIL", file=sys.stderr)
        return 1
    else:
        print("\n✅ No regressions detected")
        return 0


if __name__ == "__main__":
    sys.exit(main())

