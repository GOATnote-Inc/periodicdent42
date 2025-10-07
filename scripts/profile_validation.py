#!/usr/bin/env python3
"""Profile validation scripts to identify performance bottlenecks.

Wraps validation scripts with py-spy profiling to generate flamegraphs
and detect performance regressions automatically.

Phase 3 Week 10-11: Continuous Profiling

Usage:
    python scripts/profile_validation.py --script validate_stochastic.py --output profile.svg
    python scripts/profile_validation.py --script train_ppo_expert.py --compare baseline_profile.json

Author: GOATnote Autonomous Research Lab Initiative
Date: October 6, 2025
"""

import argparse
import json
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional


@dataclass
class ProfileResult:
    """Result of profiling a script."""
    script_name: str
    duration_seconds: float
    flamegraph_path: Optional[str]
    profile_json_path: Optional[str]
    peak_memory_mb: Optional[float]
    cpu_percent: Optional[float]


@dataclass
class PerformanceRegression:
    """Performance regression detected."""
    metric: str
    current_value: float
    baseline_value: float
    difference: float
    relative_diff: float
    threshold: float
    passed: bool


class ContinuousProfiler:
    """Continuous profiling for performance monitoring."""
    
    def __init__(self, output_dir: Path = Path("artifacts/profiling")):
        """Initialize profiler.
        
        Args:
            output_dir: Directory for profiling artifacts
        """
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def profile_script(
        self,
        script_path: Path,
        args: List[str] = None,
        output_name: str = None,
        generate_flamegraph: bool = True
    ) -> ProfileResult:
        """Profile a Python script with py-spy.
        
        Args:
            script_path: Path to script to profile
            args: Command-line arguments for script
            output_name: Base name for output files
            generate_flamegraph: Whether to generate flamegraph SVG
            
        Returns:
            ProfileResult with profiling details
        """
        if not script_path.exists():
            raise FileNotFoundError(f"Script not found: {script_path}")
        
        # Determine output names
        if output_name is None:
            output_name = script_path.stem
        
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        base_name = f"{output_name}_{timestamp}"
        
        flamegraph_path = None
        profile_json_path = None
        
        # Build py-spy command
        cmd = ["py-spy", "record"]
        
        if generate_flamegraph:
            flamegraph_path = self.output_dir / f"{base_name}.svg"
            cmd.extend(["--format", "flamegraph"])
            cmd.extend(["--output", str(flamegraph_path)])
        
        # Also save raw profile data
        profile_json_path = self.output_dir / f"{base_name}.json"
        
        # Add script and args
        cmd.append("--")
        cmd.append("python")
        cmd.append(str(script_path))
        if args:
            cmd.extend(args)
        
        # Run profiling
        print(f"🔍 Profiling: {script_path.name}")
        print(f"   Command: {' '.join(cmd)}")
        
        start_time = time.time()
        
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=True
            )
            duration = time.time() - start_time
            
            print(f"✅ Profiling complete: {duration:.2f}s")
            
            # Save profile metadata
            metadata = {
                "script": str(script_path),
                "duration_seconds": duration,
                "timestamp": timestamp,
                "flamegraph": str(flamegraph_path) if flamegraph_path else None,
                "args": args or []
            }
            
            with open(profile_json_path, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            return ProfileResult(
                script_name=script_path.name,
                duration_seconds=duration,
                flamegraph_path=str(flamegraph_path) if flamegraph_path else None,
                profile_json_path=str(profile_json_path),
                peak_memory_mb=None,  # py-spy doesn't provide this directly
                cpu_percent=None
            )
            
        except subprocess.CalledProcessError as e:
            print(f"❌ Profiling failed: {e}")
            print(f"   stdout: {e.stdout}")
            print(f"   stderr: {e.stderr}")
            raise
    
    def compare_performance(
        self,
        current_profile: Path,
        baseline_profile: Path,
        threshold: float = 0.10  # 10% regression threshold
    ) -> List[PerformanceRegression]:
        """Compare current performance against baseline.
        
        Args:
            current_profile: Path to current profile JSON
            baseline_profile: Path to baseline profile JSON
            threshold: Regression threshold (0.10 = 10% slower is regression)
            
        Returns:
            List of detected regressions
        """
        # Load profiles
        with open(current_profile) as f:
            current = json.load(f)
        
        with open(baseline_profile) as f:
            baseline = json.load(f)
        
        regressions = []
        
        # Compare duration
        current_dur = current.get("duration_seconds", 0)
        baseline_dur = baseline.get("duration_seconds", 0)
        
        if baseline_dur > 0:
            diff = current_dur - baseline_dur
            rel_diff = diff / baseline_dur
            passed = rel_diff <= threshold
            
            regressions.append(PerformanceRegression(
                metric="duration_seconds",
                current_value=current_dur,
                baseline_value=baseline_dur,
                difference=diff,
                relative_diff=rel_diff,
                threshold=threshold,
                passed=passed
            ))
        
        return regressions
    
    def generate_report(
        self,
        result: ProfileResult,
        regressions: Optional[List[PerformanceRegression]] = None
    ) -> str:
        """Generate human-readable profiling report.
        
        Args:
            result: ProfileResult to report
            regressions: Optional performance regressions
            
        Returns:
            Formatted report string
        """
        lines = []
        lines.append("=" * 80)
        lines.append("CONTINUOUS PROFILING REPORT")
        lines.append("=" * 80)
        lines.append("")
        lines.append(f"Script: {result.script_name}")
        lines.append(f"Duration: {result.duration_seconds:.2f}s")
        lines.append("")
        
        if result.flamegraph_path:
            lines.append(f"Flamegraph: {result.flamegraph_path}")
            lines.append("  Open in browser to visualize performance bottlenecks")
            lines.append("")
        
        if result.profile_json_path:
            lines.append(f"Profile Data: {result.profile_json_path}")
            lines.append("")
        
        if regressions:
            passed_all = all(r.passed for r in regressions)
            status = "✅ NO REGRESSIONS" if passed_all else "❌ REGRESSIONS DETECTED"
            lines.append(f"Performance Check: {status}")
            lines.append("-" * 80)
            
            for reg in regressions:
                if not reg.passed:
                    lines.append(f"  ❌ {reg.metric}:")
                    lines.append(f"     Current:  {reg.current_value:.3f}")
                    lines.append(f"     Baseline: {reg.baseline_value:.3f}")
                    lines.append(f"     Slower:   {reg.relative_diff:.1%} (threshold: {reg.threshold:.1%})")
                    lines.append("")
                else:
                    lines.append(f"  ✅ {reg.metric}: {reg.relative_diff:+.1%} (within {reg.threshold:.1%})")
        
        lines.append("")
        lines.append("=" * 80)
        
        return "\n".join(lines)


def main():
    """Main entry point for CLI."""
    parser = argparse.ArgumentParser(
        description="Profile Python scripts with py-spy and detect performance regressions"
    )
    parser.add_argument(
        "--script",
        type=Path,
        required=True,
        help="Path to Python script to profile"
    )
    parser.add_argument(
        "--args",
        nargs="*",
        help="Arguments to pass to script"
    )
    parser.add_argument(
        "--output",
        type=str,
        help="Base name for output files"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/profiling"),
        help="Directory for profiling artifacts"
    )
    parser.add_argument(
        "--compare",
        type=Path,
        help="Baseline profile JSON to compare against"
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.10,
        help="Performance regression threshold (default: 0.10 = 10%%)"
    )
    parser.add_argument(
        "--no-flamegraph",
        action="store_true",
        help="Skip flamegraph generation"
    )
    parser.add_argument(
        "--fail-on-regression",
        action="store_true",
        help="Exit with non-zero code if regression detected"
    )
    
    args = parser.parse_args()
    
    # Check script exists
    if not args.script.exists():
        print(f"ERROR: Script not found: {args.script}", file=sys.stderr)
        sys.exit(1)
    
    # Create profiler
    profiler = ContinuousProfiler(output_dir=args.output_dir)
    
    # Run profiling
    try:
        result = profiler.profile_script(
            args.script,
            args=args.args,
            output_name=args.output,
            generate_flamegraph=not args.no_flamegraph
        )
    except Exception as e:
        print(f"ERROR: Profiling failed: {e}", file=sys.stderr)
        sys.exit(1)
    
    # Compare against baseline if provided
    regressions = None
    if args.compare:
        if not args.compare.exists():
            print(f"ERROR: Baseline not found: {args.compare}", file=sys.stderr)
            sys.exit(1)
        
        if result.profile_json_path:
            regressions = profiler.compare_performance(
                Path(result.profile_json_path),
                args.compare,
                threshold=args.threshold
            )
    
    # Print report
    report = profiler.generate_report(result, regressions)
    print(report)
    
    # Exit with appropriate code
    if args.fail_on_regression and regressions:
        if not all(r.passed for r in regressions):
            sys.exit(1)
    
    sys.exit(0)


if __name__ == "__main__":
    main()
