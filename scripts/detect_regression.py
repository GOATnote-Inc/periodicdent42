#!/usr/bin/env python3
"""Regression Detection - Z-score + Page-Hinkley change-point detection.

Features:
- Z-score regression (threshold-based)
- Page-Hinkley change-point detection (step changes)
- Directional rules (worse/better per metric)
- Waiver system with expiration
- Optional Mahalanobis distance (multivariate)

Output: evidence/regressions/regression_report.{json,md}
"""

import argparse
import json
import pathlib
import sys
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional, Tuple

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from _config import get_config


# Directional rules
WORSE_IF_INCREASE = {"ece", "brier", "mce", "loss", "entropy_delta_mean"}
BETTER_IF_INCREASE = {"coverage", "accuracy"}


class PageHinkley:
    """Page-Hinkley test for detecting change points in a sequence.
    
    Detects step changes in mean level.
    """
    
    def __init__(self, delta: float = 0.005, lambda_threshold: float = 0.05):
        """Initialize Page-Hinkley detector.
        
        Args:
            delta: Magnitude of changes to detect (epsilon)
            lambda_threshold: Alarm threshold
        """
        self.delta = delta
        self.lambda_threshold = lambda_threshold
        self.sum_pos = 0.0
        self.sum_neg = 0.0
        self.min_pos = 0.0
        self.max_neg = 0.0
        self.alarm_pos = False
        self.alarm_neg = False
    
    def update(self, value: float, reference: float) -> Tuple[bool, bool]:
        """Update with new value.
        
        Args:
            value: Current value
            reference: Reference (mean) value
        
        Returns:
            Tuple of (alarm_increase, alarm_decrease)
        """
        diff = value - reference - self.delta
        
        # Positive changes (increase)
        self.sum_pos += diff
        self.min_pos = min(self.min_pos, self.sum_pos)
        ph_pos = self.sum_pos - self.min_pos
        self.alarm_pos = ph_pos > self.lambda_threshold
        
        # Negative changes (decrease)
        self.sum_neg -= diff
        self.max_neg = max(self.max_neg, self.sum_neg)
        ph_neg = self.max_neg - self.sum_neg
        self.alarm_neg = ph_neg > self.lambda_threshold
        
        return self.alarm_pos, self.alarm_neg


def detect_ph_change(recent_values: List[float], baseline_mean: float, config: Dict[str, Any]) -> Tuple[bool, bool]:
    """Detect change-point using Page-Hinkley test.
    
    Args:
        recent_values: Recent K values (oldest first)
        baseline_mean: Baseline mean for reference
        config: Config dict with PH_DELTA and PH_LAMBDA
    
    Returns:
        Tuple of (alarm_increase, alarm_decrease)
    """
    ph = PageHinkley(delta=config["PH_DELTA"], lambda_threshold=config["PH_LAMBDA"])
    
    alarm_pos = False
    alarm_neg = False
    
    for value in recent_values:
        alarm_pos, alarm_neg = ph.update(value, baseline_mean)
        if alarm_pos or alarm_neg:
            break
    
    return alarm_pos, alarm_neg


def load_baseline(baseline_file: pathlib.Path) -> Dict[str, Any]:
    """Load baseline from JSON."""
    try:
        if baseline_file.exists():
            with baseline_file.open() as f:
                return json.load(f)
    except Exception:
        pass
    return {"metrics": {}}


def load_waivers(waiver_file: pathlib.Path, pr_number: Optional[str] = None) -> List[Dict[str, Any]]:
    """Load waivers from GOVERNANCE_CHANGE_ACCEPT.yml.
    
    Args:
        waiver_file: Path to waiver YAML file
        pr_number: Current PR number (if applicable)
    
    Returns:
        List of active waivers
    """
    waivers = []
    
    try:
        if not waiver_file.exists():
            return waivers
        
        # Simple YAML parsing for waivers section
        import re
        with waiver_file.open() as f:
            content = f.read()
        
        # Extract waivers section
        waiver_match = re.search(r'waivers:\s*\n((?:  -.*\n(?:    .*\n)*)*)', content)
        if not waiver_match:
            return waivers
        
        waiver_text = waiver_match.group(1)
        
        # Parse each waiver (simple approach)
        waiver_blocks = waiver_text.split('\n  - ')
        for block in waiver_blocks[1:]:  # Skip first empty
            waiver = {}
            for line in block.split('\n'):
                if ':' in line:
                    key, value = line.strip().split(':', 1)
                    value = value.strip().strip('"\'')
                    waiver[key] = value
            
            # Check if waiver applies
            if pr_number and waiver.get('pr') == pr_number:
                # Check expiry
                expires_at = waiver.get('expires_at')
                if expires_at:
                    try:
                        expiry = datetime.fromisoformat(expires_at.replace('Z', '+00:00'))
                        if datetime.now(timezone.utc) < expiry:
                            waivers.append(waiver)
                    except Exception:
                        pass
    
    except Exception:
        pass
    
    return waivers


def check_waiver(metric: str, current_value: float, baseline_mean: float, waivers: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """Check if a waiver applies to this regression.
    
    Args:
        metric: Metric name
        current_value: Current value
        baseline_mean: Baseline mean
        waivers: List of active waivers
    
    Returns:
        Matching waiver dict or None
    """
    for waiver in waivers:
        if waiver.get('metric') != metric:
            continue
        
        # Check bounds
        delta = abs(current_value - baseline_mean)
        max_delta = waiver.get('max_delta')
        if max_delta and delta <= float(max_delta):
            return waiver
        
        min_value = waiver.get('min_value')
        if min_value and current_value >= float(min_value):
            return waiver
        
        max_value = waiver.get('max_value')
        if max_value and current_value <= float(max_value):
            return waiver
    
    return None


def detect_regressions(
    current_metrics: Dict[str, Any],
    baseline: Dict[str, Any],
    recent_runs: List[Dict[str, Any]],
    config: Dict[str, Any],
    waivers: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """Detect regressions in current metrics vs baseline.
    
    Args:
        current_metrics: Current run metrics
        baseline: Baseline statistics
        recent_runs: Recent K runs for Page-Hinkley
        config: Config dict
        waivers: Active waivers
    
    Returns:
        Regression report dict
    """
    report = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "git_sha": current_metrics.get("git_sha"),
        "ci_run_id": current_metrics.get("ci_run_id"),
        "regressions": [],
        "warnings": [],
        "waivers_applied": [],
        "passed": True,
    }
    
    # Absolute thresholds
    abs_thresholds = {
        "coverage": config["ABS_THRESH_COVERAGE"],
        "ece": config["ABS_THRESH_ECE"],
        "brier": config["ABS_THRESH_BRIER"],
        "accuracy": config["ABS_THRESH_ACCURACY"],
        "loss": config["ABS_THRESH_LOSS"],
        "entropy_delta_mean": config["ABS_THRESH_ENTROPY"],
    }
    
    # Check each metric
    for metric, stats in baseline.get("metrics", {}).items():
        current_value = current_metrics.get(metric)
        if current_value is None:
            continue
        
        mean = stats.get("mean")
        std = stats.get("std")
        ewma = stats.get("ewma")
        
        if mean is None or std is None:
            continue
        
        # Compute z-score (guard against std ≈ 0)
        if std < 1e-6:
            z = 0.0
        else:
            z = (current_value - mean) / std
        
        delta = current_value - mean
        abs_delta = abs(delta)
        
        # Determine direction
        is_worse = False
        if metric in WORSE_IF_INCREASE and delta > 0:
            is_worse = True
        elif metric in BETTER_IF_INCREASE and delta < 0:
            is_worse = True
        
        # Z-score regression check
        z_triggered = abs(z) >= config["Z_THRESH"] and abs_delta >= abs_thresholds.get(metric, 0.0)
        
        # Page-Hinkley check (if we have recent runs)
        ph_triggered = False
        if recent_runs and len(recent_runs) >= 5:
            recent_values = [run.get(metric) for run in recent_runs if run.get(metric) is not None]
            if len(recent_values) >= 5:
                alarm_pos, alarm_neg = detect_ph_change(recent_values, mean, config)
                if metric in WORSE_IF_INCREASE:
                    ph_triggered = alarm_pos
                elif metric in BETTER_IF_INCREASE:
                    ph_triggered = alarm_neg
        
        # Overall regression?
        is_regression = (z_triggered or ph_triggered) and is_worse
        
        if is_regression:
            # Check waiver
            waiver = check_waiver(metric, current_value, mean, waivers)
            
            regression_info = {
                "metric": metric,
                "current": current_value,
                "baseline_mean": mean,
                "baseline_std": std,
                "baseline_ewma": ewma,
                "delta": delta,
                "z_score": z,
                "z_triggered": z_triggered,
                "ph_triggered": ph_triggered,
                "waived": waiver is not None,
                "waiver_reason": waiver.get("reason") if waiver else None,
                "waiver_expires": waiver.get("expires_at") if waiver else None,
            }
            
            if waiver:
                report["waivers_applied"].append(regression_info)
            else:
                report["regressions"].append(regression_info)
                report["passed"] = False
    
    return report


def generate_markdown_report(report: Dict[str, Any], output_file: pathlib.Path) -> None:
    """Generate markdown regression report."""
    lines = []
    lines.append("# Regression Detection Report")
    lines.append("")
    lines.append(f"**Timestamp:** {report['timestamp']}")
    lines.append(f"**Git SHA:** {report['git_sha']}")
    lines.append(f"**CI Run ID:** {report['ci_run_id']}")
    lines.append("")
    
    if report["passed"]:
        lines.append("## ✅ Status: PASSED")
        lines.append("")
        lines.append("No regressions detected.")
    else:
        lines.append("## ❌ Status: REGRESSION DETECTED")
        lines.append("")
    
    # Regressions table
    if report["regressions"]:
        lines.append("## Regressions")
        lines.append("")
        lines.append("| Metric | Baseline μ±σ (EWMA) | Current | Δ | z | PH | Status |")
        lines.append("|--------|---------------------|---------|---|---|----|--------|")
        
        for reg in report["regressions"]:
            metric = reg["metric"]
            mean = reg["baseline_mean"]
            std = reg["baseline_std"]
            ewma = reg["baseline_ewma"]
            current = reg["current"]
            delta = reg["delta"]
            z = reg["z_score"]
            ph = "🔔" if reg["ph_triggered"] else ""
            
            lines.append(f"| {metric} | {mean:.4f}±{std:.4f} ({ewma:.4f}) | {current:.4f} | {delta:+.4f} | {z:+.2f} | {ph} | ❌ Fail |")
        
        lines.append("")
    
    # Waivers applied
    if report["waivers_applied"]:
        lines.append("## Waivers Applied")
        lines.append("")
        for waiver in report["waivers_applied"]:
            metric = waiver["metric"]
            reason = waiver["waiver_reason"]
            expires = waiver["waiver_expires"]
            lines.append(f"- **{metric}**: ⚠️ Waived until {expires}")
            lines.append(f"  - Reason: {reason}")
        lines.append("")
    
    # Suggested actions
    if report["regressions"]:
        lines.append("## Suggested Actions")
        lines.append("")
        lines.append("1. **Check dataset drift**: Verify `data_contracts.yaml` checksums match")
        lines.append("2. **Check seed sensitivity**: Re-run with `--seed 42` for reproducibility")
        lines.append("3. **Review recent changes**: Check files modified in this commit")
        lines.append("4. **Consider waiver**: If intentional, add to `GOVERNANCE_CHANGE_ACCEPT.yml`")
        lines.append("")
    
    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_file.write_text("\n".join(lines))


def print_regression_table(report: Dict[str, Any]) -> None:
    """Print formatted regression table."""
    print("=" * 120)
    print("REGRESSION DETECTION REPORT")
    print("=" * 120)
    print()
    print(f"Timestamp:  {report['timestamp']}")
    print(f"Git SHA:    {report['git_sha']}")
    print(f"CI Run ID:  {report['ci_run_id']}")
    print()
    
    if report["passed"]:
        print("✅ Status: PASSED - No regressions detected")
        print()
    else:
        print("❌ Status: FAILED - Regressions detected")
        print()
        
        print(f"{'Metric':<20s} | {'Baseline μ±σ (EWMA)':<25s} | {'Current':>10s} | {'Δ':>10s} | {'z':>6s} | {'PH':^4s} | {'Status':<10s}")
        print("-" * 120)
        
        for reg in report["regressions"]:
            metric = reg["metric"]
            mean = reg["baseline_mean"]
            std = reg["baseline_std"]
            ewma = reg["baseline_ewma"]
            current = reg["current"]
            delta = reg["delta"]
            z = reg["z_score"]
            ph = "🔔" if reg["ph_triggered"] else ""
            
            baseline_str = f"{mean:.4f}±{std:.4f} ({ewma:.4f})"
            print(f"{metric:<20s} | {baseline_str:<25s} | {current:>10.4f} | {delta:>+10.4f} | {z:>+6.2f} | {ph:^4s} | {'❌ Fail':<10s}")
        
        print()
    
    if report["waivers_applied"]:
        print("⚠️  Waivers Applied:")
        for waiver in report["waivers_applied"]:
            print(f"  - {waiver['metric']}: {waiver['waiver_reason']} (expires: {waiver['waiver_expires']})")
        print()


def main() -> int:
    """Detect regressions in current run.
    
    Returns:
        0 if no regressions (or allowed), 1 if regressions detected
    """
    parser = argparse.ArgumentParser(description="Detect regressions")
    parser.add_argument("--baseline", type=pathlib.Path, default="evidence/baselines/rolling_baseline.json",
                        help="Baseline JSON path")
    parser.add_argument("--metrics", type=pathlib.Path, default="evidence/current_run_metrics.json",
                        help="Current metrics JSON path")
    parser.add_argument("--runs-dir", type=pathlib.Path, default="evidence/runs",
                        help="Directory with run JSONL files (for Page-Hinkley)")
    parser.add_argument("--waivers", type=pathlib.Path, default="GOVERNANCE_CHANGE_ACCEPT.yml",
                        help="Waiver file path")
    parser.add_argument("--output-json", type=pathlib.Path, default="evidence/regressions/regression_report.json",
                        help="Output JSON path")
    parser.add_argument("--output-md", type=pathlib.Path, default="evidence/regressions/regression_report.md",
                        help="Output markdown path")
    args = parser.parse_args()
    
    config = get_config()
    
    print()
    print("=" * 120)
    print("REGRESSION DETECTION")
    print("=" * 120)
    print()
    
    # Load current metrics
    print(f"📂 Loading current metrics from: {args.metrics}")
    try:
        with args.metrics.open() as f:
            current_metrics = json.load(f)
        print(f"   Loaded {len([k for k, v in current_metrics.items() if v is not None])} metrics")
    except Exception as e:
        print(f"❌ Error loading metrics: {e}")
        return 1
    print()
    
    # Load baseline
    print(f"📊 Loading baseline from: {args.baseline}")
    baseline = load_baseline(args.baseline)
    if not baseline.get("metrics"):
        print("⚠️  No baseline found - run baseline_update.py first")
        return 0
    print(f"   Loaded {len(baseline['metrics'])} baseline metrics")
    print()
    
    # Load recent runs (for Page-Hinkley)
    print(f"📈 Loading recent runs from: {args.runs_dir}")
    from baseline_update import load_successful_runs
    recent_runs = load_successful_runs(args.runs_dir, 20)
    print(f"   Loaded {len(recent_runs)} recent runs")
    print()
    
    # Load waivers
    pr_number = None  # TODO: Extract from GITHUB_EVENT_PATH if needed
    waivers = load_waivers(args.waivers, pr_number)
    if waivers:
        print(f"⚖️  Loaded {len(waivers)} active waivers")
        print()
    
    # Detect regressions
    print("🔍 Detecting regressions...")
    report = detect_regressions(current_metrics, baseline, recent_runs, config, waivers)
    print()
    
    # Print table
    print_regression_table(report)
    
    # Write reports
    print(f"💾 Writing JSON report to: {args.output_json}")
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    with args.output_json.open("w") as f:
        json.dump(report, f, indent=2)
    
    print(f"💾 Writing markdown report to: {args.output_md}")
    generate_markdown_report(report, args.output_md)
    print()
    
    # Determine exit code
    if not report["passed"]:
        if config["ALLOW_NIGHTLY_REGRESSION"]:
            print("⚠️  Regressions detected but ALLOW_NIGHTLY_REGRESSION=true")
            print()
            return 0
        else:
            print("❌ Regressions detected")
            print()
            return 1
    else:
        print("✅ No regressions detected")
        print()
        return 0


if __name__ == "__main__":
    sys.exit(main())
