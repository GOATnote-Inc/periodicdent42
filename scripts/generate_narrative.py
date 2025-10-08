#!/usr/bin/env python3
"""Regression Narrative Generator - Automated explanations of regressions.

Features:
- Plain-English summary of regressions
- Most-impacted metrics and modules
- Likely causes (based on patterns)
- Next validation steps
- Confidence score (based on z-scores)
- Relative entropy (information loss/gain)

Output: evidence/regressions/regression_narrative.md
"""

import argparse
import json
import math
import pathlib
import sys
from datetime import datetime
from typing import Dict, Any, List, Optional

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from _config import get_config


def compute_relative_entropy(p_values: List[float], q_values: List[float]) -> float:
    """Compute relative entropy (KL divergence) between two distributions.
    
    Args:
        p_values: Baseline distribution
        q_values: Current distribution
    
    Returns:
        KL divergence in bits (0 = identical)
    """
    if len(p_values) != len(q_values) or not p_values:
        return 0.0
    
    # Normalize to probability distributions
    p_sum = sum(p_values)
    q_sum = sum(q_values)
    
    if p_sum <= 0 or q_sum <= 0:
        return 0.0
    
    p = [v / p_sum for v in p_values]
    q = [v / q_sum for v in q_values]
    
    kl_div = 0.0
    for pi, qi in zip(p, q):
        if pi > 1e-10 and qi > 1e-10:
            kl_div += pi * math.log2(pi / qi)
    
    return abs(kl_div)


def compute_confidence(regressions: List[Dict[str, Any]]) -> float:
    """Compute confidence score for regression narrative.
    
    Based on ensemble of z-scores (higher z = higher confidence).
    
    Args:
        regressions: List of regression dicts
    
    Returns:
        Confidence score (0 to 1)
    """
    if not regressions:
        return 0.0
    
    # Average of absolute z-scores, normalized to 0-1
    z_scores = [abs(reg.get("z_score", 0.0)) for reg in regressions]
    avg_z = sum(z_scores) / len(z_scores)
    
    # Sigmoid normalization: confidence = 1 / (1 + exp(-z/2))
    confidence = 1.0 / (1.0 + math.exp(-avg_z / 2.0))
    
    return min(1.0, max(0.0, confidence))


def identify_likely_cause(regressions: List[Dict[str, Any]], baseline: Dict[str, Any]) -> str:
    """Identify likely cause based on regression patterns.
    
    Args:
        regressions: List of regression dicts
        baseline: Baseline dict
    
    Returns:
        Likely cause string
    """
    if not regressions:
        return "Unknown"
    
    # Pattern matching
    metrics_affected = {reg["metric"] for reg in regressions}
    
    # Calibration drift (ece + brier)
    if "ece" in metrics_affected and "brier" in metrics_affected:
        return "Calibration drift (likely model re-training or temperature scaling change)"
    
    # Coverage drop alone
    if "coverage" in metrics_affected and len(metrics_affected) == 1:
        return "Test coverage regression (new code paths without tests)"
    
    # Entropy increase
    if "entropy_delta_mean" in metrics_affected:
        return "Increased prediction uncertainty (model confidence degradation)"
    
    # Multiple metrics (likely systematic)
    if len(metrics_affected) >= 3:
        return "Systematic regression (likely dependency bump or architectural change)"
    
    return "Isolated metric drift (see specific metrics for details)"


def suggest_next_validation(regressions: List[Dict[str, Any]]) -> List[str]:
    """Suggest next validation steps based on regressions.
    
    Args:
        regressions: List of regression dicts
    
    Returns:
        List of validation step strings
    """
    steps = []
    metrics_affected = {reg["metric"] for reg in regressions}
    
    if "coverage" in metrics_affected:
        steps.append("Add tests for new code paths (aim for ≥85% coverage)")
    
    if "ece" in metrics_affected or "brier" in metrics_affected:
        steps.append("Re-calibrate model with temperature scaling (target ECE < 0.15)")
        steps.append("Verify calibration on held-out test set v1.3")
    
    if "entropy_delta_mean" in metrics_affected:
        steps.append("Audit model confidence predictions (check for systematic shifts)")
    
    if "accuracy" in metrics_affected or "loss" in metrics_affected:
        steps.append("Re-train with more data or adjust hyperparameters")
    
    if not steps:
        steps.append("Review recent code changes (git diff)")
        steps.append("Verify dataset contracts (make validate)")
        steps.append("Re-run with fixed seed (--seed 42) for reproducibility")
    
    return steps


def generate_summary(regressions: List[Dict[str, Any]]) -> str:
    """Generate one-line summary of regressions.
    
    Args:
        regressions: List of regression dicts
    
    Returns:
        Summary string (e.g., "coverage ↓17%, ece ↑13%")
    """
    if not regressions:
        return "No regressions detected"
    
    # Show top 3 most severe regressions
    sorted_regs = sorted(regressions, key=lambda r: abs(r.get("z_score", 0.0)), reverse=True)[:3]
    
    parts = []
    for reg in sorted_regs:
        metric = reg["metric"]
        delta = reg["delta"]
        pct = delta * 100
        arrow = "↑" if delta > 0 else "↓"
        parts.append(f"{metric} {arrow}{abs(pct):.0f}%")
    
    return ", ".join(parts)


def generate_narrative(
    regression_report: Dict[str, Any],
    baseline: Dict[str, Any],
    recent_runs: List[Dict[str, Any]],
    config: Dict[str, Any]
) -> str:
    """Generate regression narrative in Markdown.
    
    Args:
        regression_report: Regression report dict
        baseline: Baseline dict
        recent_runs: Recent K runs for context
        config: Config dict
    
    Returns:
        Markdown narrative string
    """
    regressions = regression_report.get("regressions", [])
    waivers = regression_report.get("waivers_applied", [])
    
    # Compute metrics
    summary = generate_summary(regressions)
    confidence = compute_confidence(regressions)
    likely_cause = identify_likely_cause(regressions, baseline)
    next_steps = suggest_next_validation(regressions)
    
    # Compute relative entropy (information loss)
    if recent_runs and len(recent_runs) >= 2:
        # Use last 5 runs vs current as distributions
        recent_coverage = [r.get("coverage", 0.87) for r in recent_runs[-5:]]
        recent_ece = [r.get("ece", 0.12) for r in recent_runs[-5:]]
        
        current_coverage = regressions[0].get("current", 0.87) if regressions and "coverage" in {r["metric"] for r in regressions} else 0.87
        current_ece = regressions[0].get("current", 0.12) if regressions and "ece" in {r["metric"] for r in regressions} else 0.12
        
        rel_entropy = compute_relative_entropy(
            recent_coverage + recent_ece,
            [current_coverage] * 5 + [current_ece] * 5
        )
    else:
        rel_entropy = 0.0
    
    # Build narrative
    lines = []
    lines.append("# Regression Narrative")
    lines.append("")
    lines.append(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')}")
    lines.append(f"**Git SHA:** `{regression_report.get('git_sha', 'unknown')}`")
    lines.append(f"**CI Run:** `{regression_report.get('ci_run_id', 'unknown')}`")
    lines.append("")
    
    # Summary
    lines.append("## Summary")
    lines.append("")
    lines.append(f"**Regression:** {summary}")
    lines.append(f"**Confidence:** {confidence:.2f} (based on z-score ensemble)")
    lines.append(f"**Information Loss:** {rel_entropy:.4f} bits (relative entropy)")
    lines.append("")
    
    # Likely cause
    lines.append("## Likely Cause")
    lines.append("")
    lines.append(f"**Hypothesis:** {likely_cause}")
    lines.append("")
    
    # Most-impacted metrics
    if regressions:
        lines.append("## Most-Impacted Metrics")
        lines.append("")
        lines.append("| Metric | Baseline | Current | Δ | z-score | Impact |")
        lines.append("|--------|----------|---------|---|---------|--------|")
        
        for reg in sorted(regressions, key=lambda r: abs(r.get("z_score", 0.0)), reverse=True):
            metric = reg["metric"]
            baseline_val = reg["baseline_mean"]
            current_val = reg["current"]
            delta = reg["delta"]
            z = reg["z_score"]
            
            if abs(z) >= 5.0:
                impact = "🔴 Critical"
            elif abs(z) >= 3.0:
                impact = "🟡 High"
            else:
                impact = "🟢 Medium"
            
            lines.append(f"| {metric} | {baseline_val:.4f} | {current_val:.4f} | {delta:+.4f} | {z:+.2f} | {impact} |")
        
        lines.append("")
    
    # Next validation steps
    lines.append("## Next Validation Steps")
    lines.append("")
    for i, step in enumerate(next_steps, 1):
        lines.append(f"{i}. {step}")
    lines.append("")
    
    # Waivers (if any)
    if waivers:
        lines.append("## Active Waivers")
        lines.append("")
        for waiver in waivers:
            lines.append(f"- **{waiver['metric']}**: {waiver['waiver_reason']}")
            lines.append(f"  - Expires: {waiver['waiver_expires']}")
        lines.append("")
    
    # Recommendations
    lines.append("## Recommendations")
    lines.append("")
    if confidence >= config["NARRATIVE_CONFIDENCE_THRESHOLD"]:
        lines.append(f"✅ High confidence ({confidence:.2f}) in regression detection. Recommend blocking merge.")
    else:
        lines.append(f"⚠️ Moderate confidence ({confidence:.2f}). Consider manual review before blocking.")
    
    if rel_entropy > 0.5:
        lines.append(f"⚠️ High information loss ({rel_entropy:.2f} bits). System behavior has changed significantly.")
    
    lines.append("")
    
    # Footer
    lines.append("---")
    lines.append("")
    lines.append("*Generated by Periodic Labs Regression Detection System*")
    lines.append("")
    
    return "\n".join(lines)


def main() -> int:
    """Generate regression narrative.
    
    Returns:
        0 on success
    """
    parser = argparse.ArgumentParser(description="Generate regression narrative")
    parser.add_argument("--regression-report", type=pathlib.Path,
                        default="evidence/regressions/regression_report.json",
                        help="Regression report JSON")
    parser.add_argument("--baseline", type=pathlib.Path,
                        default="evidence/baselines/rolling_baseline.json",
                        help="Baseline JSON")
    parser.add_argument("--runs-dir", type=pathlib.Path,
                        default="evidence/runs",
                        help="Runs directory")
    parser.add_argument("--output", type=pathlib.Path,
                        default="evidence/regressions/regression_narrative.md",
                        help="Output narrative markdown")
    args = parser.parse_args()
    
    config = get_config()
    
    print()
    print("=" * 100)
    print("REGRESSION NARRATIVE GENERATOR")
    print("=" * 100)
    print()
    
    # Load regression report
    if not args.regression_report.exists():
        print(f"⚠️  Regression report not found: {args.regression_report}")
        print("   Run 'make detect' first")
        print()
        return 0
    
    with args.regression_report.open() as f:
        regression_report = json.load(f)
    
    # Load baseline
    baseline = {}
    if args.baseline.exists():
        with args.baseline.open() as f:
            baseline = json.load(f)
    
    # Load recent runs
    recent_runs = []
    if args.runs_dir.exists():
        from baseline_update import load_successful_runs
        recent_runs = load_successful_runs(args.runs_dir, 20)
    
    # Generate narrative
    print("📝 Generating narrative...")
    narrative = generate_narrative(regression_report, baseline, recent_runs, config)
    
    # Write output
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(narrative)
    
    print(f"💾 Narrative written to: {args.output}")
    print()
    
    # Print preview
    print("Preview:")
    print("-" * 100)
    preview_lines = narrative.split("\n")[:20]
    print("\n".join(preview_lines))
    if len(narrative.split("\n")) > 20:
        print("...")
    print("-" * 100)
    print()
    
    print("✅ Narrative generation complete!")
    print()
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
