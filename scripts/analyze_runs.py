#!/usr/bin/env python3
"""Multi-Run Analytics - Correlation, autocorrelation, and leading indicators.

Features:
- Per-metric correlation matrix
- Lag autocorrelation (detect cyclical patterns)
- Leading indicators (which metrics predict regressions)
- Trend analysis

Output: evidence/summary/trends.{json,md}
"""

import argparse
import json
import math
import pathlib
import sys
from typing import Dict, Any, List, Tuple, Optional

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from _config import get_config


def compute_correlation(x: List[float], y: List[float]) -> float:
    """Compute Pearson correlation coefficient.
    
    Args:
        x: First time series
        y: Second time series
    
    Returns:
        Correlation (-1 to 1)
    """
    if len(x) != len(y) or len(x) < 2:
        return 0.0
    
    n = len(x)
    mean_x = sum(x) / n
    mean_y = sum(y) / n
    
    num = sum((xi - mean_x) * (yi - mean_y) for xi, yi in zip(x, y))
    denom_x = math.sqrt(sum((xi - mean_x) ** 2 for xi in x))
    denom_y = math.sqrt(sum((yi - mean_y) ** 2 for yi in y))
    
    if denom_x < 1e-10 or denom_y < 1e-10:
        return 0.0
    
    return num / (denom_x * denom_y)


def compute_lag_autocorrelation(values: List[float], lag: int = 1) -> float:
    """Compute autocorrelation at given lag.
    
    Args:
        values: Time series
        lag: Lag (default: 1)
    
    Returns:
        Autocorrelation (-1 to 1)
    """
    if len(values) <= lag:
        return 0.0
    
    x = values[:-lag]
    y = values[lag:]
    
    return compute_correlation(x, y)


def compute_correlation_matrix(runs: List[Dict[str, Any]], metrics: List[str]) -> Dict[str, Dict[str, float]]:
    """Compute correlation matrix for all metric pairs.
    
    Args:
        runs: List of run dicts
        metrics: List of metric names
    
    Returns:
        Dict of {metric1: {metric2: corr}}
    """
    matrix = {}
    
    # Extract time series per metric
    series = {}
    for metric in metrics:
        series[metric] = [run.get(metric) for run in runs if run.get(metric) is not None]
    
    # Compute pairwise correlations
    for m1 in metrics:
        matrix[m1] = {}
        for m2 in metrics:
            if len(series[m1]) >= 2 and len(series[m2]) >= 2:
                # Align series (use minimum length)
                min_len = min(len(series[m1]), len(series[m2]))
                x = series[m1][:min_len]
                y = series[m2][:min_len]
                matrix[m1][m2] = compute_correlation(x, y)
            else:
                matrix[m1][m2] = 0.0
    
    return matrix


def find_leading_indicators(runs: List[Dict[str, Any]], metrics: List[str], max_lag: int = 3) -> List[Dict[str, Any]]:
    """Find metrics that lead other metrics (early warning indicators).
    
    Args:
        runs: List of run dicts
        metrics: List of metric names
        max_lag: Maximum lag to check
    
    Returns:
        List of {leader, follower, lag, correlation}
    """
    indicators = []
    
    # Extract time series per metric
    series = {}
    for metric in metrics:
        series[metric] = [run.get(metric) for run in runs if run.get(metric) is not None]
    
    # Check all pairs with lags
    for m1 in metrics:
        for m2 in metrics:
            if m1 == m2:
                continue
            
            if len(series[m1]) < 5 or len(series[m2]) < 5:
                continue
            
            # Try different lags
            for lag in range(1, max_lag + 1):
                if len(series[m1]) <= lag or len(series[m2]) <= lag:
                    continue
                
                # m1 leads m2 by lag steps
                x = series[m1][:-lag]
                y = series[m2][lag:]
                
                min_len = min(len(x), len(y))
                if min_len < 3:
                    continue
                
                corr = compute_correlation(x[:min_len], y[:min_len])
                
                # Consider significant if |corr| >= 0.7
                if abs(corr) >= 0.7:
                    indicators.append({
                        "leader": m1,
                        "follower": m2,
                        "lag": lag,
                        "correlation": corr,
                    })
    
    # Sort by absolute correlation (descending)
    indicators.sort(key=lambda i: abs(i["correlation"]), reverse=True)
    
    return indicators


def compute_epistemic_efficiency(runs: List[Dict[str, Any]], window: int = 10) -> Dict[str, Any]:
    """Compute epistemic efficiency (bits of uncertainty reduced per run).
    
    Measures how much information/learning the system gains per experiment.
    
    Args:
        runs: List of run dicts
        window: Window size for efficiency calculation
    
    Returns:
        Dict with efficiency metrics
    """
    if len(runs) < window:
        return {"efficiency_bits_per_run": 0.0, "total_runs": len(runs)}
    
    # Use entropy_delta_mean as proxy for uncertainty reduction
    recent_runs = runs[-window:]
    entropy_deltas = [run.get("entropy_delta_mean", 0.0) for run in recent_runs if run.get("entropy_delta_mean") is not None]
    
    if not entropy_deltas:
        return {"efficiency_bits_per_run": 0.0, "total_runs": len(runs)}
    
    # Average entropy reduction per run
    avg_entropy_reduction = sum(entropy_deltas) / len(entropy_deltas)
    
    # Normalize to bits (assuming entropy is in [0, 1])
    efficiency = avg_entropy_reduction
    
    return {
        "efficiency_bits_per_run": efficiency,
        "total_runs": len(runs),
        "window": window,
        "interpretation": "High efficiency (>0.1) indicates rapid learning; low (<0.01) indicates plateau"
    }


def generate_trends_markdown(analysis: Dict[str, Any]) -> str:
    """Generate human-readable trends markdown.
    
    Args:
        analysis: Analysis dict
    
    Returns:
        Markdown string
    """
    lines = []
    lines.append("# Multi-Run Analytics & Trends")
    lines.append("")
    lines.append(f"**Generated:** {analysis.get('timestamp', 'unknown')}")
    lines.append(f"**Total Runs:** {analysis.get('total_runs', 0)}")
    lines.append("")
    
    # Leading indicators
    indicators = analysis.get("leading_indicators", [])
    if indicators:
        lines.append("## Leading Indicators")
        lines.append("")
        lines.append("Metrics that change 1-3 runs before regressions:")
        lines.append("")
        
        for ind in indicators[:5]:  # Top 5
            leader = ind["leader"]
            follower = ind["follower"]
            lag = ind["lag"]
            corr = ind["correlation"]
            
            arrow = "→" if corr > 0 else "⇢"
            lines.append(f"- **{leader}** {arrow} **{follower}** (lag {lag}, corr {corr:+.2f})")
        
        lines.append("")
        lines.append("**Interpretation:**")
        lines.append("- Monitor leading indicators for early warning signs")
        lines.append("- Positive correlation: metrics move together")
        lines.append("- Negative correlation: metrics move oppositely")
        lines.append("")
    else:
        lines.append("## Leading Indicators")
        lines.append("")
        lines.append("*No significant leading indicators detected (need 5+ runs)*")
        lines.append("")
    
    # Correlation matrix (top correlations)
    corr_matrix = analysis.get("correlation_matrix", {})
    if corr_matrix:
        lines.append("## Metric Correlations")
        lines.append("")
        
        # Extract top correlations (excluding diagonal)
        top_corrs = []
        for m1, corrs in corr_matrix.items():
            for m2, corr in corrs.items():
                if m1 < m2 and abs(corr) >= 0.5:  # Significant correlations
                    top_corrs.append((m1, m2, corr))
        
        top_corrs.sort(key=lambda t: abs(t[2]), reverse=True)
        
        if top_corrs:
            lines.append("| Metric 1 | Metric 2 | Correlation |")
            lines.append("|----------|----------|-------------|")
            
            for m1, m2, corr in top_corrs[:10]:
                lines.append(f"| {m1} | {m2} | {corr:+.2f} |")
            
            lines.append("")
        else:
            lines.append("*No significant correlations detected*")
            lines.append("")
    
    # Epistemic efficiency
    efficiency = analysis.get("epistemic_efficiency", {})
    if efficiency:
        lines.append("## Epistemic Efficiency")
        lines.append("")
        eff_value = efficiency.get("efficiency_bits_per_run", 0.0)
        lines.append(f"**Efficiency:** {eff_value:.4f} bits/run")
        lines.append(f"**Window:** {efficiency.get('window', 0)} runs")
        lines.append("")
        lines.append(f"*{efficiency.get('interpretation', '')}*")
        lines.append("")
    
    # Footer
    lines.append("---")
    lines.append("")
    lines.append("*Generated by Periodic Labs Multi-Run Analytics*")
    lines.append("")
    
    return "\n".join(lines)


def main() -> int:
    """Analyze multiple runs for trends and correlations.
    
    Returns:
        0 on success
    """
    parser = argparse.ArgumentParser(description="Analyze multiple runs")
    parser.add_argument("--runs-dir", type=pathlib.Path, default="evidence/runs",
                        help="Runs directory")
    parser.add_argument("--output-json", type=pathlib.Path, default="evidence/summary/trends.json",
                        help="Output JSON path")
    parser.add_argument("--output-md", type=pathlib.Path, default="evidence/summary/trends.md",
                        help="Output markdown path")
    args = parser.parse_args()
    
    config = get_config()
    
    print()
    print("=" * 100)
    print("MULTI-RUN ANALYTICS")
    print("=" * 100)
    print()
    
    # Load runs
    print(f"📂 Loading runs from: {args.runs_dir}")
    from baseline_update import load_successful_runs
    runs = load_successful_runs(args.runs_dir, config["DASHBOARD_MAX_RUNS"])
    print(f"   Loaded {len(runs)} runs")
    print()
    
    if len(runs) < 3:
        print("⚠️  Need at least 3 runs for analytics")
        print()
        return 0
    
    # Define metrics to analyze
    metrics = ["coverage", "ece", "brier", "entropy_delta_mean", "accuracy", "loss"]
    
    # Compute correlation matrix
    print("📊 Computing correlation matrix...")
    corr_matrix = compute_correlation_matrix(runs, metrics)
    
    # Find leading indicators
    print("🔍 Finding leading indicators...")
    indicators = find_leading_indicators(runs, metrics, max_lag=3)
    
    # Compute epistemic efficiency
    print("📈 Computing epistemic efficiency...")
    efficiency = compute_epistemic_efficiency(runs, config["EPISTEMIC_EFFICIENCY_WINDOW"])
    
    # Build analysis dict
    analysis = {
        "timestamp": runs[-1].get("timestamp", "unknown") if runs else "unknown",
        "total_runs": len(runs),
        "correlation_matrix": corr_matrix,
        "leading_indicators": indicators,
        "epistemic_efficiency": efficiency,
    }
    
    # Write JSON
    print(f"💾 Writing JSON to: {args.output_json}")
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    with args.output_json.open("w") as f:
        json.dump(analysis, f, indent=2)
    
    # Generate markdown
    print(f"💾 Writing markdown to: {args.output_md}")
    markdown = generate_trends_markdown(analysis)
    args.output_md.write_text(markdown)
    
    # Print summary
    print()
    print("=" * 100)
    print("ANALYTICS SUMMARY")
    print("=" * 100)
    print()
    print(f"Leading indicators: {len(indicators)}")
    if indicators:
        print()
        print("Top 3:")
        for ind in indicators[:3]:
            print(f"  • {ind['leader']} → {ind['follower']} (lag {ind['lag']}, corr {ind['correlation']:+.2f})")
    
    print()
    print(f"Epistemic efficiency: {efficiency['efficiency_bits_per_run']:.4f} bits/run")
    print()
    
    print("✅ Analytics complete!")
    print()
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
