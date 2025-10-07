#!/usr/bin/env python3
"""Generate epistemic CI metrics and human-readable report.

Computes information-theoretic and practical metrics from test selection results.
Emits experiment ledger for uncertainty telemetry tracking.

Usage:
    python scripts/gen_ci_report.py
"""

import argparse
import hashlib
import json
import os
import pathlib
import subprocess
import sys
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional


def compute_metrics(
    selected: List[Dict[str, Any]],
    all_tests: List[Dict[str, Any]],
    selection_stats: Dict[str, Any]
) -> Dict[str, Any]:
    """Compute epistemic and practical CI metrics.
    
    Args:
        selected: Selected tests
        all_tests: All available tests
        selection_stats: Stats from test selection
        
    Returns:
        Metrics dict
    """
    # Selected test stats
    selected_time = sum(t.get("duration_sec", 0) for t in selected)
    selected_cost = sum(t.get("cost_usd", 0) for t in selected)
    selected_eig = sum(t.get("eig_bits", 0) for t in selected)
    
    # Full suite stats
    full_time = sum(t.get("duration_sec", 0) for t in all_tests)
    full_cost = sum(t.get("cost_usd", 0) for t in all_tests)
    full_eig = sum(t.get("eig_bits", 0) for t in all_tests)
    
    # Savings
    time_saved = full_time - selected_time
    cost_saved = full_cost - selected_cost
    
    # Failures caught (estimate based on model uncertainty)
    selected_failures_est = sum(
        t.get("model_uncertainty", 0.1) for t in selected
    )
    full_failures_est = sum(
        t.get("model_uncertainty", 0.1) for t in all_tests
    )
    
    # Detection rate
    detection_rate = selected_failures_est / max(full_failures_est, 0.001)
    
    # Information efficiency
    bits_per_dollar = selected_eig / max(selected_cost, 0.001)
    bits_per_second = selected_eig / max(selected_time, 0.001)
    
    # Entropy reduction (if available)
    entropy_before = sum(t.get("entropy_before", 0) for t in selected)
    entropy_after = sum(t.get("entropy_after", 0) for t in selected)
    delta_entropy = entropy_before - entropy_after if entropy_before > 0 else None

    budget_sec = selection_stats.get("budget_sec", 0)
    budget_usd = selection_stats.get("budget_usd", 0)
    budget_util_time = selected_time / budget_sec if budget_sec else None
    budget_util_cost = selected_cost / budget_usd if budget_usd else None

    decision_rationale = selection_stats.get("decision_rationale") or selection_stats.get(
        "policy_reason"
    ) or "max_eig_under_budget"
    
    metrics = {
        "run_time_saved_sec": time_saved,
        "run_cost_saved_usd": cost_saved,
        "time_reduction_pct": (time_saved / max(full_time, 0.001)) * 100,
        "cost_reduction_pct": (cost_saved / max(full_cost, 0.001)) * 100,
        "tests_selected": len(selected),
        "tests_total": len(all_tests),
        "tests_skipped": len(all_tests) - len(selected),
        "failures_caught_est": selected_failures_est,
        "failures_total_est": full_failures_est,
        "detection_rate": detection_rate,
        "bits_gained": selected_eig,
        "bits_per_dollar": bits_per_dollar,
        "bits_per_second": bits_per_second,
        "delta_entropy_bits": delta_entropy,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "information_possible_bits": full_eig,
        "budget_sec": budget_sec,
        "budget_usd": budget_usd,
        "budget_utilization_time": budget_util_time,
        "budget_utilization_cost": budget_util_cost,
        "decision_rationale": decision_rationale,
    }
    
    return metrics


def generate_markdown_report(
    metrics: Dict[str, Any],
    selected: List[Dict[str, Any]],
    all_tests: List[Dict[str, Any]]
) -> str:
    """Generate human-readable markdown report.
    
    Args:
        metrics: Computed metrics
        selected: Selected tests
        all_tests: All available tests
        
    Returns:
        Markdown report string
    """
    lines = []
    
    lines.append("# Epistemic CI Report")
    lines.append("")
    lines.append(f"**Generated:** {metrics['timestamp']}")
    lines.append("")
    
    # Executive Summary
    lines.append("## Executive Summary")
    lines.append("")
    lines.append(f"Selected **{metrics['tests_selected']}/{metrics['tests_total']} tests** "
                 f"to maximize information gain under budget constraints.")
    lines.append("")
    lines.append(f"- **Time saved:** {metrics['run_time_saved_sec']:.1f}s "
                 f"({metrics['time_reduction_pct']:.1f}% reduction)")
    lines.append(f"- **Cost saved:** ${metrics['run_cost_saved_usd']:.4f} "
                 f"({metrics['cost_reduction_pct']:.1f}% reduction)")
    lines.append(f"- **Information gained:** {metrics['bits_gained']:.2f} / "
                 f"{metrics['information_possible_bits']:.2f} bits")
    lines.append(f"- **Efficiency:** {metrics['bits_per_dollar']:.2f} bits per dollar")
    lines.append(f"- **Detection rate:** {metrics['detection_rate']*100:.1f}% "
                 f"({metrics['failures_caught_est']:.1f}/{metrics['failures_total_est']:.1f} failures)")
    if metrics.get("budget_utilization_time") is not None:
        lines.append(f"- **Budget utilization (time):** {metrics['budget_utilization_time']*100:.1f}%")
    if metrics.get("budget_utilization_cost") is not None:
        lines.append(f"- **Budget utilization (cost):** {metrics['budget_utilization_cost']*100:.1f}%")
    lines.append("")
    
    # Information Theory Metrics
    lines.append("## Information Theory Metrics")
    lines.append("")
    lines.append("| Metric | Value |")
    lines.append("|--------|-------|")
    lines.append(f"| Total EIG | {metrics['bits_gained']:.3f} bits |")
    lines.append(f"| Possible EIG | {metrics['information_possible_bits']:.3f} bits |")
    lines.append(f"| Bits per dollar | {metrics['bits_per_dollar']:.3f} |")
    lines.append(f"| Bits per second | {metrics['bits_per_second']:.5f} |")
    if metrics['delta_entropy_bits'] is not None:
        lines.append(f"| ΔH (entropy reduction) | {metrics['delta_entropy_bits']:.3f} bits |")
    lines.append("")
    
    # Practical Metrics
    lines.append("## Practical Metrics")
    lines.append("")
    lines.append("| Metric | Value |")
    lines.append("|--------|-------|")
    lines.append(f"| Tests selected | {metrics['tests_selected']} / {metrics['tests_total']} |")
    lines.append(f"| Time saved | {metrics['run_time_saved_sec']:.1f}s ({metrics['time_reduction_pct']:.1f}%) |")
    lines.append(f"| Cost saved | ${metrics['run_cost_saved_usd']:.4f} ({metrics['cost_reduction_pct']:.1f}%) |")
    lines.append(f"| Failures caught (est.) | {metrics['failures_caught_est']:.2f} / {metrics['failures_total_est']:.2f} |")
    lines.append(f"| Detection rate | {metrics['detection_rate']*100:.1f}% |")
    if metrics.get("budget_utilization_time") is not None:
        lines.append(
            f"| Budget utilization (time) | {metrics['budget_utilization_time']*100:.1f}% |"
        )
    if metrics.get("budget_utilization_cost") is not None:
        lines.append(
            f"| Budget utilization (cost) | {metrics['budget_utilization_cost']*100:.1f}% |"
        )
    lines.append("")
    
    # Top Selected Tests by EIG
    lines.append("## Top Selected Tests (by EIG)")
    lines.append("")
    lines.append("| Rank | Test | EIG (bits) | Cost ($) | Domain |")
    lines.append("|------|------|------------|----------|--------|")
    
    for i, test in enumerate(selected[:15], 1):
        name = test.get("name", "unknown")
        eig = test.get("eig_bits", 0)
        cost = test.get("cost_usd", 0)
        domain = test.get("domain", "generic")
        
        # Truncate long test names
        if len(name) > 60:
            name = name[:57] + "..."
        
        lines.append(f"| {i} | `{name}` | {eig:.4f} | {cost:.4f} | {domain} |")
    
    if len(selected) > 15:
        lines.append(f"| ... | *({len(selected) - 15} more)* | ... | ... | ... |")
    
    lines.append("")
    
    # Domain Breakdown
    lines.append("## Domain Breakdown")
    lines.append("")
    
    domain_stats = {}
    for test in selected:
        domain = test.get("domain", "generic")
        if domain not in domain_stats:
            domain_stats[domain] = {"count": 0, "eig": 0, "cost": 0}
        domain_stats[domain]["count"] += 1
        domain_stats[domain]["eig"] += test.get("eig_bits", 0)
        domain_stats[domain]["cost"] += test.get("cost_usd", 0)
    
    lines.append("| Domain | Tests | EIG (bits) | Cost ($) | Bits/$ |")
    lines.append("|--------|-------|------------|----------|--------|")
    
    for domain in sorted(domain_stats.keys()):
        stats = domain_stats[domain]
        bits_per_dollar = stats["eig"] / max(stats["cost"], 0.001)
        lines.append(f"| {domain} | {stats['count']} | {stats['eig']:.3f} | "
                    f"{stats['cost']:.4f} | {bits_per_dollar:.2f} |")
    
    lines.append("")
    
    # Methodology
    lines.append("## Methodology")
    lines.append("")
    lines.append("**Test Selection Algorithm:** Greedy knapsack maximizing EIG per cost")
    lines.append("")
    lines.append("**EIG Computation:**")
    lines.append("1. If `entropy_before` and `entropy_after` available: ΔH = H_before - H_after")
    lines.append("2. Else if `model_uncertainty` (predicted failure prob p): H(p) = -p log₂(p) - (1-p) log₂(1-p)")
    lines.append("3. Else: Wilson-smoothed empirical failure rate")
    lines.append("")
    lines.append("**Detection Rate:** Estimated from sum of predicted failure probabilities")
    lines.append("")
    
    return "\n".join(lines)


def get_git_sha() -> str:
    """Get current git commit SHA."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True
        )
        return result.stdout.strip()
    except Exception:
        return "unknown"


def get_env_hash() -> str:
    """Compute hash of relevant environment variables for reproducibility."""
    env_keys = sorted([
        "PYTHON_VERSION",
        "PYTEST_VERSION",
        "RUNNER_USD_PER_HOUR",
        "CI_BUDGET_FRACTION",
    ])
    env_str = "|".join(f"{k}={os.getenv(k, '')}" for k in env_keys)
    return hashlib.sha256(env_str.encode()).hexdigest()[:16]


def get_branch_name() -> str:
    """Get current git branch name."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            capture_output=True,
            text=True,
            check=True
        )
        return result.stdout.strip()
    except Exception:
        return "unknown"


def emit_experiment_ledger(
    run_id: str,
    metrics: Dict[str, Any],
    selected: List[Dict[str, Any]],
    all_tests: List[Dict[str, Any]],
    selection_stats: Dict[str, Any],
    seed: Optional[int],
    ledger_dir: pathlib.Path
) -> None:
    """Emit experiment ledger with uncertainty telemetry.
    
    Args:
        run_id: Unique run identifier
        metrics: Computed metrics
        selected: Selected tests
        all_tests: All available tests
        selection_stats: Selection statistics
        seed: Random seed (None if not set)
        ledger_dir: Directory for ledger files
    """
    commit_sha = get_git_sha()
    
    # Load ML model metadata if available
    model_meta_path = pathlib.Path("models/metadata.json")
    ml_model_f1 = None
    ml_model_version = "unknown"
    if model_meta_path.exists():
        try:
            model_meta = json.loads(model_meta_path.read_text())
            ml_model_f1 = model_meta.get("f1")
            ml_model_version = "selector-v1.pkl"
        except Exception:
            pass
    
    # Build per-test data with enriched metrics
    tests_ledger = []
    selected_names = {t["name"] for t in selected}
    
    for rank, test in enumerate(all_tests, 1):
        test_entry = {
            "name": test["name"],
            "suite": test.get("domain", test.get("suite", "generic")),
            "selected": test["name"] in selected_names,
            "duration_sec": test.get("duration_sec", 0),
            "cost_usd": test.get("cost_usd", 0),
            "result": test.get("result", "skip"),
            "model_uncertainty": test.get("model_uncertainty", None),
            "eig_bits": test.get("eig_bits", 0),
            "eig_rank": rank,
        }

        # Add optional fields
        if test.get("failure_type"):
            test_entry["failure_type"] = test["failure_type"]
        if test.get("metrics"):
            test_entry["metrics"] = test["metrics"]
        if test.get("entropy_before") is not None:
            test_entry["entropy_before"] = test["entropy_before"]
        if test.get("entropy_after") is not None:
            test_entry["entropy_after"] = test["entropy_after"]

        tests_ledger.append(test_entry)
    
    # Construct ledger matching schema
    ledger = {
        "run_id": run_id,
        "timestamp": metrics["timestamp"],
        "commit_sha": commit_sha,
        "branch": get_branch_name(),
        "tests_selected": metrics["tests_selected"],
        "tests_total": metrics["tests_total"],
        "information_gained_bits": metrics["bits_gained"],
        "information_possible_bits": metrics["information_possible_bits"],
        "efficiency_bits_per_dollar": metrics["bits_per_dollar"],
        "budget_sec": metrics["budget_sec"],
        "budget_usd": metrics["budget_usd"],
        "walltime_sec": sum(t["duration_sec"] for t in selected),
        "cost_usd": sum(t["cost_usd"] for t in selected),
        "detection_rate": metrics["detection_rate"],
        "seed": seed,
        "env_hash": get_env_hash(),
        "tests": tests_ledger,
        "selection_strategy": "epistemic",
        "ml_model_version": ml_model_version,
        "ml_model_f1": ml_model_f1,
        "budget_utilization": {
            "time": metrics.get("budget_utilization_time"),
            "cost": metrics.get("budget_utilization_cost"),
        },
        "decision_rationale": metrics.get("decision_rationale"),
        "delta_entropy_bits": metrics.get("delta_entropy_bits"),
    }
    
    # Write ledger
    ledger_dir.mkdir(parents=True, exist_ok=True)
    ledger_path = ledger_dir / f"{run_id}.json"
    
    with ledger_path.open("w", encoding="utf-8") as f:
        json.dump(ledger, f, indent=2)
    
    print(f"📝 Emitted experiment ledger to {ledger_path}", flush=True)


def main() -> int:
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Generate epistemic CI report")
    parser.add_argument(
        "--selected",
        default="artifact/selected_tests.json",
        help="Selected tests JSON"
    )
    parser.add_argument(
        "--rankings",
        default="artifact/eig_rankings.json",
        help="EIG rankings JSON (all tests)"
    )
    parser.add_argument(
        "--metrics-out",
        default="artifact/ci_metrics.json",
        help="Output metrics JSON"
    )
    parser.add_argument(
        "--report-out",
        default="artifact/ci_report.md",
        help="Output report markdown"
    )
    parser.add_argument(
        "--ledger-dir",
        default="experiments/ledger",
        help="Directory for experiment ledger files"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed used for this run (for reproducibility tracking)"
    )
    
    args = parser.parse_args()
    
    selected_path = pathlib.Path(args.selected)
    rankings_path = pathlib.Path(args.rankings)
    metrics_path = pathlib.Path(args.metrics_out)
    report_path = pathlib.Path(args.report_out)
    ledger_dir = pathlib.Path(args.ledger_dir)
    
    print("=" * 80, flush=True)
    print("📊 Epistemic CI Report Generation", flush=True)
    print("=" * 80, flush=True)
    
    # Load data
    if not selected_path.exists():
        print(f"⚠️  Selected tests not found: {selected_path}", flush=True)
        return 1
    
    if not rankings_path.exists():
        print(f"⚠️  EIG rankings not found: {rankings_path}", flush=True)
        return 1
    
    selected_data = json.loads(selected_path.read_text(encoding="utf-8"))
    all_tests = json.loads(rankings_path.read_text(encoding="utf-8"))
    
    selected = selected_data.get("tests", [])
    selection_stats = selected_data.get("stats", {})
    
    print(f"📊 Loaded {len(selected)} selected / {len(all_tests)} total tests", flush=True)
    
    # Compute metrics
    metrics = compute_metrics(selected, all_tests, selection_stats)
    
    # Generate report
    report = generate_markdown_report(metrics, selected, all_tests)
    
    # Save outputs
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    
    metrics_path.write_text(json.dumps(metrics, indent=2))
    report_path.write_text(report)
    
    print(f"\n✅ Saved metrics to {metrics_path}", flush=True)
    print(f"✅ Saved report to {report_path}", flush=True)
    
    # Emit experiment ledger
    run_id = get_git_sha()[:12]  # Use first 12 chars of commit SHA
    emit_experiment_ledger(
        run_id=run_id,
        metrics=metrics,
        selected=selected,
        all_tests=all_tests,
        selection_stats=selection_stats,
        seed=args.seed,
        ledger_dir=ledger_dir
    )
    
    # Print summary
    print(f"\n📈 Summary:", flush=True)
    print(f"   Time saved: {metrics['run_time_saved_sec']:.1f}s ({metrics['time_reduction_pct']:.1f}%)", flush=True)
    print(f"   Cost saved: ${metrics['run_cost_saved_usd']:.4f} ({metrics['cost_reduction_pct']:.1f}%)", flush=True)
    print(f"   Information gained: {metrics['bits_gained']:.2f} bits", flush=True)
    print(f"   Efficiency: {metrics['bits_per_dollar']:.2f} bits/$", flush=True)
    
    print("=" * 80, flush=True)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
