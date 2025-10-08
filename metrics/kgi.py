#!/usr/bin/env python3
"""Knowledge-Gain Index (KGI) - Formal metric for epistemic efficiency.

KGI measures the expected reduction in predictive uncertainty per experimental run.
Combines entropy reduction, calibration improvement, and prediction reliability
into a single scalar that quantifies R&D learning rate.

Formula:
    KGI = w_entropy * Δ_entropy + w_ece * (1 - ECE) + w_brier * (1 - Brier)
    where weights sum to 1.0

Interpretation:
    - High KGI (>0.5): Rapid knowledge gain
    - Medium KGI (0.1-0.5): Steady progress
    - Low KGI (<0.1): Learning plateau

Output: evidence/summary/kgi.json
"""

import json
import math
import pathlib
import sys
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent / "scripts"))
from _config import get_config


def compute_kgi(
    current_run: Dict[str, Any],
    baseline_window: List[Dict[str, Any]],
    config: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """Compute Knowledge-Gain Index (KGI) for current run.
    
    Args:
        current_run: Current run metrics dict
        baseline_window: List of recent runs for baseline
        config: Config dict (optional, loads from _config if None)
    
    Returns:
        Dict with KGI score and components
    """
    if config is None:
        config = get_config()
    
    # Extract weights
    w_entropy = config["KGI_WEIGHT_ENTROPY"]
    w_ece = config["KGI_WEIGHT_ECE"]
    w_brier = config["KGI_WEIGHT_BRIER"]
    
    # Normalize weights to sum to 1.0
    total_weight = w_entropy + w_ece + w_brier
    w_entropy /= total_weight
    w_ece /= total_weight
    w_brier /= total_weight
    
    # Component 1: Entropy reduction (bits/run)
    entropy_delta_current = current_run.get("entropy_delta_mean", 0.0)
    
    # Baseline entropy delta (average over window)
    baseline_entropy_deltas = [
        r.get("entropy_delta_mean", 0.0) for r in baseline_window
        if r.get("entropy_delta_mean") is not None
    ]
    
    if baseline_entropy_deltas:
        baseline_entropy = sum(baseline_entropy_deltas) / len(baseline_entropy_deltas)
        # Normalized entropy gain (0 = no change, 1 = halved uncertainty)
        if baseline_entropy > 1e-6:
            entropy_gain = max(0.0, (baseline_entropy - entropy_delta_current) / baseline_entropy)
        else:
            entropy_gain = 0.0
    else:
        entropy_gain = 0.0
    
    # Component 2: Calibration quality (1 - ECE, higher is better)
    ece_current = current_run.get("ece", 0.5)
    calibration_quality = max(0.0, 1.0 - ece_current)
    
    # Component 3: Prediction reliability (1 - Brier, higher is better)
    brier_current = current_run.get("brier", 0.5)
    reliability = max(0.0, 1.0 - brier_current)
    
    # Compute weighted KGI
    kgi = w_entropy * entropy_gain + w_ece * calibration_quality + w_brier * reliability
    
    # Guardrails: clamp to [0, 1]
    kgi = max(0.0, min(1.0, kgi))
    
    return {
        "kgi_u": kgi,
        "units": "unitless",
        "disclaimer": "Unitless composite score (0-1); not Shannon entropy in bits.",
        "components": {
            "entropy_gain": entropy_gain,
            "calibration_quality": calibration_quality,
            "reliability": reliability,
        },
        "weights": {
            "entropy": w_entropy,
            "ece": w_ece,
            "brier": w_brier,
        },
        "interpretation": _interpret_kgi(kgi),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


def _interpret_kgi(kgi: float) -> str:
    """Interpret KGI value.
    
    Args:
        kgi: KGI score (0-1)
    
    Returns:
        Human-readable interpretation
    """
    if kgi >= 0.7:
        return "Excellent - Rapid knowledge gain (top 10% of runs)"
    elif kgi >= 0.5:
        return "Good - Strong learning progress"
    elif kgi >= 0.3:
        return "Fair - Moderate knowledge gain"
    elif kgi >= 0.1:
        return "Low - Slow progress, consider exploration"
    else:
        return "Very Low - Learning plateau, intervention needed"


def compute_kgi_trend(runs: List[Dict[str, Any]], config: Dict[str, Any]) -> Dict[str, Any]:
    """Compute KGI trend over multiple runs.
    
    Args:
        runs: List of run dicts (oldest first)
        config: Config dict
    
    Returns:
        Dict with trend data
    """
    window = config["KGI_WINDOW"]
    
    if len(runs) < 2:
        return {"trend": [], "ewma": None, "improvement": None}
    
    # Compute KGI for each run (using previous runs as baseline)
    kgi_values = []
    
    for i in range(1, len(runs)):
        baseline_start = max(0, i - window)
        baseline_window = runs[baseline_start:i]
        current_run = runs[i]
        
        kgi_result = compute_kgi(current_run, baseline_window, config)
        kgi_values.append({
            "timestamp": current_run.get("timestamp", "unknown"),
            "kgi": kgi_result["kgi_u"],
            "run_id": current_run.get("ci_run_id", "unknown"),
        })
    
    # Compute EWMA (α = 0.2)
    if kgi_values:
        ewma = kgi_values[0]["kgi"]
        for kgi_point in kgi_values[1:]:
            ewma = 0.2 * kgi_point["kgi"] + 0.8 * ewma
    else:
        ewma = None
    
    # Compute improvement (recent vs old)
    if len(kgi_values) >= 5:
        recent_avg = sum(kp["kgi"] for kp in kgi_values[-5:]) / 5
        old_avg = sum(kp["kgi"] for kp in kgi_values[:5]) / 5
        improvement = recent_avg - old_avg
    else:
        improvement = None
    
    return {
        "trend": kgi_values[-20:],  # Last 20 points
        "ewma": ewma,
        "improvement": improvement,
    }


def main() -> int:
    """Compute and save KGI for current run.
    
    Returns:
        0 on success
    """
    config = get_config()
    
    print()
    print("=" * 100)
    print("KNOWLEDGE-GAIN INDEX (KGI) COMPUTATION")
    print("=" * 100)
    print()
    
    # Load current run metrics
    current_metrics_file = pathlib.Path("evidence/current_run_metrics.json")
    if not current_metrics_file.exists():
        print("⚠️  Current metrics not found, run 'python metrics/registry.py' first")
        return 0
    
    with current_metrics_file.open() as f:
        current_run = json.load(f)
    
    # Load baseline window
    runs_dir = pathlib.Path("evidence/runs")
    if runs_dir.exists():
        sys.path.insert(0, str(pathlib.Path(__file__).parent.parent / "scripts"))
        from baseline_update import load_successful_runs
        baseline_window = load_successful_runs(runs_dir, config["KGI_WINDOW"])
    else:
        baseline_window = []
    
    print(f"📊 Current run: {current_run.get('ci_run_id', 'unknown')}")
    print(f"   Baseline window: {len(baseline_window)} runs")
    print()
    
    # Compute KGI
    kgi_result = compute_kgi(current_run, baseline_window, config)
    
    print(f"{'=' * 100}")
    print(f"KGI_u SCORE (unitless): {kgi_result['kgi_u']:.4f}")
    print(f"{'=' * 100}")
    print()
    print("Components:")
    print(f"  Entropy Gain:         {kgi_result['components']['entropy_gain']:.4f} (weight: {kgi_result['weights']['entropy']:.2f})")
    print(f"  Calibration Quality:  {kgi_result['components']['calibration_quality']:.4f} (weight: {kgi_result['weights']['ece']:.2f})")
    print(f"  Reliability:          {kgi_result['components']['reliability']:.4f} (weight: {kgi_result['weights']['brier']:.2f})")
    print()
    print(f"Interpretation: {kgi_result['interpretation']}")
    print()
    
    # Compute trend if we have enough runs
    if len(baseline_window) >= 2:
        all_runs = baseline_window + [current_run]
        trend = compute_kgi_trend(all_runs, config)
        
        if trend["ewma"] is not None:
            print(f"Trend:")
            print(f"  EWMA:        {trend['ewma']:.4f}")
            if trend["improvement"] is not None:
                arrow = "↑" if trend["improvement"] > 0 else "↓"
                print(f"  Improvement: {arrow} {abs(trend['improvement']):.4f} (recent 5 vs old 5)")
            print()
        
        kgi_result["trend"] = trend
    
    # Save to JSON
    output_file = pathlib.Path("evidence/summary/kgi.json")
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    with output_file.open("w") as f:
        json.dump(kgi_result, f, indent=2)
    
    print(f"💾 KGI saved to: {output_file}")
    print()
    
    print("✅ KGI computation complete!")
    print()
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
