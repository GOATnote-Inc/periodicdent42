#!/usr/bin/env python3
"""
Validate calibration of uncertainty estimates on UCI Superconductivity test set.

Success Criteria:
- Expected Calibration Error (ECE) < 0.05
- Prediction Interval Coverage Probability (PICP) at 95%: [0.94, 0.96]
- Maximum Calibration Error (MCE) < 0.10
- Sharpness: Mean prediction interval width < 20 K

Usage:
    python scripts/validate_calibration.py \\
        --model models/rf_baseline.pkl \\
        --data data/processed/uci_splits/test.csv \\
        --output evidence/validation/calibration/
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.rf_qrf import RandomForestQRF
from src.uncertainty.calibration_metrics import (
    expected_calibration_error,
    prediction_interval_coverage_probability,
    mean_prediction_interval_width,
    sharpness
)
from src.uncertainty.conformal import SplitConformalPredictor


def load_model(model_path: Path) -> RandomForestQRF:
    """Load trained model."""
    print(f"📂 Loading model from {model_path}")
    model = RandomForestQRF.load(model_path)
    print(f"✅ Model loaded (n_estimators={model.n_estimators})")
    return model


def load_test_data(data_path: Path) -> Tuple[np.ndarray, np.ndarray]:
    """Load test data."""
    print(f"📂 Loading test data from {data_path}")
    df = pd.read_csv(data_path)
    
    feature_cols = [c for c in df.columns if c != 'critical_temp']
    X_test = df[feature_cols].values
    y_test = df['critical_temp'].values
    
    print(f"✅ Loaded {len(X_test)} test samples with {X_test.shape[1]} features")
    return X_test, y_test


def conformal_intervals(
    y_pred: np.ndarray,
    y_std: np.ndarray,
    alpha: float = 0.05
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute conformal prediction intervals.
    
    For simplicity, this uses a Gaussian approximation:
    PI = [y_pred ± z_alpha/2 * y_std]
    
    Args:
        y_pred: Point predictions
        y_std: Prediction standard deviations
        alpha: Significance level (0.05 for 95% intervals)
    
    Returns:
        y_lower, y_upper: Prediction interval bounds
    """
    z_score = stats.norm.ppf(1 - alpha / 2)
    y_lower = y_pred - z_score * y_std
    y_upper = y_pred + z_score * y_std
    return y_lower, y_upper


def bootstrap_ci(
    func,
    *args,
    n_boot: int = 1000,
    confidence: float = 0.95,
    random_state: int = 42
) -> Tuple[float, float]:
    """
    Compute bootstrap confidence interval for a metric.
    
    Args:
        func: Metric function
        *args: Arguments to pass to func
        n_boot: Number of bootstrap iterations
        confidence: Confidence level
        random_state: Random seed
    
    Returns:
        lower, upper: Confidence interval bounds
    """
    rng = np.random.RandomState(random_state)
    n_samples = len(args[0])
    
    boot_scores = []
    for _ in range(n_boot):
        # Resample with replacement
        idx = rng.choice(n_samples, size=n_samples, replace=True)
        boot_args = [arg[idx] if hasattr(arg, '__getitem__') else arg for arg in args]
        
        try:
            score = func(*boot_args)
            boot_scores.append(score)
        except:
            continue
    
    # Compute percentiles
    alpha = 1 - confidence
    lower = np.percentile(boot_scores, alpha / 2 * 100)
    upper = np.percentile(boot_scores, (1 - alpha / 2) * 100)
    
    return float(lower), float(upper)


def plot_reliability_diagram(
    ax,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_std: np.ndarray,
    n_bins: int = 10
):
    """
    Plot reliability diagram (calibration curve).
    
    Args:
        ax: Matplotlib axis
        y_true: True values
        y_pred: Predicted values
        y_std: Prediction standard deviations
        n_bins: Number of confidence bins
    """
    # Compute confidence levels and empirical coverage
    confidence_levels = np.linspace(0.05, 0.95, n_bins)
    empirical_coverage = []
    
    for conf in confidence_levels:
        alpha = 1 - conf
        z_score = stats.norm.ppf(1 - alpha / 2)
        y_lower = y_pred - z_score * y_std
        y_upper = y_pred + z_score * y_std
        
        # Compute empirical coverage
        in_interval = (y_true >= y_lower) & (y_true <= y_upper)
        coverage = np.mean(in_interval)
        empirical_coverage.append(coverage)
    
    # Plot
    ax.plot(confidence_levels, empirical_coverage, 'o-', linewidth=2, markersize=8,
            label='Model', color='#2E86AB')
    ax.plot([0, 1], [0, 1], 'k--', linewidth=1.5, label='Perfect calibration')
    
    ax.set_xlabel('Predicted Confidence Level', fontsize=12)
    ax.set_ylabel('Empirical Coverage', fontsize=12)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    ax.grid(alpha=0.3)
    ax.legend(fontsize=10)


def compute_calibration_metrics(
    y_test: np.ndarray,
    y_pred: np.ndarray,
    y_std: np.ndarray,
    y_lower: np.ndarray,
    y_upper: np.ndarray
) -> dict:
    """
    Compute all calibration metrics with bootstrap confidence intervals.
    
    Returns:
        Dictionary of metrics with CIs
    """
    print("\n📊 Computing calibration metrics...")
    
    # ECE (Expected Calibration Error)
    ece = expected_calibration_error(y_test, y_pred, y_std, n_bins=10)
    ece_ci = bootstrap_ci(
        expected_calibration_error,
        y_test, y_pred, y_std,
        n_boot=1000
    )
    print(f"   ECE: {ece:.4f} (95% CI: [{ece_ci[0]:.4f}, {ece_ci[1]:.4f}])")
    
    # PICP (Prediction Interval Coverage Probability) at 95%
    picp = prediction_interval_coverage_probability(y_test, y_lower, y_upper)
    picp_ci = bootstrap_ci(
        prediction_interval_coverage_probability,
        y_test, y_lower, y_upper,
        n_boot=1000
    )
    print(f"   PICP@95%: {picp:.3f} (95% CI: [{picp_ci[0]:.3f}, {picp_ci[1]:.3f}])")
    
    # MCE (Maximum Calibration Error)
    # Approximate as max deviation in reliability diagram
    confidence_levels = np.linspace(0.05, 0.95, 10)
    empirical_coverage = []
    for conf in confidence_levels:
        alpha = 1 - conf
        z_score = stats.norm.ppf(1 - alpha / 2)
        y_l = y_pred - z_score * y_std
        y_u = y_pred + z_score * y_std
        in_interval = (y_test >= y_l) & (y_test <= y_u)
        empirical_coverage.append(np.mean(in_interval))
    
    mce = np.max(np.abs(np.array(empirical_coverage) - confidence_levels))
    print(f"   MCE: {mce:.4f}")
    
    # Sharpness (mean interval width)
    sharp = sharpness(y_lower, y_upper)
    print(f"   Sharpness: {sharp:.2f} K (mean interval width)")
    
    # MPIW (Mean Prediction Interval Width) - same as sharpness
    mpiw = mean_prediction_interval_width(y_lower, y_upper)
    
    # Check success criteria
    success = (
        ece < 0.05 and
        0.94 <= picp <= 0.96 and
        mce < 0.10 and
        sharp < 20.0
    )
    
    metrics = {
        'ece': float(ece),
        'ece_ci_95': [float(ece_ci[0]), float(ece_ci[1])],
        'mce': float(mce),
        'picp_95': float(picp),
        'picp_ci_95': [float(picp_ci[0]), float(picp_ci[1])],
        'sharpness_mean': float(sharp),
        'mpiw': float(mpiw),
        'success': bool(success),
        'timestamp': datetime.now().isoformat(),
        'n_test_samples': len(y_test)
    }
    
    return metrics


def interpret_calibration(ece: float, picp: float, sharpness: float) -> str:
    """Generate human-readable interpretation of calibration results."""
    interpretation_parts = []
    
    # ECE interpretation
    if ece < 0.05:
        interpretation_parts.append(
            f"✅ ECE = {ece:.4f} < 0.05: Model is well-calibrated. "
            "Predicted confidence levels match empirical coverage."
        )
    else:
        interpretation_parts.append(
            f"❌ ECE = {ece:.4f} ≥ 0.05: Model is miscalibrated. "
            "Predictions are either overconfident or underconfident. "
            "Recommendation: Apply temperature scaling or recalibrate conformal predictor."
        )
    
    # PICP interpretation
    if 0.94 <= picp <= 0.96:
        interpretation_parts.append(
            f"✅ PICP@95% = {picp:.3f} ∈ [0.94, 0.96]: Prediction intervals have correct coverage. "
            "Suitable for autonomous decision-making."
        )
    elif picp < 0.94:
        interpretation_parts.append(
            f"❌ PICP@95% = {picp:.3f} < 0.94: Prediction intervals are too narrow (overconfident). "
            "Recommendation: Increase conformal quantile or apply safety margin."
        )
    else:  # picp > 0.96
        interpretation_parts.append(
            f"⚠️  PICP@95% = {picp:.3f} > 0.96: Prediction intervals are too wide (underconfident). "
            "Model uncertainty may be overestimated. Consider tighter intervals."
        )
    
    # Sharpness interpretation
    if sharpness < 20.0:
        interpretation_parts.append(
            f"✅ Sharpness = {sharpness:.1f} K < 20 K: Intervals are reasonably tight. "
            "Good balance between coverage and precision."
        )
    else:
        interpretation_parts.append(
            f"⚠️  Sharpness = {sharpness:.1f} K ≥ 20 K: Intervals are wide. "
            "Model may lack confidence or data may be noisy."
        )
    
    return "\n\n".join(interpretation_parts)


def validate_calibration(
    model_path: Path,
    test_data_path: Path,
    output_dir: Path,
    train_data_path: Path = None,
    val_data_path: Path = None
):
    """
    Main calibration validation pipeline.
    
    Generates:
    1. calibration_curve.png - Reliability diagram
    2. calibration_by_tc_range.png - Stratified by Tc quartiles
    3. calibration_metrics.json - ECE, MCE, PICP, sharpness with 95% CIs
    4. calibration_interpretation.txt - Plain English interpretation
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 70)
    print("CALIBRATION VALIDATION")
    print("=" * 70)
    
    # Load model and data
    model = load_model(model_path)
    X_test, y_test = load_test_data(test_data_path)
    
    # Generate predictions with uncertainty
    print("\n🔮 Generating predictions with uncertainty...")
    y_pred, y_lower, y_upper = model.predict_with_uncertainty(X_test)
    y_std = model.get_epistemic_uncertainty(X_test)
    
    print(f"✅ Generated predictions for {len(y_test)} samples")
    print(f"   Mean prediction: {np.mean(y_pred):.2f} K")
    print(f"   Mean uncertainty: {np.mean(y_std):.2f} K")
    print(f"   Mean interval width: {np.mean(y_upper - y_lower):.2f} K")
    
    # Compute calibration metrics
    metrics = compute_calibration_metrics(y_test, y_pred, y_std, y_lower, y_upper)
    
    # Plot 1: Reliability diagram
    print("\n📊 Generating reliability diagram...")
    fig, ax = plt.subplots(figsize=(8, 6))
    plot_reliability_diagram(ax, y_test, y_pred, y_std)
    ax.set_title(
        f'Calibration: ECE={metrics["ece"]:.4f}, PICP@95%={metrics["picp_95"]:.3f}',
        fontsize=14, fontweight='bold'
    )
    plt.tight_layout()
    plt.savefig(output_dir / 'calibration_curve.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"💾 Saved: {output_dir / 'calibration_curve.png'}")
    
    # Plot 2: Stratified calibration by Tc range
    print("\n📊 Generating stratified calibration plots...")
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    tc_quartiles = pd.qcut(y_test, q=4, labels=['Q1: Low', 'Q2: Mid-Low', 'Q3: Mid-High', 'Q4: High'])
    
    for ax_idx, (ax, q_label) in enumerate(zip(axes.flat, ['Q1: Low', 'Q2: Mid-Low', 'Q3: Mid-High', 'Q4: High'])):
        mask = tc_quartiles == q_label
        plot_reliability_diagram(ax, y_test[mask], y_pred[mask], y_std[mask])
        
        # Compute ECE for this quartile
        q_ece = expected_calibration_error(y_test[mask], y_pred[mask], y_std[mask], n_bins=10)
        ax.set_title(f'{q_label} Tc: ECE={q_ece:.4f}', fontsize=12, fontweight='bold')
    
    plt.suptitle('Calibration Stratified by Tc Range', fontsize=14, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.savefig(output_dir / 'calibration_by_tc_range.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"💾 Saved: {output_dir / 'calibration_by_tc_range.png'}")
    
    # Save metrics
    metrics_path = output_dir / 'calibration_metrics.json'
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"\n💾 Saved metrics: {metrics_path}")
    
    # Generate interpretation
    interpretation = f"""
CALIBRATION VALIDATION RESULTS
==============================

Dataset: UCI Superconductivity Test Set (N={len(y_test)})
Timestamp: {metrics['timestamp']}

SUMMARY
-------
ECE: {metrics['ece']:.4f} (95% CI: [{metrics['ece_ci_95'][0]:.4f}, {metrics['ece_ci_95'][1]:.4f}])
Target: < 0.05
Status: {'✅ PASS' if metrics['ece'] < 0.05 else '❌ FAIL'}

PICP@95%: {metrics['picp_95']:.3f} (95% CI: [{metrics['picp_ci_95'][0]:.3f}, {metrics['picp_ci_95'][1]:.3f}])
Target: [0.94, 0.96]
Status: {'✅ PASS' if 0.94 <= metrics['picp_95'] <= 0.96 else '❌ FAIL'}

MCE: {metrics['mce']:.4f}
Target: < 0.10
Status: {'✅ PASS' if metrics['mce'] < 0.10 else '❌ FAIL'}

Sharpness: {metrics['sharpness_mean']:.1f} K (mean interval width)
Target: < 20 K
Status: {'✅ PASS' if metrics['sharpness_mean'] < 20.0 else '❌ FAIL'}

OVERALL: {'✅ ALL CRITERIA MET' if metrics['success'] else '❌ SOME CRITERIA NOT MET'}

INTERPRETATION
--------------
{interpret_calibration(metrics['ece'], metrics['picp_95'], metrics['sharpness_mean'])}

DEPLOYMENT RECOMMENDATION
-------------------------
"""
    
    if metrics['success']:
        interpretation += """
✅ READY FOR DEPLOYMENT

This model demonstrates well-calibrated uncertainty estimates suitable for autonomous 
decision-making in a robotic laboratory. Prediction intervals can be trusted for:
- GO/NO-GO synthesis decisions
- Resource allocation (prioritize high-confidence predictions)
- Risk assessment (flag low-confidence regions for human review)

Confidence level: HIGH (all success criteria met)
"""
    else:
        interpretation += """
⚠️  RECALIBRATION REQUIRED

Model requires recalibration before deployment. Recommendations:
1. Apply temperature scaling or Platt scaling to ECE
2. Adjust conformal prediction quantiles for PICP
3. Consider ensemble methods to improve sharpness

Re-run validation after recalibration.

Confidence level: LOW (recalibration needed)
"""
    
    interpretation_path = output_dir / 'calibration_interpretation.txt'
    with open(interpretation_path, 'w') as f:
        f.write(interpretation)
    print(f"📄 Saved interpretation: {interpretation_path}")
    
    # Print summary
    print("\n" + "=" * 70)
    if metrics['success']:
        print("✅ CALIBRATION VALIDATION: PASSED")
    else:
        print("❌ CALIBRATION VALIDATION: FAILED")
    print("=" * 70)
    print(f"\nResults saved to: {output_dir}")
    print(f"  - calibration_curve.png")
    print(f"  - calibration_by_tc_range.png")
    print(f"  - calibration_metrics.json")
    print(f"  - calibration_interpretation.txt")
    
    # Exit with appropriate code
    sys.exit(0 if metrics['success'] else 1)


def main():
    parser = argparse.ArgumentParser(
        description="Validate calibration of uncertainty estimates"
    )
    parser.add_argument(
        '--model',
        type=Path,
        required=True,
        help='Path to trained model (e.g., models/rf_baseline.pkl)'
    )
    parser.add_argument(
        '--data',
        type=Path,
        required=True,
        help='Path to test data CSV'
    )
    parser.add_argument(
        '--output',
        type=Path,
        default=Path('evidence/validation/calibration'),
        help='Output directory for validation artifacts'
    )
    parser.add_argument(
        '--train-data',
        type=Path,
        help='Optional: Path to training data (for conformal prediction)'
    )
    parser.add_argument(
        '--val-data',
        type=Path,
        help='Optional: Path to validation data (for conformal prediction)'
    )
    
    args = parser.parse_args()
    
    validate_calibration(
        args.model,
        args.data,
        args.output,
        args.train_data,
        args.val_data
    )


if __name__ == '__main__':
    main()

