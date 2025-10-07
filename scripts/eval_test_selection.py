#!/usr/bin/env python3
"""Evaluate ML test selection model with confidence intervals.

Usage:
    python scripts/eval_test_selection.py --output reports/ml_eval.json
"""

import argparse
import json
import numpy as np
from pathlib import Path
from sklearn.metrics import precision_recall_curve, roc_auc_score, f1_score
from sklearn.model_selection import cross_val_score
import joblib


def wilson_ci(p, n, confidence=0.95):
    """Compute Wilson score confidence interval for a proportion."""
    from scipy.stats import norm
    z = norm.ppf((1 + confidence) / 2)
    denominator = 1 + z**2 / n
    center = (p + z**2 / (2 * n)) / denominator
    margin = z * np.sqrt((p * (1 - p) / n + z**2 / (4 * n**2))) / denominator
    return (center - margin, center + margin)


def bootstrap_ci(data, stat_func, n_bootstrap=1000, confidence=0.95):
    """Compute bootstrap confidence interval."""
    bootstrap_stats = []
    for _ in range(n_bootstrap):
        sample = np.random.choice(data, size=len(data), replace=True)
        bootstrap_stats.append(stat_func(sample))
    
    alpha = (1 - confidence) / 2
    return (np.percentile(bootstrap_stats, alpha * 100),
            np.percentile(bootstrap_stats, (1 - alpha) * 100))


def main():
    parser = argparse.ArgumentParser(description="Evaluate ML test selection")
    parser.add_argument("--model", default="test_selector.pkl",
                        help="Path to trained model")
    parser.add_argument("--data", default="training_data.json",
                        help="Path to training data")
    parser.add_argument("--output", default="reports/ml_eval.json",
                        help="Output JSON file")
    args = parser.parse_args()
    
    print("=" * 80)
    print("ML TEST SELECTION EVALUATION")
    print("=" * 80)
    print()
    
    # Load data
    data_path = Path(args.data)
    if not data_path.exists():
        print(f"✗ Data file not found: {data_path}")
        return
    
    with open(data_path) as f:
        records = json.load(f)
    
    print(f"✓ Loaded {len(records)} training records")
    print()
    
    # Extract features and labels
    feature_names = ["lines_added", "lines_deleted", "files_changed",
                     "complexity_delta", "recent_failure_rate",
                     "avg_duration", "days_since_last_change"]
    
    X = []
    y = []
    for record in records:
        features = record.get("features", {})
        X.append([features.get(name, 0) for name in feature_names])
        y.append(0 if record["passed"] else 1)  # 1 = fail
    
    X = np.array(X)
    y = np.array(y)
    
    n_total = len(y)
    n_failures = np.sum(y)
    n_passes = n_total - n_failures
    failure_rate = n_failures / n_total
    
    print(f"Dataset statistics:")
    print(f"  Total samples: {n_total}")
    print(f"  Failures: {n_failures} ({failure_rate:.1%})")
    print(f"  Passes: {n_passes} ({1-failure_rate:.1%})")
    
    # Compute CI for failure rate
    fail_ci_low, fail_ci_high = wilson_ci(failure_rate, n_total)
    print(f"  Failure rate 95% CI: [{fail_ci_low:.3f}, {fail_ci_high:.3f}]")
    print()
    
    # Load model if available
    model_path = Path(args.model)
    if model_path.exists():
        print(f"✓ Loading model from {model_path}")
        model = joblib.load(model_path)
        
        # Cross-validation F1 score
        cv_scores = cross_val_score(model, X, y, cv=5, scoring='f1')
        cv_mean = np.mean(cv_scores)
        cv_std = np.std(cv_scores)
        print(f"  5-fold CV F1: {cv_mean:.3f} ± {cv_std:.3f}")
        
        # Test set performance (using full dataset as we don't have a held-out set)
        y_pred = model.predict(X)
        y_proba = model.predict_proba(X)[:, 1]
        
        f1 = f1_score(y, y_pred)
        print(f"  Training F1: {f1:.3f}")
        
        try:
            auc = roc_auc_score(y, y_proba)
            print(f"  Training AUC: {auc:.3f}")
        except:
            auc = None
            print(f"  Training AUC: N/A (requires multiple classes)")
        
        # Precision-recall at different thresholds
        precision, recall, thresholds = precision_recall_curve(y, y_proba)
        
        # Find operating point with 90% recall (catch most failures)
        target_recall = 0.90
        idx = np.argmin(np.abs(recall - target_recall))
        op_precision = precision[idx]
        op_recall = recall[idx]
        op_threshold = thresholds[idx] if idx < len(thresholds) else 0.5
        
        print(f"  Operating point (recall={target_recall}):")
        print(f"    Precision: {op_precision:.3f}")
        print(f"    Recall: {op_recall:.3f}")
        print(f"    Threshold: {op_threshold:.3f}")
        print()
        
        # Estimate CI time savings
        # Assume: running all tests takes 90s, selected tests save proportional time
        # False negatives cost: missed failures that cause debugging time
        all_tests_time = 90  # seconds (from claim)
        test_reduction = 1 - op_recall  # proportion of tests we skip
        time_saved = all_tests_time * test_reduction
        new_time = all_tests_time - time_saved
        reduction_pct = (time_saved / all_tests_time) * 100
        
        print(f"Expected CI time impact (90% recall):")
        print(f"  Baseline (all tests): {all_tests_time}s")
        print(f"  With ML selection: {new_time:.1f}s")
        print(f"  Time saved: {time_saved:.1f}s ({reduction_pct:.1f}%)")
        print()
        
        # Compile results
        results = {
            "data_source": str(data_path),
            "model_source": str(model_path),
            "n_samples": int(n_total),
            "n_failures": int(n_failures),
            "n_passes": int(n_passes),
            "failure_rate": float(failure_rate),
            "failure_rate_ci_95": [float(fail_ci_low), float(fail_ci_high)],
            "cv_f1_mean": float(cv_mean),
            "cv_f1_std": float(cv_std),
            "training_f1": float(f1),
            "training_auc": float(auc) if auc is not None else None,
            "operating_point": {
                "target_recall": float(target_recall),
                "precision": float(op_precision),
                "recall": float(op_recall),
                "threshold": float(op_threshold)
            },
            "ci_time_estimate": {
                "baseline_seconds": float(all_tests_time),
                "with_ml_seconds": float(new_time),
                "time_saved_seconds": float(time_saved),
                "reduction_percent": float(reduction_pct),
                "note": "Estimate based on synthetic data; real reduction TBD"
            },
            "data_type": "synthetic_baseline",
            "warnings": [
                "Model trained on synthetic data (N=100)",
                "Real-world performance requires 50+ real test runs",
                "CI time reduction is estimated, not measured",
                "False negative rate needs production validation"
            ]
        }
    else:
        print(f"✗ Model file not found: {model_path}")
        results = {
            "error": "Model file not found",
            "data_source": str(data_path),
            "n_samples": int(n_total),
            "n_failures": int(n_failures)
        }
    
    # Write output
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"✓ Results written to {output_path}")
    print("=" * 80)


if __name__ == "__main__":
    main()
