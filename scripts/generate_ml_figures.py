#!/usr/bin/env python3
"""Generate ML test selection figures with publication-quality styling.

Usage:
    python scripts/generate_ml_figures.py --output figs/
"""

import argparse
import json
import numpy as np
from pathlib import Path
from sklearn.metrics import precision_recall_curve, roc_curve, auc
import joblib

# Check for matplotlib
try:
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend
    import matplotlib.pyplot as plt
    import seaborn as sns
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    print("⚠️  matplotlib not installed (pip install matplotlib seaborn)")


def main():
    parser = argparse.ArgumentParser(description="Generate ML figures")
    parser.add_argument("--model", default="test_selector.pkl",
                        help="Path to trained model")
    parser.add_argument("--data", default="training_data.json",
                        help="Path to training data")
    parser.add_argument("--output", default="figs/",
                        help="Output directory")
    args = parser.parse_args()
    
    if not MATPLOTLIB_AVAILABLE:
        print("❌ Cannot generate figures without matplotlib")
        print("   Install: pip install matplotlib seaborn")
        return
    
    print("=" * 80)
    print("ML TEST SELECTION - FIGURE GENERATION")
    print("=" * 80)
    print()
    
    # Load data
    data_path = Path(args.data)
    if not data_path.exists():
        print(f"✗ Data file not found: {data_path}")
        return
    
    with open(data_path) as f:
        records = json.load(f)
    
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
    
    # Load model
    model_path = Path(args.model)
    if not model_path.exists():
        print(f"✗ Model file not found: {model_path}")
        return
    
    model = joblib.load(model_path)
    y_proba = model.predict_proba(X)[:, 1]
    
    # Setup output
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Set style
    sns.set_style("whitegrid")
    plt.rcParams['figure.figsize'] = (10, 6)
    plt.rcParams['font.size'] = 12
    
    # Figure 1: Precision-Recall Curve
    print("📊 Generating precision-recall curve...")
    precision, recall, thresholds = precision_recall_curve(y, y_proba)
    
    fig, ax = plt.subplots()
    ax.plot(recall, precision, 'b-', linewidth=2, label='PR Curve')
    ax.axhline(y=np.mean(y), color='r', linestyle='--', 
               label=f'Baseline (failure rate={np.mean(y):.2f})')
    
    # Mark operating point (90% recall)
    target_recall = 0.90
    idx = np.argmin(np.abs(recall - target_recall))
    ax.plot(recall[idx], precision[idx], 'ro', markersize=10, 
            label=f'Operating Point (recall={recall[idx]:.2f}, precision={precision[idx]:.2f})')
    
    ax.set_xlabel('Recall', fontsize=14)
    ax.set_ylabel('Precision', fontsize=14)
    ax.set_title('ML Test Selection: Precision-Recall Curve (N=100, synthetic)', fontsize=16)
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0, 1.05])
    ax.set_ylim([0, 1.05])
    
    pr_path = output_dir / "precision_recall_curve.png"
    plt.tight_layout()
    plt.savefig(pr_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: {pr_path}")
    
    # Figure 2: Feature Importance
    print("📊 Generating feature importance...")
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1]
        
        fig, ax = plt.subplots()
        colors = sns.color_palette("viridis", len(feature_names))
        ax.barh(range(len(feature_names)), importances[indices], color=colors)
        ax.set_yticks(range(len(feature_names)))
        ax.set_yticklabels([feature_names[i] for i in indices])
        ax.set_xlabel('Feature Importance', fontsize=14)
        ax.set_title('ML Test Selection: Feature Importance (RandomForest)', fontsize=16)
        ax.grid(True, alpha=0.3, axis='x')
        
        fi_path = output_dir / "feature_importance.png"
        plt.tight_layout()
        plt.savefig(fi_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✓ Saved: {fi_path}")
    
    # Figure 3: Utility Curve (Expected CI time savings vs threshold)
    print("📊 Generating utility curve...")
    all_tests_time = 90  # seconds
    thresholds_eval = np.linspace(0, 1, 100)
    utilities = []
    
    for thresh in thresholds_eval:
        y_pred = (y_proba >= thresh).astype(int)
        # Tests selected = predicted failures + some false positives
        tests_selected = np.sum(y_pred)
        test_fraction = tests_selected / len(y)
        time_with_ml = all_tests_time * test_fraction
        time_saved = all_tests_time - time_with_ml
        # False negatives cost debugging time (assume 60 min per missed failure)
        false_negatives = np.sum((y_pred == 0) & (y == 1))
        fn_cost = false_negatives * 60 * 60  # seconds
        # Net utility
        utility = time_saved - (fn_cost * 0.01)  # Scale FN cost
        utilities.append(utility)
    
    fig, ax = plt.subplots()
    ax.plot(thresholds_eval, utilities, 'b-', linewidth=2)
    max_idx = np.argmax(utilities)
    ax.plot(thresholds_eval[max_idx], utilities[max_idx], 'ro', markersize=10,
            label=f'Optimal (threshold={thresholds_eval[max_idx]:.2f})')
    
    ax.set_xlabel('Decision Threshold', fontsize=14)
    ax.set_ylabel('Expected Utility (time saved - FN cost, seconds)', fontsize=14)
    ax.set_title('ML Test Selection: Expected Utility vs Threshold', fontsize=16)
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0, color='k', linestyle='-', linewidth=0.5)
    
    util_path = output_dir / "utility_curve.png"
    plt.tight_layout()
    plt.savefig(util_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: {util_path}")
    
    print()
    print("=" * 80)
    print("✓ FIGURE GENERATION COMPLETE")
    print("=" * 80)
    print(f"  Output directory: {output_dir}")
    print(f"  Figures generated: 3")
    print()


if __name__ == "__main__":
    main()
