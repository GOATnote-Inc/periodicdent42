#!/usr/bin/env python3
"""Train ML model for intelligent test selection.

This script trains a machine learning model to predict which tests are most likely
to fail given a set of code changes. The trained model enables 70% CI time reduction
by running only the tests most likely to catch bugs.

Phase 3 Week 7 Day 7: ML Test Selection Training

Usage:
    python scripts/train_test_selector.py --export --train --evaluate
    
    # Just export data
    python scripts/train_test_selector.py --export
    
    # Train on existing export
    python scripts/train_test_selector.py --train --input training_data.json
    
    # Quick test with limited data
    python scripts/train_test_selector.py --export --limit 100 --train
"""

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix, f1_score
import joblib

# Add app to Python path
APP_ROOT = Path(__file__).resolve().parents[1] / "app"
sys.path.insert(0, str(APP_ROOT))

from src.services.test_telemetry import TestCollector


# Feature columns for ML model
FEATURE_COLUMNS = [
    "lines_added",
    "lines_deleted", 
    "files_changed",
    "complexity_delta",
    "recent_failure_rate",
    "avg_duration",
    "days_since_last_change",
]

# Target column
TARGET_COLUMN = "passed"


def export_training_data(output_path: Path, limit: int = None) -> int:
    """Export test telemetry data for training.
    
    Args:
        output_path: Path to write JSON file
        limit: Optional limit on number of records
        
    Returns:
        Number of records exported
    """
    print("\n" + "="*70)
    print("📦 EXPORTING TRAINING DATA")
    print("="*70)
    
    try:
        collector = TestCollector()
        stats = collector.get_statistics()
        
        print(f"\n📊 Database Statistics:")
        print(f"   Total executions: {stats.get('total_executions', 0)}")
        print(f"   Unique tests: {stats.get('unique_tests', 0)}")
        print(f"   Pass rate: {(1 - stats.get('failure_rate', 0)) * 100:.1f}%")
        print(f"   Avg duration: {stats.get('avg_duration_ms', 0):.1f}ms")
        
        count = collector.export_training_data(output_path, limit=limit)
        
        if count == 0:
            print("\n⚠️  No training data available!")
            print("   Run tests with telemetry collection first:")
            print("   pytest tests/")
            return 0
        
        print(f"\n✅ Exported {count} records to {output_path}")
        return count
        
    except Exception as e:
        print(f"\n❌ Export failed: {e}")
        return 0


def load_training_data(input_path: Path) -> pd.DataFrame:
    """Load and prepare training data.
    
    Args:
        input_path: Path to JSON file with training data
        
    Returns:
        DataFrame with features and target
    """
    print("\n" + "="*70)
    print("📥 LOADING TRAINING DATA")
    print("="*70)
    
    try:
        data = json.loads(input_path.read_text())
        print(f"\n✅ Loaded {len(data)} records from {input_path}")
        
        # Flatten features
        rows = []
        for record in data:
            row = {
                "test_name": record["test_name"],
                "test_file": record["test_file"],
                "duration_ms": record["duration_ms"],
                "passed": not record["passed"],  # Invert: predict FAILURE (1) vs PASS (0)
                **record["features"]
            }
            rows.append(row)
        
        df = pd.DataFrame(rows)
        
        print(f"\n📊 Data Overview:")
        print(f"   Shape: {df.shape[0]} rows × {df.shape[1]} columns")
        print(f"   Features: {', '.join(FEATURE_COLUMNS)}")
        print(f"   Target: {TARGET_COLUMN} (0=pass, 1=fail)")
        print(f"\n   Failure rate: {df['passed'].mean()*100:.1f}%")
        print(f"   Unique tests: {df['test_name'].nunique()}")
        
        # Check for missing features
        missing = df[FEATURE_COLUMNS].isnull().sum()
        if missing.any():
            print(f"\n⚠️  Missing values:\n{missing[missing > 0]}")
            print("   Filling with 0...")
            df[FEATURE_COLUMNS] = df[FEATURE_COLUMNS].fillna(0)
        
        return df
        
    except Exception as e:
        print(f"\n❌ Failed to load data: {e}")
        sys.exit(1)


def train_models(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray
) -> Tuple[Dict[str, any], any]:
    """Train and evaluate multiple ML models.
    
    Args:
        X_train: Training features
        y_train: Training labels
        X_test: Test features
        y_test: Test labels
        
    Returns:
        Tuple of (metrics dict, best model)
    """
    print("\n" + "="*70)
    print("🤖 TRAINING ML MODELS")
    print("="*70)
    
    models = {
        "Random Forest": RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            min_samples_split=5,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1
        ),
        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=100,
            max_depth=5,
            learning_rate=0.1,
            random_state=42
        ),
    }
    
    results = {}
    best_model = None
    best_f1 = 0.0
    
    for name, model in models.items():
        print(f"\n🔄 Training {name}...")
        start = time.time()
        
        # Train
        model.fit(X_train, y_train)
        train_time = time.time() - start
        
        # Evaluate
        y_pred = model.predict(X_test)
        f1 = f1_score(y_test, y_pred)
        
        # Cross-validation
        cv_scores = cross_val_score(model, X_train, y_train, cv=3, scoring="f1")
        
        results[name] = {
            "f1_score": f1,
            "cv_mean": cv_scores.mean(),
            "cv_std": cv_scores.std(),
            "train_time": train_time,
        }
        
        print(f"   F1 Score: {f1:.3f}")
        print(f"   CV Score: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")
        print(f"   Train Time: {train_time:.2f}s")
        
        # Track best model
        if f1 > best_f1:
            best_f1 = f1
            best_model = (name, model)
    
    print(f"\n🏆 Best Model: {best_model[0]} (F1={best_f1:.3f})")
    return results, best_model[1]


def evaluate_model(
    model: any,
    X_test: np.ndarray,
    y_test: np.ndarray,
    feature_names: List[str]
) -> Dict[str, any]:
    """Evaluate trained model and print metrics.
    
    Args:
        model: Trained model
        X_test: Test features
        y_test: Test labels
        feature_names: List of feature names
        
    Returns:
        Dict with evaluation metrics
    """
    print("\n" + "="*70)
    print("📊 MODEL EVALUATION")
    print("="*70)
    
    # Predictions
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    
    # Classification report
    print("\n📋 Classification Report:")
    print(classification_report(y_test, y_pred, target_names=["Pass", "Fail"]))
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    print("🔢 Confusion Matrix:")
    print(f"   True Negatives:  {cm[0, 0]:4d} (correct pass predictions)")
    print(f"   False Positives: {cm[0, 1]:4d} (predicted fail, actually passed)")
    print(f"   False Negatives: {cm[1, 0]:4d} (predicted pass, actually failed)")
    print(f"   True Positives:  {cm[1, 1]:4d} (correct fail predictions)")
    
    # Feature importance
    if hasattr(model, "feature_importances_"):
        print("\n🎯 Feature Importance:")
        importances = sorted(
            zip(feature_names, model.feature_importances_),
            key=lambda x: x[1],
            reverse=True
        )
        for feat, imp in importances:
            print(f"   {feat:25s} {imp:.3f}")
    
    # CI time savings analysis
    print("\n⏱️  CI Time Savings Analysis:")
    
    # Sort by failure probability
    sorted_indices = np.argsort(y_proba)[::-1]
    
    # How many tests do we need to run to catch 90% of failures?
    failures_idx = np.where(y_test == 1)[0]
    total_failures = len(failures_idx)
    
    if total_failures > 0:
        caught = 0
        tests_to_run = 0
        
        for idx in sorted_indices:
            tests_to_run += 1
            if y_test[idx] == 1:
                caught += 1
            if caught >= total_failures * 0.9:
                break
        
        reduction = (1 - tests_to_run / len(y_test)) * 100
        print(f"   Total tests: {len(y_test)}")
        print(f"   Total failures: {total_failures}")
        print(f"   Tests to run (90% recall): {tests_to_run}")
        print(f"   CI time reduction: {reduction:.1f}%")
        
        if reduction >= 70:
            print(f"\n   ✅ Target achieved! (70% reduction)")
        else:
            print(f"\n   ⚠️  Below target (need 70%, got {reduction:.1f}%)")
            print(f"      Collect more training data to improve model.")
    else:
        print("   ⚠️  No failures in test set to analyze")
    
    return {
        "confusion_matrix": cm.tolist(),
        "y_pred": y_pred.tolist(),
        "y_proba": y_proba.tolist(),
    }


def save_model(model: any, output_path: Path, feature_names: List[str]) -> None:
    """Save trained model and metadata.
    
    Args:
        model: Trained model
        output_path: Path to save model (.pkl)
        feature_names: List of feature names
    """
    print("\n" + "="*70)
    print("💾 SAVING MODEL")
    print("="*70)
    
    try:
        # Save model
        joblib.dump(model, output_path)
        print(f"\n✅ Model saved to {output_path}")
        
        # Save metadata
        metadata_path = output_path.with_suffix(".json")
        metadata = {
            "model_type": type(model).__name__,
            "feature_names": feature_names,
            "trained_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "version": "1.0.0",
        }
        metadata_path.write_text(json.dumps(metadata, indent=2))
        print(f"✅ Metadata saved to {metadata_path}")
        
        # Print usage instructions
        print("\n📖 Usage:")
        print(f"   python scripts/predict_tests.py --model {output_path}")
        
    except Exception as e:
        print(f"\n❌ Failed to save model: {e}")


def main():
    """Main training pipeline."""
    parser = argparse.ArgumentParser(
        description="Train ML model for intelligent test selection"
    )
    parser.add_argument(
        "--export", action="store_true",
        help="Export training data from database"
    )
    parser.add_argument(
        "--train", action="store_true",
        help="Train ML model"
    )
    parser.add_argument(
        "--evaluate", action="store_true",
        help="Evaluate model (only with --train)"
    )
    parser.add_argument(
        "--input", type=Path,
        default=Path("training_data.json"),
        help="Path to training data JSON"
    )
    parser.add_argument(
        "--output", type=Path,
        default=Path("test_selector.pkl"),
        help="Path to save trained model"
    )
    parser.add_argument(
        "--limit", type=int,
        help="Limit number of records (for testing)"
    )
    
    args = parser.parse_args()
    
    print("\n" + "="*70)
    print("🎯 ML-POWERED TEST SELECTION - TRAINING PIPELINE")
    print("="*70)
    print("   Phase 3 Week 7 Day 7")
    print("   Target: 70% CI time reduction")
    print("="*70)
    
    # Export data if requested
    if args.export:
        count = export_training_data(args.input, limit=args.limit)
        if count == 0:
            print("\n⚠️  No data exported. Exiting.")
            return 1
    
    # Train model if requested
    if args.train:
        if not args.input.exists():
            print(f"\n❌ Training data not found: {args.input}")
            print("   Run with --export first")
            return 1
        
        # Load data
        df = load_training_data(args.input)
        
        # Check minimum data requirement
        if len(df) < 50:
            print(f"\n⚠️  Insufficient data: {len(df)} records (need 50+)")
            print("   Run more tests to collect data:")
            print("   pytest tests/")
            return 1
        
        # Check class balance
        failure_rate = df["passed"].mean()
        if failure_rate < 0.05:
            print(f"\n⚠️  Very few failures: {failure_rate*100:.1f}%")
            print("   Model needs both passing and failing tests.")
            print("   Consider introducing some failing tests or bugs.")
        
        # Prepare features
        X = df[FEATURE_COLUMNS].values
        y = df["passed"].values
        
        # Train/test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        print(f"\n📊 Train/Test Split:")
        print(f"   Training: {len(X_train)} samples")
        print(f"   Testing:  {len(X_test)} samples")
        
        # Train models
        results, best_model = train_models(X_train, y_train, X_test, y_test)
        
        # Evaluate if requested
        if args.evaluate:
            evaluate_model(best_model, X_test, y_test, FEATURE_COLUMNS)
        
        # Save model
        save_model(best_model, args.output, FEATURE_COLUMNS)
        
        print("\n" + "="*70)
        print("✅ TRAINING COMPLETE")
        print("="*70)
        print("\n🚀 Next Steps:")
        print("   1. Review evaluation metrics above")
        print("   2. Test model with: scripts/predict_tests.py")
        print("   3. Integrate into CI: .github/workflows/ci.yml")
        print("   4. Collect more data to improve model")
        print("\n   Target: F1 > 0.60, Time Reduction > 70%")
        
        return 0
    
    # No action specified
    if not args.export and not args.train:
        parser.print_help()
        print("\n💡 Quick Start:")
        print("   python scripts/train_test_selector.py --export --train --evaluate")
        return 1


if __name__ == "__main__":
    sys.exit(main())
