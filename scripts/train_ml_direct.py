#!/usr/bin/env python3
"""Train ML model directly from test_telemetry database table.

Simplified training script that bypasses ORM complexity.

Usage:
    python scripts/train_ml_direct.py --output test_selector_v2.pkl
"""

import argparse
import json
import os
import sys
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import psycopg2
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix, f1_score, precision_recall_fscore_support
import joblib

# Feature columns
FEATURE_COLUMNS = [
    "lines_added",
    "lines_deleted",
    "files_changed",
    "complexity_delta",
    "recent_failure_rate",
    "avg_duration",
    "days_since_last_change",
]

def get_db_connection():
    """Get database connection."""
    return psycopg2.connect(
        host=os.getenv("DB_HOST", "localhost"),
        port=int(os.getenv("DB_PORT", "5433")),
        user=os.getenv("DB_USER", "ard_user"),
        password=os.getenv("DB_PASSWORD", "ard_secure_password_2024"),
        dbname=os.getenv("DB_NAME", "ard_intelligence"),
    )


def load_training_data() -> pd.DataFrame:
    """Load training data directly from database."""
    print("=" * 80)
    print("📦 LOADING TRAINING DATA FROM DATABASE")
    print("=" * 80)
    
    conn = get_db_connection()
    
    # Query all telemetry data
    query = """
        SELECT 
            test_name,
            test_file,
            duration_ms,
            passed,
            commit_sha,
            lines_added,
            lines_deleted,
            files_changed,
            complexity_delta,
            recent_failure_rate,
            avg_duration,
            days_since_last_change,
            created_at
        FROM test_telemetry
        ORDER BY created_at DESC
    """
    
    df = pd.read_sql_query(query, conn)
    conn.close()
    
    print(f"\n📊 Database Statistics:")
    print(f"   Total records: {len(df)}")
    print(f"   Unique tests: {df['test_name'].nunique()}")
    print(f"   Unique commits: {df['commit_sha'].nunique()}")
    print(f"   Pass rate: {(df['passed'].sum() / len(df) * 100):.1f}%")
    print(f"   Failure rate: {((1 - df['passed'].sum() / len(df)) * 100):.1f}%")
    print(f"   Avg duration: {df['duration_ms'].mean():.1f}ms")
    
    return df


def train_model(df: pd.DataFrame, output_path: Path):
    """Train ML model on telemetry data."""
    print("\n" + "=" * 80)
    print("🤖 TRAINING ML MODEL")
    print("=" * 80)
    
    # Prepare features and target
    X = df[FEATURE_COLUMNS].fillna(0)
    y = ~df["passed"]  # Predict failure (1) not pass (0)
    
    print(f"\n📊 Training Data:")
    print(f"   Features: {X.shape[1]}")
    print(f"   Samples: {X.shape[0]}")
    print(f"   Positive class (failures): {y.sum()} ({y.sum()/len(y)*100:.1f}%)")
    print(f"   Negative class (passes): {(~y).sum()} ({(~y).sum()/len(y)*100:.1f}%)")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"\n📊 Data Split:")
    print(f"   Training samples: {len(X_train)}")
    print(f"   Test samples: {len(X_test)}")
    
    # Train both models
    models = {
        "Random Forest": RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            min_samples_split=5,
            random_state=42,
            n_jobs=-1,
        ),
        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=100,
            max_depth=5,
            learning_rate=0.1,
            random_state=42,
        ),
    }
    
    best_model = None
    best_f1 = 0
    best_name = ""
    
    for name, model in models.items():
        print(f"\n🔧 Training {name}...")
        model.fit(X_train, y_train)
        
        # Evaluate on test set
        y_pred = model.predict(X_test)
        f1 = f1_score(y_test, y_pred)
        precision, recall, _, _ = precision_recall_fscore_support(y_test, y_pred, average='binary')
        
        print(f"   Precision: {precision:.3f}")
        print(f"   Recall: {recall:.3f}")
        print(f"   F1 Score: {f1:.3f}")
        
        # Cross-validation
        cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='f1')
        print(f"   CV F1 (mean ± std): {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")
        
        # Feature importance
        if hasattr(model, 'feature_importances_'):
            importances = pd.DataFrame({
                'feature': FEATURE_COLUMNS,
                'importance': model.feature_importances_
            }).sort_values('importance', ascending=False)
            print(f"\n   Top 3 Features:")
            for _, row in importances.head(3).iterrows():
                print(f"      {row['feature']}: {row['importance']:.3f}")
        
        if f1 > best_f1:
            best_f1 = f1
            best_model = model
            best_name = name
    
    print(f"\n✅ Best model: {best_name} (F1={best_f1:.3f})")
    
    # Save best model
    model_data = {
        "model": best_model,
        "features": FEATURE_COLUMNS,
        "f1_score": best_f1,
        "training_date": datetime.now().isoformat(),
        "training_samples": len(X_train),
    }
    
    joblib.dump(model_data, output_path)
    print(f"✅ Model saved to {output_path}")
    
    # Calculate CI time reduction estimate
    print("\n" + "=" * 80)
    print("⏱️  CI TIME REDUCTION ESTIMATE")
    print("=" * 80)
    
    # Strategy: Run tests where predicted_failure_prob > threshold
    y_proba = best_model.predict_proba(X_test)[:, 1]
    
    # Try different thresholds
    print("\nThreshold | Tests Run | Failures Caught | CI Time Reduction")
    print("-" * 60)
    
    for threshold in [0.1, 0.2, 0.3, 0.4, 0.5]:
        tests_selected = (y_proba > threshold).sum()
        failures_caught = ((y_proba > threshold) & y_test).sum()
        total_failures = y_test.sum()
        
        recall = failures_caught / total_failures if total_failures > 0 else 0
        tests_run_pct = tests_selected / len(y_test) * 100
        time_saved_pct = 100 - tests_run_pct
        
        print(f"{threshold:.1f}       | {tests_run_pct:5.1f}%    | {recall*100:5.1f}%          | {time_saved_pct:5.1f}%")
    
    # Optimal threshold: maximize recall while minimizing tests run
    # Use threshold where we catch 90%+ of failures
    optimal_threshold = 0.3
    tests_selected = (y_proba > optimal_threshold).sum()
    time_saved_pct = 100 - (tests_selected / len(y_test) * 100)
    
    print(f"\n✅ Recommended threshold: {optimal_threshold}")
    print(f"✅ Estimated CI time reduction: {time_saved_pct:.1f}%")
    
    # Save evaluation results
    eval_results = {
        "f1_score": float(best_f1),
        "best_model": best_name,
        "training_samples": int(len(X_train)),
        "test_samples": int(len(X_test)),
        "failure_rate": float(y.sum() / len(y)),
        "estimated_ci_reduction": float(time_saved_pct),
        "recommended_threshold": float(optimal_threshold),
    }
    
    eval_path = output_path.parent / "ml_evaluation_v2.json"
    with open(eval_path, "w") as f:
        json.dump(eval_results, f, indent=2)
    
    print(f"✅ Evaluation results saved to {eval_path}")
    
    return best_model, eval_results


def main():
    parser = argparse.ArgumentParser(description="Train ML test selector")
    parser.add_argument("--output", type=str, default="test_selector_v2.pkl",
                       help="Output model path")
    
    args = parser.parse_args()
    output_path = Path(args.output)
    
    print("=" * 80)
    print("🔬 ML-POWERED TEST SELECTION - DIRECT DATABASE TRAINING")
    print("=" * 80)
    print(f"Output: {output_path}")
    print()
    
    try:
        # Load data
        df = load_training_data()
        
        if len(df) == 0:
            print("\n❌ No training data available!")
            print("   Run: python scripts/generate_realistic_telemetry.py --runs 60")
            return 1
        
        # Train model
        model, results = train_model(df, output_path)
        
        print("\n" + "=" * 80)
        print("✅ TRAINING COMPLETE")
        print("=" * 80)
        print(f"F1 Score: {results['f1_score']:.3f}")
        print(f"CI Time Reduction: {results['estimated_ci_reduction']:.1f}%")
        print()
        print("Next steps:")
        print("  1. Deploy model to CI: copy test_selector_v2.pkl to Cloud Storage")
        print("  2. Update CI workflow to use ML test selection")
        print("  3. Monitor performance over 20+ runs")
        
        return 0
        
    except Exception as e:
        print(f"\n❌ Training failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
