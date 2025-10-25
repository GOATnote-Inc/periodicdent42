#!/usr/bin/env python3
"""Train ML model for intelligent test selection.

This script trains a baseline ML model to predict which tests should run
based on CI run metadata. It gracefully handles insufficient data and
provides clear feedback on training progress.

Usage:
    # Train on collected CI data (reproducible with seed)
    python scripts/train_selector.py --seed 42
    
    # Custom paths
    python scripts/train_selector.py --data data/ci_runs.jsonl --out models/selector-v2.pkl
    
    # With verbose output
    python scripts/train_selector.py --verbose
"""

import argparse
import hashlib
import json
import os
import pathlib
import random
import subprocess
import sys
import datetime as dt
from typing import List, Dict, Any, Optional


def load_jsonl(path: pathlib.Path) -> List[Dict[str, Any]]:
    """Load JSONL file into list of dictionaries.
    
    Args:
        path: Path to JSONL file
        
    Returns:
        List of parsed JSON objects
    """
    if not path.exists():
        return []
    
    lines = path.read_text(encoding="utf-8").splitlines()
    return [json.loads(line) for line in lines if line.strip()]


def write_stub_model(out_path: pathlib.Path, meta_path: pathlib.Path, rows: int, note: str) -> None:
    """Write stub model placeholder when insufficient data.
    
    Args:
        out_path: Path for model file
        meta_path: Path for metadata file
        rows: Number of data rows
        note: Explanatory note
    """
    out_path.write_bytes(b"STUB-MODEL-INSUFFICIENT-DATA")
    
    metadata = {
        "timestamp": dt.datetime.utcnow().isoformat() + "Z",
        "rows": rows,
        "f1": None,
        "precision": None,
        "recall": None,
        "note": note,
        "status": "stub"
    }
    
    meta_path.write_text(json.dumps(metadata, indent=2))


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
        "SKLEARN_SEED",
        "NUMPY_SEED",
    ])
    env_str = "|".join(f"{k}={os.getenv(k, '')}" for k in env_keys)
    return hashlib.sha256(env_str.encode()).hexdigest()[:16]


def emit_training_metadata(seed: Optional[int], artifact_dir: pathlib.Path) -> None:
    """Emit reproducibility metadata to artifact/train_meta.json.
    
    Args:
        seed: Random seed used (None if not set)
        artifact_dir: Directory for metadata artifact
    """
    metadata = {
        "timestamp": dt.datetime.utcnow().isoformat() + "Z",
        "git_sha": get_git_sha(),
        "env_hash": get_env_hash(),
        "seed": seed,
        "script": "train_selector.py",
        "reproducible": seed is not None,
    }
    
    artifact_dir.mkdir(parents=True, exist_ok=True)
    
    meta_path = artifact_dir / "train_meta.json"
    with meta_path.open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    
    print(f"📝 Emitted training metadata to {meta_path}", flush=True)


def train_baseline_model(
    data_path: pathlib.Path,
    out_path: pathlib.Path,
    meta_path: pathlib.Path,
    verbose: bool = False,
    seed: Optional[int] = 42
) -> int:
    """Train baseline ML model for test selection.
    
    Args:
        data_path: Path to CI runs JSONL
        out_path: Path to save trained model
        meta_path: Path to save metadata
        verbose: Print detailed progress
        seed: Random seed for reproducibility
        
    Returns:
        0 on success (even with stub model)
    """
    # Load data
    rows = load_jsonl(data_path)
    n = len(rows)
    
    print(f"📊 Loaded {n} CI run records from {data_path}", flush=True)
    
    # Check if we have enough data
    MIN_ROWS = 50
    if n < MIN_ROWS:
        note = (
            f"Insufficient data ({n}/{MIN_ROWS} rows). "
            f"Collect {MIN_ROWS - n} more CI runs to train baseline model."
        )
        print(f"⚠️  {note}", flush=True)
        print(f"💡 Generate mock data: python scripts/collect_ci_runs.py --mock", flush=True)
        
        write_stub_model(out_path, meta_path, n, note)
        return 0
    
    # Check for ML dependencies
    try:
        import pandas as pd
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import f1_score, precision_score, recall_score, classification_report
        from sklearn.ensemble import GradientBoostingClassifier
        import joblib
    except ImportError as e:
        note = f"Missing ML dependencies: {e}. Install: pip install pandas scikit-learn joblib"
        print(f"⚠️  {note}", flush=True)
        write_stub_model(out_path, meta_path, n, note)
        return 0
    
    # Prepare data
    print(f"🔧 Preparing training data...", flush=True)
    df = pd.DataFrame(rows)
    
    # Create binary target: did any tests fail?
    if "tests_failed" in df.columns:
        df["has_failures"] = (df["tests_failed"] > 0).astype(int)
    else:
        df["has_failures"] = 0
    
    # Select features
    feature_cols = [
        "duration_sec",
        "tests_total",
        "changed_files",
        "lines_added",
        "lines_deleted",
    ]
    
    # Ensure all features exist
    for col in feature_cols:
        if col not in df.columns:
            df[col] = 0
    
    X = df[feature_cols].fillna(0)
    y = df["has_failures"]
    
    # Check class balance
    failure_rate = y.mean()
    print(f"📈 Failure rate: {failure_rate:.1%} ({y.sum()}/{len(y)} runs)", flush=True)
    
    if failure_rate < 0.01 or failure_rate > 0.99:
        note = f"Imbalanced data: {failure_rate:.1%} failure rate. Need more diverse CI runs."
        print(f"⚠️  {note}", flush=True)
        write_stub_model(out_path, meta_path, n, note)
        return 0
    
    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=seed, stratify=y
    )
    
    print(f"📦 Training set: {len(X_train)} samples", flush=True)
    print(f"📦 Test set: {len(X_test)} samples", flush=True)
    
    # Train model
    print(f"🤖 Training Gradient Boosting classifier (seed={seed})...", flush=True)
    clf = GradientBoostingClassifier(
        n_estimators=100,
        max_depth=5,
        learning_rate=0.1,
        random_state=seed
    )
    clf.fit(X_train, y_train)
    
    # Evaluate
    print(f"📊 Evaluating model...", flush=True)
    y_pred = clf.predict(X_test)
    
    f1 = float(f1_score(y_test, y_pred))
    precision = float(precision_score(y_test, y_pred, zero_division=0))
    recall = float(recall_score(y_test, y_pred, zero_division=0))
    
    print(f"\n✅ Model trained successfully!", flush=True)
    print(f"   F1 Score: {f1:.3f}", flush=True)
    print(f"   Precision: {precision:.3f}", flush=True)
    print(f"   Recall: {recall:.3f}", flush=True)
    
    if verbose:
        print(f"\n📋 Classification Report:", flush=True)
        print(classification_report(y_test, y_pred), flush=True)
        
        # Feature importance
        importances = pd.DataFrame({
            'feature': feature_cols,
            'importance': clf.feature_importances_
        }).sort_values('importance', ascending=False)
        
        print(f"\n📊 Feature Importance:", flush=True)
        for _, row in importances.iterrows():
            print(f"   {row['feature']}: {row['importance']:.3f}", flush=True)
    
    # Save model
    joblib.dump(clf, out_path)
    print(f"\n💾 Saved model to {out_path}", flush=True)
    
    # Save metadata
    metadata = {
        "timestamp": dt.datetime.utcnow().isoformat() + "Z",
        "rows": int(n),
        "train_samples": int(len(X_train)),
        "test_samples": int(len(X_test)),
        "features": feature_cols,
        "f1": f1,
        "precision": precision,
        "recall": recall,
        "failure_rate": float(failure_rate),
        "seed": seed,
        "reproducible": True,
        "status": "trained"
    }
    
    meta_path.write_text(json.dumps(metadata, indent=2))
    print(f"💾 Saved metadata to {meta_path}", flush=True)
    
    # Emit reproducibility metadata
    artifact_dir = out_path.parent.parent / "artifact"
    emit_training_metadata(seed, artifact_dir)
    
    return 0


def main() -> int:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Train ML model for test selection"
    )
    parser.add_argument(
        "--data",
        default="data/ci_runs.jsonl",
        help="Input JSONL file with CI runs (default: data/ci_runs.jsonl)"
    )
    parser.add_argument(
        "--out",
        default="models/selector-v1.pkl",
        help="Output model file (default: models/selector-v1.pkl)"
    )
    parser.add_argument(
        "--meta",
        default="models/metadata.json",
        help="Output metadata file (default: models/metadata.json)"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print detailed training progress"
    )
    parser.add_argument(
        "--seed",
        type=int,
        metavar="N",
        default=42,
        help="Random seed for reproducibility (default: 42)"
    )
    
    args = parser.parse_args()
    
    # Ensure models directory exists
    models_dir = pathlib.Path(args.out).parent
    models_dir.mkdir(parents=True, exist_ok=True)
    
    # Convert to Path objects
    data_path = pathlib.Path(args.data)
    out_path = pathlib.Path(args.out)
    meta_path = pathlib.Path(args.meta)
    
    print("=" * 80, flush=True)
    print("🎯 ML Test Selection - Model Training", flush=True)
    print("=" * 80, flush=True)
    
    if args.seed is not None:
        random.seed(args.seed)
        print(f"🎲 Using random seed: {args.seed}", flush=True)
    
    result = train_baseline_model(data_path, out_path, meta_path, args.verbose, args.seed)
    
    print("=" * 80, flush=True)
    
    return result


if __name__ == "__main__":
    sys.exit(main())
