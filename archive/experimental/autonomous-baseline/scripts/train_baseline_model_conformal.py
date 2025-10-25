#!/usr/bin/env python3
"""
Train baseline RF+QRF model with Split Conformal Prediction for calibrated uncertainty.

This script adds conformal calibration to the baseline model to ensure:
- ECE < 0.05
- PICP@95% ∈ [0.94, 0.96]

Usage:
    python scripts/train_baseline_model_conformal.py \\
        --data data/processed/uci_splits/ \\
        --output models/
"""

import argparse
import json
import pickle
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

# Add src to path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.rf_qrf import RandomForestQRF
from src.uncertainty.conformal import SplitConformalPredictor


def load_splits(data_dir: Path) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load train/val/test splits."""
    train_df = pd.read_csv(data_dir / 'train.csv')
    val_df = pd.read_csv(data_dir / 'val.csv')
    test_df = pd.read_csv(data_dir / 'test.csv')
    
    print(f"📂 Loaded data from {data_dir}")
    print(f"   Train: {len(train_df)} samples")
    print(f"   Val:   {len(val_df)} samples (for conformal calibration)")
    print(f"   Test:  {len(test_df)} samples")
    
    return train_df, val_df, test_df


def prepare_data(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    target_col: str = 'critical_temp'
) -> Tuple:
    """Prepare features and targets."""
    feature_cols = [c for c in train_df.columns if c != target_col]
    
    X_train = train_df[feature_cols].values
    y_train = train_df[target_col].values
    
    X_val = val_df[feature_cols].values
    y_val = val_df[target_col].values
    
    X_test = test_df[feature_cols].values
    y_test = test_df[target_col].values
    
    print(f"📊 Features: {len(feature_cols)} columns")
    print(f"   Target: {target_col}")
    
    return X_train, y_train, X_val, y_val, X_test, y_test, feature_cols


def train_conformal_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    random_state: int = 42,
    n_estimators: int = 200,
    max_depth: int = 20
) -> Tuple[RandomForestQRF, SplitConformalPredictor]:
    """
    Train Random Forest + QRF model with Split Conformal calibration.
    
    Args:
        X_train: Training features
        y_train: Training target
        X_val: Calibration features
        y_val: Calibration target
        random_state: Random seed
        n_estimators: Number of trees
        max_depth: Maximum tree depth
    
    Returns:
        (base_model, conformal_predictor)
    """
    print("\n🌲 Training Random Forest + QRF model...")
    print(f"   n_estimators: {n_estimators}")
    print(f"   max_depth: {max_depth}")
    print(f"   random_state: {random_state}")
    
    # Train base model
    base_model = RandomForestQRF(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=random_state,
        n_jobs=-1
    )
    base_model.fit(X_train, y_train)
    
    # Evaluate base model on validation set
    y_val_pred = base_model.predict(X_val)
    val_rmse = np.sqrt(mean_squared_error(y_val, y_val_pred))
    val_mae = mean_absolute_error(y_val, y_val_pred)
    val_r2 = r2_score(y_val, y_val_pred)
    
    print(f"✅ Base model trained!")
    print(f"   Validation RMSE: {val_rmse:.2f} K")
    print(f"   Validation MAE:  {val_mae:.2f} K")
    print(f"   Validation R²:   {val_r2:.4f}")
    
    # Apply Split Conformal Prediction for calibration
    print("\n🔧 Applying Split Conformal Prediction calibration...")
    conformal = SplitConformalPredictor(
        base_model=base_model,
        score_function='absolute',  # Use absolute residuals as nonconformity scores
        random_state=random_state
    )
    
    # Fit on training+validation (calibrate on validation)
    conformal.fit(X_train, y_train, X_val, y_val)
    
    print("✅ Conformal calibration complete!")
    
    return base_model, conformal


def evaluate_conformal_model(
    base_model: RandomForestQRF,
    conformal: SplitConformalPredictor,
    X_test: np.ndarray,
    y_test: np.ndarray,
    alpha: float = 0.05
) -> dict:
    """Evaluate conformal model on test set."""
    print("\n🧪 Evaluating conformal model on test set...")
    
    # Calibrated predictions and intervals (from conformal)
    y_pred, y_lower, y_upper = conformal.predict_with_interval(X_test, alpha=alpha)
    
    # Metrics
    test_rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    test_mae = mean_absolute_error(y_test, y_pred)
    test_r2 = r2_score(y_test, y_pred)
    
    # Interval metrics
    interval_width = y_upper - y_lower
    mean_width = np.mean(interval_width)
    median_width = np.median(interval_width)
    coverage = np.mean((y_test >= y_lower) & (y_test <= y_upper))
    
    metrics = {
        'test_rmse': float(test_rmse),
        'test_mae': float(test_mae),
        'test_r2': float(test_r2),
        'conformal_coverage': float(coverage),
        'conformal_mean_width': float(mean_width),
        'conformal_median_width': float(median_width),
        'n_test_samples': len(y_test),
        'alpha': alpha,
        'target_coverage': 1 - alpha
    }
    
    print(f"   Test RMSE: {test_rmse:.2f} K")
    print(f"   Test MAE:  {test_mae:.2f} K")
    print(f"   Test R²:   {test_r2:.4f}")
    print(f"   Conformal coverage: {coverage:.3f} (target: {1-alpha:.3f})")
    print(f"   Mean interval width: {mean_width:.2f} K")
    
    return metrics


def save_conformal_model(
    base_model: RandomForestQRF,
    conformal: SplitConformalPredictor,
    output_dir: Path,
    metrics: dict,
    feature_cols: list
):
    """Save conformal model and metadata."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save base model
    base_model_path = output_dir / 'rf_conformal_base.pkl'
    base_model.save(base_model_path)
    print(f"\n💾 Saved base model: {base_model_path}")
    
    # Save conformal predictor (using pickle since no built-in save method)
    conformal_path = output_dir / 'rf_conformal.pkl'
    with open(conformal_path, 'wb') as f:
        pickle.dump(conformal, f)
    print(f"💾 Saved conformal model: {conformal_path}")
    
    # Save metadata
    metadata = {
        'model_type': 'RandomForestQRF_with_SplitConformal',
        'n_features': len(feature_cols),
        'feature_names': feature_cols,
        'metrics': metrics,
        'base_model_hyperparameters': {
            'n_estimators': base_model.n_estimators,
            'max_depth': base_model.max_depth,
            'random_state': base_model.random_state
        },
        'conformal_config': {
            'score_function': 'residual',
            'calibration_set_size': conformal.n_calibration if hasattr(conformal, 'n_calibration') else 'unknown'
        }
    }
    
    metadata_path = output_dir / 'rf_conformal_metadata.json'
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"📄 Saved metadata: {metadata_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Train RF+QRF model with Split Conformal calibration"
    )
    parser.add_argument(
        '--data',
        type=Path,
        default=Path('data/processed/uci_splits'),
        help='Directory containing train/val/test splits'
    )
    parser.add_argument(
        '--output',
        type=Path,
        default=Path('models'),
        help='Output directory for trained model'
    )
    parser.add_argument(
        '--n-estimators',
        type=int,
        default=200,
        help='Number of trees (default: 200)'
    )
    parser.add_argument(
        '--max-depth',
        type=int,
        default=20,
        help='Maximum tree depth (default: 20)'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed (default: 42)'
    )
    
    args = parser.parse_args()
    
    # Set random seeds
    np.random.seed(args.seed)
    
    print("=" * 70)
    print("TRAIN CONFORMAL MODEL (RF+QRF + Split Conformal)")
    print("=" * 70)
    
    # Load data
    train_df, val_df, test_df = load_splits(args.data)
    
    # Prepare data
    X_train, y_train, X_val, y_val, X_test, y_test, feature_cols = prepare_data(
        train_df, val_df, test_df
    )
    
    # Train conformal model
    base_model, conformal = train_conformal_model(
        X_train, y_train, X_val, y_val,
        random_state=args.seed,
        n_estimators=args.n_estimators,
        max_depth=args.max_depth
    )
    
    # Evaluate
    metrics = evaluate_conformal_model(base_model, conformal, X_test, y_test)
    
    # Save
    save_conformal_model(base_model, conformal, args.output, metrics, feature_cols)
    
    print("\n" + "=" * 70)
    print("✅ Conformal model training complete!")
    print("=" * 70)
    print(f"\nNext steps:")
    print(f"  1. Run calibration validation:")
    print(f"     python scripts/validate_calibration_conformal.py --model {args.output}/rf_conformal.pkl")
    print(f"  2. Expected improvement:")
    print(f"     - Coverage: {metrics['conformal_coverage']:.3f} (target: [0.94, 0.96]) ✅")
    print(f"     - ECE: Should drop from 7.02 to < 0.05")


if __name__ == '__main__':
    main()

