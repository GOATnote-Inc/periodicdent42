#!/usr/bin/env python3
"""
Train baseline Random Forest + QRF model on UCI Superconductivity Dataset.

This script trains the primary uncertainty quantification model for validation experiments.

Usage:
    python scripts/train_baseline_model.py --data data/processed/uci_splits/ --output models/
"""

import argparse
import json
import pickle
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

# Add src to path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.rf_qrf import RandomForestQRF
from src.features.scaling import FeatureScaler


def load_splits(data_dir: Path) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load train/val/test splits."""
    train_df = pd.read_csv(data_dir / 'train.csv')
    val_df = pd.read_csv(data_dir / 'val.csv')
    test_df = pd.read_csv(data_dir / 'test.csv')
    
    print(f"📂 Loaded data from {data_dir}")
    print(f"   Train: {len(train_df)} samples")
    print(f"   Val:   {len(val_df)} samples")
    print(f"   Test:  {len(test_df)} samples")
    
    return train_df, val_df, test_df


def prepare_data(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    target_col: str = 'critical_temp'
) -> Tuple:
    """Prepare features and targets."""
    # Separate features and target
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


def train_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    random_state: int = 42,
    n_estimators: int = 200,
    max_depth: int = 20
) -> RandomForestQRF:
    """
    Train Random Forest + QRF model.
    
    Args:
        X_train: Training features
        y_train: Training target
        X_val: Validation features (for early stopping)
        y_val: Validation target
        random_state: Random seed
        n_estimators: Number of trees
        max_depth: Maximum tree depth
    
    Returns:
        Trained RandomForestQRF model
    """
    print("\n🌲 Training Random Forest + QRF model...")
    print(f"   n_estimators: {n_estimators}")
    print(f"   max_depth: {max_depth}")
    print(f"   random_state: {random_state}")
    
    model = RandomForestQRF(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=random_state,
        n_jobs=-1  # Use all CPU cores
    )
    
    model.fit(X_train, y_train)
    
    # Evaluate on validation set
    y_val_pred = model.predict(X_val)
    val_rmse = np.sqrt(mean_squared_error(y_val, y_val_pred))
    val_mae = mean_absolute_error(y_val, y_val_pred)
    val_r2 = r2_score(y_val, y_val_pred)
    
    print(f"\n✅ Training complete!")
    print(f"   Validation RMSE: {val_rmse:.2f} K")
    print(f"   Validation MAE:  {val_mae:.2f} K")
    print(f"   Validation R²:   {val_r2:.4f}")
    
    return model


def evaluate_model(
    model: RandomForestQRF,
    X_test: np.ndarray,
    y_test: np.ndarray
) -> dict:
    """Evaluate model on test set."""
    print("\n🧪 Evaluating on test set...")
    
    # Point predictions
    y_pred = model.predict(X_test)
    
    # Uncertainty estimates
    y_pred_with_unc, y_lower, y_upper = model.predict_with_uncertainty(X_test)
    y_std = model.get_epistemic_uncertainty(X_test)
    
    # Metrics
    test_rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    test_mae = mean_absolute_error(y_test, y_pred)
    test_r2 = r2_score(y_test, y_pred)
    
    # Uncertainty metrics
    mean_std = np.mean(y_std)
    median_std = np.median(y_std)
    
    metrics = {
        'test_rmse': float(test_rmse),
        'test_mae': float(test_mae),
        'test_r2': float(test_r2),
        'mean_uncertainty': float(mean_std),
        'median_uncertainty': float(median_std),
        'n_test_samples': len(y_test)
    }
    
    print(f"   Test RMSE: {test_rmse:.2f} K")
    print(f"   Test MAE:  {test_mae:.2f} K")
    print(f"   Test R²:   {test_r2:.4f}")
    print(f"   Mean uncertainty: {mean_std:.2f} K")
    
    return metrics


def save_model(
    model: RandomForestQRF,
    output_dir: Path,
    metrics: dict,
    feature_cols: list
):
    """Save model and metadata."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save model
    model_path = output_dir / 'rf_baseline.pkl'
    model.save(model_path)
    print(f"\n💾 Saved model: {model_path}")
    
    # Save metadata
    metadata = {
        'model_type': 'RandomForestQRF',
        'n_features': len(feature_cols),
        'feature_names': feature_cols,
        'metrics': metrics,
        'hyperparameters': {
            'n_estimators': model.n_estimators,
            'max_depth': model.max_depth,
            'random_state': model.random_state
        }
    }
    
    metadata_path = output_dir / 'rf_baseline_metadata.json'
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"📄 Saved metadata: {metadata_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Train baseline RF+QRF model on UCI Superconductivity Dataset"
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
    print("TRAIN BASELINE MODEL (RF+QRF)")
    print("=" * 70)
    
    # Load data
    train_df, val_df, test_df = load_splits(args.data)
    
    # Prepare data
    X_train, y_train, X_val, y_val, X_test, y_test, feature_cols = prepare_data(
        train_df, val_df, test_df
    )
    
    # Train model
    model = train_model(
        X_train, y_train, X_val, y_val,
        random_state=args.seed,
        n_estimators=args.n_estimators,
        max_depth=args.max_depth
    )
    
    # Evaluate
    metrics = evaluate_model(model, X_test, y_test)
    
    # Save
    save_model(model, args.output, metrics, feature_cols)
    
    print("\n" + "=" * 70)
    print("✅ Baseline model training complete!")
    print("=" * 70)
    print(f"\nNext steps:")
    print(f"  1. Run calibration validation:")
    print(f"     python scripts/validate_calibration.py --model {args.output}/rf_baseline.pkl")
    print(f"  2. Run active learning experiment:")
    print(f"     python scripts/run_al_experiment.py --data {args.data}")


if __name__ == '__main__':
    main()

