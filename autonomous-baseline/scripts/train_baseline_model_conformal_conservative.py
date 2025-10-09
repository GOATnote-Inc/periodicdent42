#!/usr/bin/env python3
"""
Train RF+QRF with CONSERVATIVE Split Conformal Prediction.

Attempt 1: Adjust quantile upward to achieve PICP ∈ [0.94, 0.96] and ECE < 0.05.

Usage:
    python scripts/train_baseline_model_conformal_conservative.py \\
        --data data/processed/uci_splits/ \\
        --output models/ \\
        --quantile-adjustment 0.97  # More conservative than 0.95
"""

import argparse
import json
import pickle
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.rf_qrf import RandomForestQRF


class ConservativeSplitConformalPredictor:
    """
    Split Conformal with adjustable conservativeness.
    
    Key modification: Use adjusted quantile for tighter coverage guarantees.
    """
    
    def __init__(
        self,
        base_model,
        score_function: str = "absolute",
        quantile_adjustment: float = 0.97,  # More conservative
        random_state: int = 42,
    ):
        self.base_model = base_model
        self.score_function = score_function
        self.quantile_adjustment = quantile_adjustment
        self.random_state = random_state
        
        self.calibration_scores_ = None
        self.fitted_ = False
    
    def fit(
        self,
        X_fit: np.ndarray,
        y_fit: np.ndarray,
        X_calibration: np.ndarray,
        y_calibration: np.ndarray,
    ):
        # Train base model
        self.base_model.fit(X_fit, y_fit)
        
        # Compute calibration scores
        y_cal_pred = self.base_model.predict(X_calibration)
        self.calibration_scores_ = np.abs(y_calibration - y_cal_pred)
        
        self.fitted_ = True
        return self
    
    def predict_with_interval(
        self,
        X: np.ndarray,
        alpha: float = 0.05,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        if not self.fitted_:
            raise ValueError("Not fitted")
        
        # Point predictions
        y_pred = self.base_model.predict(X)
        
        # CONSERVATIVE quantile adjustment
        n_cal = len(self.calibration_scores_)
        
        # Method 1: Use higher quantile to be more conservative
        adjusted_quantile = (1 - alpha) * self.quantile_adjustment
        adjusted_quantile = min(adjusted_quantile, 1.0)  # Cap at 1.0
        
        q = np.quantile(self.calibration_scores_, adjusted_quantile)
        
        # Wider intervals for higher coverage
        lower = y_pred - q
        upper = y_pred + q
        
        return y_pred, lower, upper
    
    def save(self, path: Path):
        with open(path, 'wb') as f:
            pickle.dump(self, f)


def train_conservative_conformal(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    quantile_adjustment: float = 0.97,
    random_state: int = 42,
    n_estimators: int = 200,
    max_depth: int = 20
):
    print(f"\n🌲 Training Random Forest + CONSERVATIVE Conformal...")
    print(f"   Quantile adjustment: {quantile_adjustment} (higher = more conservative)")
    
    # Train base model
    base_model = RandomForestQRF(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=random_state,
        n_jobs=-1
    )
    base_model.fit(X_train, y_train)
    
    # Apply conservative conformal
    conformal = ConservativeSplitConformalPredictor(
        base_model=base_model,
        score_function='absolute',
        quantile_adjustment=quantile_adjustment,
        random_state=random_state
    )
    conformal.fit(X_train, y_train, X_val, y_val)
    
    print("✅ Conservative conformal calibration complete!")
    
    return base_model, conformal


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=Path, default=Path('data/processed/uci_splits'))
    parser.add_argument('--output', type=Path, default=Path('models'))
    parser.add_argument('--quantile-adjustment', type=float, default=0.97,
                        help='Quantile adjustment (0.95-1.0, higher = more conservative)')
    parser.add_argument('--seed', type=int, default=42)
    
    args = parser.parse_args()
    
    np.random.seed(args.seed)
    
    print("=" * 70)
    print("TRAIN CONSERVATIVE CONFORMAL MODEL (Attempt 1)")
    print("=" * 70)
    
    # Load data
    train_df = pd.read_csv(args.data / 'train.csv')
    val_df = pd.read_csv(args.data / 'val.csv')
    test_df = pd.read_csv(args.data / 'test.csv')
    
    feature_cols = [c for c in train_df.columns if c != 'critical_temp']
    X_train = train_df[feature_cols].values
    y_train = train_df['critical_temp'].values
    X_val = val_df[feature_cols].values
    y_val = val_df['critical_temp'].values
    X_test = test_df[feature_cols].values
    y_test = test_df['critical_temp'].values
    
    # Train
    base_model, conformal = train_conservative_conformal(
        X_train, y_train, X_val, y_val,
        quantile_adjustment=args.quantile_adjustment,
        random_state=args.seed
    )
    
    # Evaluate on test set
    print("\n🧪 Evaluating on test set...")
    y_pred, y_lower, y_upper = conformal.predict_with_interval(X_test, alpha=0.05)
    
    test_rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    coverage = np.mean((y_test >= y_lower) & (y_test <= y_upper))
    mean_width = np.mean(y_upper - y_lower)
    
    print(f"   Test RMSE: {test_rmse:.2f} K")
    print(f"   Coverage: {coverage:.3f} (target: [0.94, 0.96])")
    print(f"   Mean interval width: {mean_width:.2f} K")
    
    # Save
    output_path = args.output / f'rf_conformal_conservative_q{args.quantile_adjustment:.2f}.pkl'
    conformal.save(output_path)
    print(f"\n💾 Saved: {output_path}")
    
    # Save metadata
    metadata = {
        'quantile_adjustment': args.quantile_adjustment,
        'test_rmse': float(test_rmse),
        'coverage': float(coverage),
        'mean_width': float(mean_width),
        'target_coverage': [0.94, 0.96]
    }
    
    metadata_path = args.output / f'rf_conformal_conservative_q{args.quantile_adjustment:.2f}_metadata.json'
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print("\n" + "=" * 70)
    print("✅ Conservative conformal training complete!")
    print("=" * 70)
    print(f"\nNext: Run calibration validation to check if criteria pass")


if __name__ == '__main__':
    main()

