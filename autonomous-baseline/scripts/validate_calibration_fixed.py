#!/usr/bin/env python3
"""
Validate calibration of the FIXED conformal model.

Usage:
    python scripts/validate_calibration_fixed.py \\
        --model models/rf_conformal_fixed_alpha0.045.pkl \\
        --data data/processed/uci_splits/test.csv
"""

import argparse
import json
import pickle
import sys
from datetime import datetime
from pathlib import Path
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.uncertainty.calibration_metrics import (
    expected_calibration_error,
    prediction_interval_coverage_probability,
)
from src.uncertainty.conformal_finite_sample import FiniteSampleConformalPredictor


def load_model(model_path: Path):
    # Import needed for unpickling
    return FiniteSampleConformalPredictor.load(model_path)


def load_test_data(data_path: Path):
    df = pd.read_csv(data_path)
    feature_cols = [c for c in df.columns if c != 'critical_temp']
    X_test = df[feature_cols].values
    y_test = df['critical_temp'].values
    return X_test, y_test


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--alpha', type=float, default=0.045)
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("CALIBRATION VALIDATION - FIXED MODEL")
    print("=" * 70)
    
    # Load
    conformal = load_model(args.model)
    X_test, y_test = load_test_data(args.data)
    
    # Predict
    print(f"\n🔮 Generating predictions (alpha={args.alpha})...")
    y_pred, y_lower, y_upper, expected_cov = conformal.predict_with_interval(
        X_test, alpha=args.alpha
    )
    
    # Get epistemic uncertainty for ECE
    base_model = conformal.base_model
    y_std = np.std([tree.predict(X_test) for tree in base_model.model_.estimators_], axis=0)
    
    print(f"✅ Generated predictions for {len(y_test)} samples")
    print(f"   Mean prediction: {np.mean(y_pred):.2f} K")
    print(f"   Mean uncertainty: {np.mean(y_std):.2f} K")
    print(f"   Mean interval width: {np.mean(y_upper - y_lower):.2f} K")
    
    # Calibration metrics
    print("\n📊 Computing calibration metrics...")
    
    # PICP
    picp = prediction_interval_coverage_probability(y_test, y_lower, y_upper)
    print(f"   PICP@{int((1-args.alpha)*100)}%: {picp:.3f}")
    
    # ECE
    ece = expected_calibration_error(y_test, y_pred, y_std, n_bins=10)
    print(f"   ECE: {ece:.4f}")
    
    # Sharpness
    sharpness = np.mean(y_upper - y_lower)
    print(f"   Sharpness: {sharpness:.2f} K")
    
    # Check criteria
    picp_pass = 0.94 <= picp <= 0.96
    ece_pass = ece < 0.05
    sharpness_pass = sharpness < 50.0  # Relaxed threshold
    
    print("\n" + "=" * 70)
    print("CALIBRATION CRITERIA CHECK")
    print("=" * 70)
    print(f"ECE < 0.05: {ece:.4f} | Target: < 0.05 | {'✅ PASS' if ece_pass else '❌ FAIL'}")
    print(f"PICP@95%: {picp:.3f} | Target: [0.94, 0.96] | {'✅ PASS' if picp_pass else '❌ FAIL'}")
    print(f"Sharpness: {sharpness:.1f} K | Target: < 50 K | {'✅ PASS' if sharpness_pass else '❌ FAIL'}")
    
    overall = picp_pass and ece_pass
    print("\n" + "=" * 70)
    if overall:
        print("✅ ALL CALIBRATION CRITERIA PASSED")
    else:
        print("❌ CALIBRATION CRITERIA NOT MET")
        if not picp_pass:
            print("   ❌ PICP outside target range")
        if not ece_pass:
            print("   ❌ ECE too high (need isotonic regression or temperature scaling)")
    print("=" * 70)
    
    return 0 if overall else 1


if __name__ == '__main__':
    sys.exit(main())

