#!/usr/bin/env python3
"""
Prepare production model from validated components.

Takes the validated conformal predictor and OOD detector and packages
them into a production-ready AutonomousPredictor.

Usage:
    python scripts/prepare_production_model.py \\
        --conformal-model models/rf_conformal_fixed_alpha0.045.pkl \\
        --training-data data/processed/uci_splits/train.csv \\
        --output models/production/
"""

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.deployment import AutonomousPredictor, OODDetector
from src.uncertainty.conformal_finite_sample import FiniteSampleConformalPredictor


def prepare_production_model(
    conformal_model_path: Path,
    training_data_path: Path,
    output_dir: Path,
    ood_threshold: float = 150.0
):
    """
    Prepare production model.
    
    Args:
        conformal_model_path: Path to validated conformal model
        training_data_path: Path to training data (for OOD fitting)
        output_dir: Output directory for production model
        ood_threshold: OOD detection threshold (default: 150.0)
    """
    print("=" * 70)
    print("PREPARING PRODUCTION MODEL")
    print("=" * 70)
    
    # Load conformal predictor
    print(f"\n📂 Loading validated conformal predictor...")
    print(f"   Path: {conformal_model_path}")
    conformal_predictor = FiniteSampleConformalPredictor.load(conformal_model_path)
    print(f"   ✅ Loaded conformal predictor")
    
    # Load training data
    print(f"\n📂 Loading training data...")
    print(f"   Path: {training_data_path}")
    train_df = pd.read_csv(training_data_path)
    feature_cols = [c for c in train_df.columns if c != 'critical_temp']
    X_train = train_df[feature_cols].values
    print(f"   ✅ Loaded {len(X_train)} samples, {len(feature_cols)} features")
    
    # Fit OOD detector
    print(f"\n🔧 Fitting OOD detector (threshold={ood_threshold})...")
    ood_detector = OODDetector(threshold=ood_threshold)
    ood_detector.fit(X_train)
    print(f"   ✅ OOD detector fitted")
    
    # Create autonomous predictor
    print(f"\n🚀 Creating AutonomousPredictor...")
    metadata = {
        'validation_dataset': 'UCI Superconductivity (N=21,263)',
        'validation_date': '2025-10-09',
        'picp_95': 0.944,
        'ood_auc': 1.00,
        'ood_tpr_at_10fpr': 1.00,
        'physics_validation': '100% features unbiased',
        'active_learning_status': 'NOT VALIDATED - use random sampling',
        'deployment_recommendation': 'Deploy uncertainty + OOD; random sampling only'
    }
    
    predictor = AutonomousPredictor(
        conformal_predictor=conformal_predictor,
        ood_detector=ood_detector,
        feature_names=feature_cols,
        metadata=metadata
    )
    
    # Save production model
    print(f"\n💾 Saving production model...")
    output_dir = Path(output_dir)
    predictor.save(output_dir)
    print(f"   ✅ Saved to {output_dir}")
    
    # Test prediction
    print(f"\n🧪 Testing prediction on first training sample...")
    test_sample = X_train[0:1]
    results = predictor.predict_with_safety(test_sample)
    print(results[0])
    
    print("\n" + "=" * 70)
    print("✅ PRODUCTION MODEL READY")
    print("=" * 70)
    print(f"\nModel location: {output_dir}")
    print(f"Files created:")
    print(f"  - conformal_predictor.pkl")
    print(f"  - ood_detector.pkl")
    print(f"  - predictor_metadata.json")
    print(f"\nUsage:")
    print(f"  from src.deployment import AutonomousPredictor")
    print(f"  predictor = AutonomousPredictor.load('{output_dir}')")
    print(f"  results = predictor.predict_with_safety(X)")


def main():
    parser = argparse.ArgumentParser(
        description="Prepare production model from validated components"
    )
    parser.add_argument(
        '--conformal-model',
        type=Path,
        default=Path('models/rf_conformal_fixed_alpha0.045.pkl'),
        help='Path to validated conformal model'
    )
    parser.add_argument(
        '--training-data',
        type=Path,
        default=Path('data/processed/uci_splits/train.csv'),
        help='Path to training data for OOD fitting'
    )
    parser.add_argument(
        '--output',
        type=Path,
        default=Path('models/production'),
        help='Output directory for production model'
    )
    parser.add_argument(
        '--ood-threshold',
        type=float,
        default=150.0,
        help='OOD detection threshold (default: 150.0)'
    )
    
    args = parser.parse_args()
    
    prepare_production_model(
        args.conformal_model,
        args.training_data,
        args.output,
        args.ood_threshold
    )


if __name__ == '__main__':
    main()

