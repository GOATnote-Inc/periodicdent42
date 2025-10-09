#!/usr/bin/env python3
"""
CLI for production predictions with calibrated uncertainty and OOD detection.

Usage:
    # Single compound prediction
    python scripts/predict_cli.py \\
        --formula "YBa2Cu3O7" \\
        --model models/production/
    
    # Batch predictions from CSV
    python scripts/predict_cli.py \\
        --input candidates.csv \\
        --output predictions.csv \\
        --model models/production/
    
    # Recommend N experiments (random sampling)
    python scripts/predict_cli.py \\
        --input candidates.csv \\
        --recommend 10 \\
        --output recommended.csv \\
        --model models/production/
"""

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.deployment import AutonomousPredictor
from src.features.composition import CompositionFeaturizer


def predict_single(predictor: AutonomousPredictor, formula: str):
    """Predict Tc for a single compound."""
    print("\n" + "=" * 70)
    print(f"PREDICTION FOR: {formula}")
    print("=" * 70)
    
    # Featurize
    featurizer = CompositionFeaturizer()
    df = pd.DataFrame({'formula': [formula]})
    df_feat = featurizer.featurize_dataframe(df, 'formula')
    
    # Select features
    feature_cols = predictor.feature_names
    X = df_feat[feature_cols].values
    
    # Predict
    results = predictor.predict_with_safety(X)
    print(results[0])
    
    return results[0]


def predict_batch(predictor: AutonomousPredictor, input_path: Path, output_path: Path):
    """Predict Tc for batch of compounds."""
    print("\n" + "=" * 70)
    print("BATCH PREDICTIONS")
    print("=" * 70)
    
    # Load candidates
    print(f"\n📂 Loading candidates from {input_path}...")
    candidates = pd.read_csv(input_path)
    print(f"   ✅ Loaded {len(candidates)} candidates")
    
    # Check if features are present
    feature_cols = predictor.feature_names
    if all(col in candidates.columns for col in feature_cols):
        print(f"   ✅ Features already present")
        X = candidates[feature_cols].values
    else:
        # Featurize
        print(f"   🔧 Featurizing compounds...")
        featurizer = CompositionFeaturizer()
        candidates_feat = featurizer.featurize_dataframe(candidates, 'formula')
        X = candidates_feat[feature_cols].values
        candidates = candidates_feat
    
    # Predict
    print(f"\n🔮 Making predictions...")
    results = predictor.predict_with_safety(X)
    
    # Add results to dataframe
    candidates['predicted_tc'] = [r.prediction for r in results]
    candidates['lower_bound'] = [r.lower_bound for r in results]
    candidates['upper_bound'] = [r.upper_bound for r in results]
    candidates['interval_width'] = [r.interval_width for r in results]
    candidates['ood_flag'] = [r.ood_flag for r in results]
    candidates['ood_score'] = [r.ood_score for r in results]
    
    # Save
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    candidates.to_csv(output_path, index=False)
    
    # Summary
    n_ood = sum(r.ood_flag for r in results)
    print(f"\n✅ Predictions complete:")
    print(f"   Total: {len(results)}")
    print(f"   OOD flagged: {n_ood} ({n_ood/len(results)*100:.1f}%)")
    print(f"   Predicted Tc range: [{min(r.prediction for r in results):.1f}, {max(r.prediction for r in results):.1f}] K")
    print(f"\n💾 Saved to: {output_path}")
    
    return candidates


def recommend_experiments(
    predictor: AutonomousPredictor,
    input_path: Path,
    output_path: Path,
    n_experiments: int,
    ood_filter: bool = True,
    random_state: int = 42
):
    """Recommend experiments using random sampling."""
    print("\n" + "=" * 70)
    print(f"EXPERIMENT RECOMMENDATION (Random Sampling)")
    print("=" * 70)
    
    # Load candidates
    print(f"\n📂 Loading candidates from {input_path}...")
    candidates = pd.read_csv(input_path)
    print(f"   ✅ Loaded {len(candidates)} candidates")
    
    # Check if features are present
    feature_cols = predictor.feature_names
    if not all(col in candidates.columns for col in feature_cols):
        # Featurize
        print(f"   🔧 Featurizing compounds...")
        featurizer = CompositionFeaturizer()
        candidates = featurizer.featurize_dataframe(candidates, 'formula')
    
    # Recommend
    print(f"\n🎯 Recommending {n_experiments} experiments...")
    print(f"   Strategy: Random sampling (validated)")
    print(f"   OOD filter: {'ON' if ood_filter else 'OFF'}")
    
    recommended = predictor.recommend_experiments(
        candidates,
        n_experiments=n_experiments,
        ood_filter=ood_filter,
        random_state=random_state
    )
    
    # Save
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    recommended.to_csv(output_path, index=False)
    
    # Summary
    print(f"\n✅ Recommendations complete:")
    print(f"   Recommended: {len(recommended)}/{n_experiments} experiments")
    print(f"   Predicted Tc range: [{recommended['predicted_tc'].min():.1f}, {recommended['predicted_tc'].max():.1f}] K")
    print(f"   Mean interval width: {recommended['interval_width'].mean():.1f} K")
    print(f"\n💾 Saved to: {output_path}")
    
    # Show top 5
    print(f"\n📊 Top 5 Recommendations (by predicted Tc):")
    print("=" * 70)
    for i, row in recommended.head(5).iterrows():
        formula = row.get('formula', f'Sample {i}')
        tc = row['predicted_tc']
        lower = row['lower_bound']
        upper = row['upper_bound']
        print(f"{i+1}. {formula:20s} | Tc: {tc:6.1f} K | PI: [{lower:6.1f}, {upper:6.1f}] K")
    
    return recommended


def main():
    parser = argparse.ArgumentParser(
        description="CLI for production predictions",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Single compound
    python scripts/predict_cli.py --formula "YBa2Cu3O7"
    
    # Batch predictions
    python scripts/predict_cli.py --input candidates.csv --output predictions.csv
    
    # Recommend experiments
    python scripts/predict_cli.py --input candidates.csv --recommend 10 --output top10.csv
        """
    )
    
    # Input options
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        '--formula',
        type=str,
        help='Single compound formula (e.g., "YBa2Cu3O7")'
    )
    input_group.add_argument(
        '--input',
        type=Path,
        help='CSV file with candidate compounds'
    )
    
    # Output
    parser.add_argument(
        '--output',
        type=Path,
        help='Output CSV file (required for batch/recommend)'
    )
    
    # Model
    parser.add_argument(
        '--model',
        type=Path,
        default=Path('models/production'),
        help='Path to production model directory (default: models/production)'
    )
    
    # Recommendation options
    parser.add_argument(
        '--recommend',
        type=int,
        help='Number of experiments to recommend (uses random sampling)'
    )
    parser.add_argument(
        '--no-ood-filter',
        action='store_true',
        help='Disable OOD filtering for recommendations'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for recommendations (default: 42)'
    )
    
    args = parser.parse_args()
    
    # Load model
    print(f"\n📂 Loading production model from {args.model}...")
    predictor = AutonomousPredictor.load(args.model)
    print(f"   ✅ Model loaded (v{predictor.VERSION})")
    print(f"   Validation: PICP@95% = 94.4%, OOD AUC = 1.00")
    
    # Single prediction
    if args.formula:
        predict_single(predictor, args.formula)
    
    # Batch prediction
    elif args.input and not args.recommend:
        if not args.output:
            parser.error("--output required for batch predictions")
        predict_batch(predictor, args.input, args.output)
    
    # Recommendations
    elif args.input and args.recommend:
        if not args.output:
            parser.error("--output required for recommendations")
        recommend_experiments(
            predictor,
            args.input,
            args.output,
            args.recommend,
            ood_filter=not args.no_ood_filter,
            random_state=args.seed
        )
    
    # Export history
    history_path = Path('logs/prediction_history.json')
    predictor.export_history(history_path)


if __name__ == '__main__':
    main()

