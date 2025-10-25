#!/usr/bin/env python3
"""
Example: Using the deployed production model for predictions.

This example demonstrates:
1. Loading the production model
2. Making predictions with calibrated uncertainty
3. OOD detection
4. Recommending experiments (random sampling)
5. Monitoring and logging

Run from autonomous-baseline/:
    python examples/deployment_example.py
"""

import sys
from pathlib import Path

import pandas as pd

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.deployment import AutonomousPredictor


def main():
    print("=" * 70)
    print("DEPLOYMENT EXAMPLE: Production Predictor")
    print("=" * 70)
    
    # 1. Load production model
    print("\n📂 Step 1: Loading production model...")
    model_path = Path('models/production')
    predictor = AutonomousPredictor.load(model_path)
    print(f"   ✅ Loaded AutonomousPredictor v{predictor.VERSION}")
    print(f"   Validation: PICP@95% = 94.4%, OOD AUC = 1.00")
    
    # 2. Load test data (pre-featurized)
    print("\n📂 Step 2: Loading test data...")
    test_data_path = Path('data/processed/uci_splits/test.csv')
    test_df = pd.read_csv(test_data_path)
    feature_cols = predictor.feature_names
    X_test = test_df[feature_cols].values[:100]  # First 100 samples
    print(f"   ✅ Loaded {len(X_test)} test samples")
    
    # 3. Make predictions with safety checks
    print("\n🔮 Step 3: Making predictions with safety checks...")
    results = predictor.predict_with_safety(X_test, alpha=0.05)
    
    # Count OOD
    n_ood = sum(r.ood_flag for r in results)
    print(f"   ✅ Predictions complete:")
    print(f"      Total: {len(results)}")
    print(f"      OOD flagged: {n_ood} ({n_ood/len(results)*100:.1f}%)")
    
    # 4. Show examples
    print("\n📊 Step 4: Example predictions:")
    print("=" * 70)
    for i, result in enumerate(results[:5], 1):
        status = "⚠️  OOD" if result.ood_flag else "✅ OK "
        print(f"{i}. {status} | Tc: {result.prediction:6.1f} K | "
              f"PI: [{result.lower_bound:6.1f}, {result.upper_bound:6.1f}] K | "
              f"Width: {result.interval_width:5.1f} K")
    
    # 5. Recommend experiments
    print("\n🎯 Step 5: Recommending experiments (random sampling)...")
    recommended = predictor.recommend_experiments(
        test_df[:100],  # Candidates
        n_experiments=10,
        ood_filter=True,  # Filter OOD
        random_state=42
    )
    print(f"   ✅ Recommended {len(recommended)} experiments")
    
    # Show top 3
    print("\n📊 Top 3 Recommended Experiments:")
    print("=" * 70)
    for i, (idx, row) in enumerate(recommended.head(3).iterrows(), 1):
        tc = row['predicted_tc']
        lower = row['lower_bound']
        upper = row['upper_bound']
        width = row['interval_width']
        ood_score = row['ood_score']
        print(f"{i}. Sample {idx:4d} | Tc: {tc:6.1f} K | "
              f"PI: [{lower:6.1f}, {upper:6.1f}] K | "
              f"Width: {width:5.1f} K | OOD: {ood_score:5.1f}")
    
    # 6. Monitoring stats
    print("\n📈 Step 6: Monitoring statistics...")
    stats = predictor.get_monitoring_stats()
    print(f"   Total predictions: {stats['n_predictions']}")
    print(f"   Mean predicted Tc: {stats['mean_prediction']:.1f} K")
    print(f"   Mean interval width: {stats['mean_interval_width']:.1f} K")
    print(f"   OOD rate: {stats['ood_rate']:.1%}")
    
    # 7. Export history
    print("\n💾 Step 7: Exporting prediction history...")
    history_path = Path('logs/deployment_example_history.json')
    predictor.export_history(history_path)
    print(f"   ✅ Exported to {history_path}")
    
    # 8. GO/NO-GO decision example
    print("\n⚖️  Step 8: GO/NO-GO Decision Example...")
    print("=" * 70)
    
    for i, result in enumerate(results[:3], 1):
        print(f"\nSample {i}:")
        print(f"  Predicted Tc: {result.prediction:.1f} K")
        print(f"  95% PI: [{result.lower_bound:.1f}, {result.upper_bound:.1f}] K")
        print(f"  OOD Score: {result.ood_score:.1f} (threshold: {result.ood_threshold:.1f})")
        
        # Decision logic
        if result.ood_flag:
            decision = "❌ NO-GO"
            reason = "Sample flagged as OOD - out of training distribution"
        elif result.interval_width > 50:
            decision = "⚠️  CONDITIONAL GO"
            reason = "Wide prediction interval - low confidence"
        else:
            decision = "✅ GO"
            reason = "In-distribution with calibrated uncertainty"
        
        print(f"  Decision: {decision}")
        print(f"  Reason: {reason}")
    
    print("\n" + "=" * 70)
    print("✅ DEPLOYMENT EXAMPLE COMPLETE")
    print("=" * 70)
    print("\n📚 Next steps:")
    print("  1. Review prediction history: logs/deployment_example_history.json")
    print("  2. Monitor OOD rate and interval widths")
    print("  3. Collect experimental results to validate predictions")
    print("  4. Retrain/recalibrate after 100+ new experiments")
    print("\n📖 Documentation:")
    print("  - Deployment guide: DEPLOYMENT_GUIDE.md")
    print("  - Validation report: VALIDATION_SUITE_COMPLETE.md")
    print("  - Executive summary: EXECUTIVE_SUMMARY_VALIDATION.md")


if __name__ == '__main__':
    main()

