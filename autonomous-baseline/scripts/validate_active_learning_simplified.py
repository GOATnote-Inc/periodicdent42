#!/usr/bin/env python3
"""
Simplified Active Learning validation on UCI Superconductivity dataset.

Success Criteria:
- RMSE reduction ≥ 20% vs random baseline
- Statistical significance: p < 0.01 (paired t-test, n=5 seeds)

Strategies:
- Random: Random sampling baseline
- UCB: Upper confidence bound (exploitation + exploration)

Usage:
    python scripts/validate_active_learning_simplified.py \\
        --data data/processed/uci_splits/ \\
        --output evidence/validation/active_learning/
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import mean_squared_error

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.rf_qrf import RandomForestQRF


def simulate_active_learning(
    X_train_full: np.ndarray,
    y_train_full: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    strategy: str,
    n_rounds: int = 10,
    batch_size: int = 20,
    n_init: int = 100,
    random_state: int = 42
) -> list:
    """
    Simulate active learning loop.
    
    Args:
        X_train_full: Full training pool
        y_train_full: Full training labels
        X_test: Test features
        y_test: Test labels
        strategy: 'random' or 'ucb'
        n_rounds: Number of AL rounds
        batch_size: Samples per round
        n_init: Initial labeled set size
        random_state: Random seed
    
    Returns:
        List of RMSE values (one per round)
    """
    rng = np.random.RandomState(random_state)
    
    # Initialize with random samples
    all_idx = np.arange(len(X_train_full))
    labeled_idx = rng.choice(all_idx, size=n_init, replace=False)
    unlabeled_idx = np.setdiff1d(all_idx, labeled_idx)
    
    rmse_history = []
    
    for round_i in range(n_rounds):
        # Train model on labeled data
        X_labeled = X_train_full[labeled_idx]
        y_labeled = y_train_full[labeled_idx]
        
        model = RandomForestQRF(
            n_estimators=100,  # Fewer trees for speed
            max_depth=15,
            random_state=random_state,
            n_jobs=-1
        )
        model.fit(X_labeled, y_labeled)
        
        # Evaluate on test set
        y_pred = model.predict(X_test)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        rmse_history.append(rmse)
        
        # Select next batch
        if len(unlabeled_idx) == 0:
            break
        
        X_unlabeled = X_train_full[unlabeled_idx]
        
        if strategy == 'random':
            # Random baseline
            n_select = min(batch_size, len(unlabeled_idx))
            selected_local_idx = rng.choice(len(unlabeled_idx), size=n_select, replace=False)
        
        elif strategy == 'maxvar':
            # Maximum Variance - select samples with highest uncertainty
            # This maximizes information gain and reduces overall RMSE
            y_std_unlabeled = model.get_epistemic_uncertainty(X_unlabeled)
            
            # Select top k with highest variance
            n_select = min(batch_size, len(unlabeled_idx))
            selected_local_idx = np.argsort(y_std_unlabeled)[-n_select:]
        
        else:
            raise ValueError(f"Unknown strategy: {strategy}")
        
        # Add to labeled set
        selected_global_idx = unlabeled_idx[selected_local_idx]
        labeled_idx = np.concatenate([labeled_idx, selected_global_idx])
        unlabeled_idx = np.setdiff1d(unlabeled_idx, selected_global_idx)
    
    return rmse_history


def run_al_experiment(
    data_dir: Path,
    output_dir: Path,
    n_seeds: int = 5,
    n_rounds: int = 10,
    batch_size: int = 20
):
    """Run AL experiment with multiple seeds."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 70)
    print("ACTIVE LEARNING VALIDATION (Simplified)")
    print("=" * 70)
    
    # Load data
    print("\n📂 Loading data...")
    train_df = pd.read_csv(data_dir / 'train.csv')
    test_df = pd.read_csv(data_dir / 'test.csv')
    
    feature_cols = [c for c in train_df.columns if c != 'critical_temp']
    X_train = train_df[feature_cols].values
    y_train = train_df['critical_temp'].values
    X_test = test_df[feature_cols].values
    y_test = test_df['critical_temp'].values
    
    print(f"   Train pool: {len(X_train)} samples")
    print(f"   Test: {len(X_test)} samples")
    
    # Run experiments
    strategies = ['random', 'maxvar']
    results = {s: [] for s in strategies}
    
    print(f"\n🔄 Running AL simulation ({n_seeds} seeds, {n_rounds} rounds)...")
    for seed in range(42, 42 + n_seeds):
        print(f"\n   Seed {seed}:")
        for strategy in strategies:
            rmse_history = simulate_active_learning(
                X_train, y_train, X_test, y_test,
                strategy=strategy,
                n_rounds=n_rounds,
                batch_size=batch_size,
                random_state=seed
            )
            results[strategy].append(rmse_history)
            final_rmse = rmse_history[-1]
            print(f"      {strategy:8s}: final RMSE = {final_rmse:.2f} K")
    
    # Aggregate results
    print("\n📊 Computing statistics...")
    random_final = [results['random'][i][-1] for i in range(n_seeds)]
    maxvar_final = [results['maxvar'][i][-1] for i in range(n_seeds)]
    
    random_mean = np.mean(random_final)
    maxvar_mean = np.mean(maxvar_final)
    improvement_pct = (random_mean - maxvar_mean) / random_mean * 100
    
    # Statistical test
    t_stat, p_value = stats.ttest_rel(random_final, maxvar_final)
    
    print(f"\n   Random baseline: {random_mean:.2f} ± {np.std(random_final):.2f} K")
    print(f"   MaxVar strategy: {maxvar_mean:.2f} ± {np.std(maxvar_final):.2f} K")
    print(f"   Improvement:     {improvement_pct:.1f}%")
    print(f"   p-value:         {p_value:.4f}")
    
    # Check criteria
    improvement_pass = improvement_pct >= 20.0
    significance_pass = p_value < 0.01
    
    # Plot learning curves
    print("\n📈 Generating learning curves...")
    fig, ax = plt.subplots(figsize=(10, 6))
    
    for strategy, color in zip(strategies, ['#E63946', '#2A9D8F']):
        mean_rmse = np.mean(results[strategy], axis=0)
        std_rmse = np.std(results[strategy], axis=0)
        queries = 100 + np.arange(len(mean_rmse)) * batch_size
        
        label = f"{strategy.upper()}" + (f" (final: {np.mean([r[-1] for r in results[strategy]]):.1f} K)" if strategy == 'maxvar' else f" (baseline: {np.mean([r[-1] for r in results[strategy]]):.1f} K)")
        ax.plot(queries, mean_rmse, label=label, color=color, linewidth=2.5)
        ax.fill_between(queries, mean_rmse - std_rmse, mean_rmse + std_rmse,
                        alpha=0.2, color=color)
    
    ax.set_xlabel('Number of Labeled Samples', fontsize=12)
    ax.set_ylabel('Test RMSE (K)', fontsize=12)
    ax.set_title(f'Active Learning: {improvement_pct:.1f}% improvement (p={p_value:.4f})',
                fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / 'al_learning_curve.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"   Saved: {output_dir / 'al_learning_curve.png'}")
    
    # Save metrics
    metrics = {
        'random_final_rmse_mean': float(random_mean),
        'random_final_rmse_std': float(np.std(random_final)),
        'maxvar_final_rmse_mean': float(maxvar_mean),
        'maxvar_final_rmse_std': float(np.std(maxvar_final)),
        'improvement_percent': float(improvement_pct),
        'p_value': float(p_value),
        't_statistic': float(t_stat),
        'n_seeds': n_seeds,
        'n_rounds': n_rounds,
        'batch_size': batch_size,
        'improvement_target': 20.0,
        'significance_target': 0.01,
        'improvement_pass': bool(improvement_pass),
        'significance_pass': bool(significance_pass),
        'overall_success': bool(improvement_pass and significance_pass),
        'timestamp': datetime.now().isoformat()
    }
    
    metrics_path = output_dir / 'al_metrics.json'
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"   Saved: {metrics_path}")
    
    # Interpretation
    interpretation = f"""
ACTIVE LEARNING VALIDATION RESULTS
===================================

Dataset: UCI Superconductivity
Strategy: Maximum Variance (MaxVar) vs Random Baseline
Seeds: {n_seeds}
Rounds: {n_rounds}
Batch size: {batch_size}

SUMMARY
-------
Random Baseline:  {random_mean:.2f} ± {np.std(random_final):.2f} K
MaxVar Strategy:  {maxvar_mean:.2f} ± {np.std(maxvar_final):.2f} K
Improvement:      {improvement_pct:.1f}%
Statistical Test: t={t_stat:.2f}, p={p_value:.4f}

SUCCESS CRITERIA
----------------
Improvement ≥ 20%: {improvement_pct:.1f}% | Target: ≥ 20% | {'✅ PASS' if improvement_pass else '❌ FAIL'}
Significance p<0.01: {p_value:.4f} | Target: < 0.01 | {'✅ PASS' if significance_pass else '❌ FAIL'}

OVERALL: {'✅ ALL CRITERIA MET' if (improvement_pass and significance_pass) else '❌ SOME CRITERIA NOT MET'}

INTERPRETATION
--------------
"""
    
    if improvement_pass and significance_pass:
        interpretation += f"""
✅ ACTIVE LEARNING: SUCCESSFUL

MaxVar-based active learning achieves statistically significant RMSE reduction 
compared to random sampling. With {n_rounds * batch_size} queries, MaxVar reduces 
error by {improvement_pct:.1f}%, demonstrating that uncertainty-driven sampling 
effectively reduces model error.

This validates the framework for autonomous experiment prioritization in materials discovery.
"""
    elif improvement_pass and not significance_pass:
        interpretation += f"""
⚠️  ACTIVE LEARNING: MARGINAL

While MaxVar shows {improvement_pct:.1f}% improvement, the result is not statistically 
significant (p={p_value:.4f} ≥ 0.01). This may indicate:
1. High variance in individual runs
2. Need for more seeds (currently n={n_seeds})
3. MaxVar benefit is modest but consistent

Recommendation: Run with n=10 seeds to increase statistical power.
"""
    else:
        interpretation += f"""
❌ ACTIVE LEARNING: NEEDS IMPROVEMENT

MaxVar did not achieve the target improvement. Possible causes:
1. Batch size too large relative to data dimensionality
2. RF uncertainty estimates not informative enough for AL
3. Need diversity-aware selection (k-Medoids or DPP)

Recommendation: Try smaller batch sizes or add diversity constraints.
"""
    
    interpretation_path = output_dir / 'al_interpretation.txt'
    with open(interpretation_path, 'w') as f:
        f.write(interpretation)
    print(f"   Saved: {interpretation_path}")
    
    # Print summary
    print("\n" + "=" * 70)
    if improvement_pass and significance_pass:
        print("✅ ACTIVE LEARNING VALIDATION: PASSED")
    else:
        print("❌ ACTIVE LEARNING VALIDATION: FAILED")
    print("=" * 70)
    print(f"\nResults saved to: {output_dir}")
    
    return 0 if (improvement_pass and significance_pass) else 1


def main():
    parser = argparse.ArgumentParser(
        description="Validate active learning on UCI Superconductivity dataset"
    )
    parser.add_argument(
        '--data',
        type=Path,
        default=Path('data/processed/uci_splits'),
        help='Directory containing train/test splits'
    )
    parser.add_argument(
        '--output',
        type=Path,
        default=Path('evidence/validation/active_learning'),
        help='Output directory for validation artifacts'
    )
    parser.add_argument(
        '--n-seeds',
        type=int,
        default=5,
        help='Number of random seeds (default: 5)'
    )
    parser.add_argument(
        '--n-rounds',
        type=int,
        default=10,
        help='Number of AL rounds (default: 10)'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=20,
        help='Samples per round (default: 20)'
    )
    
    args = parser.parse_args()
    
    return run_al_experiment(
        args.data,
        args.output,
        n_seeds=args.n_seeds,
        n_rounds=args.n_rounds,
        batch_size=args.batch_size
    )


if __name__ == '__main__':
    sys.exit(main())

