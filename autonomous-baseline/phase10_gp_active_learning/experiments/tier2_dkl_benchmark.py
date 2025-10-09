"""
Tier 2 DKL Benchmark: Compare DKL vs GP vs Random

Compares three active learning strategies:
1. DKL-EI: Deep Kernel Learning + Expected Improvement
2. GP-EI: Basic GP + Expected Improvement (Tier 1 baseline)
3. Random: Random sampling (control)

Goal: Prove DKL fixes Tier 1 GP failure (40-50% improvement target)

Author: GOATnote Autonomous Research Lab Initiative
Contact: b@thegoatnote.com
License: MIT
"""

import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
import logging
import sys
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from scipy import stats
import json
from datetime import datetime

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from phase10_gp_active_learning.data.uci_loader import load_uci_superconductor
from phase10_gp_active_learning.models.gp_model import GPModel
from phase10_gp_active_learning.models.dkl_model import create_dkl_model
from phase10_gp_active_learning.acquisition.expected_improvement import expected_improvement_acquisition

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Set seeds for reproducibility
def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def run_active_learning(
    X_pool: pd.DataFrame,
    y_pool: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    strategy: str = "dkl_ei",
    initial_samples: int = 100,
    num_rounds: int = 20,
    batch_size: int = 20,
    random_seed: int = 42
) -> dict:
    """
    Run single active learning simulation.
    
    Args:
        X_pool: Unlabeled pool
        y_pool: Pool labels (hidden, queried incrementally)
        X_test: Test features
        y_test: Test labels
        strategy: "dkl_ei", "gp_ei", or "random"
        initial_samples: Initial labeled samples
        num_rounds: Number of AL rounds
        batch_size: Samples per round
        random_seed: Random seed
    
    Returns:
        Dict with RMSE history and metadata
    """
    set_seed(random_seed)
    logger.info(f"🚀 Running {strategy} strategy (seed={random_seed})")
    
    # Normalize features (CRITICAL for GP/DKL)
    scaler = StandardScaler()
    
    # Initialize labeled and unlabeled sets
    initial_indices = np.random.choice(X_pool.index, initial_samples, replace=False)
    X_labeled = X_pool.loc[initial_indices]
    y_labeled = y_pool.loc[initial_indices]
    X_unlabeled = X_pool.drop(initial_indices)
    y_unlabeled = y_pool.drop(initial_indices)
    
    # Normalize
    X_labeled_scaled = scaler.fit_transform(X_labeled.values)
    X_unlabeled_scaled = scaler.transform(X_unlabeled.values)
    X_test_scaled = scaler.transform(X_test.values)
    
    rmse_history = []
    
    for round_idx in range(num_rounds):
        # Train model
        if strategy == "dkl_ei":
            # DKL: NN feature extraction + GP
            model = create_dkl_model(
                X_labeled_scaled, y_labeled.values,
                n_epochs=50,  # Faster for benchmarking
                lr=0.001,
                verbose=False
            )
            y_pred_test, _ = model.predict(X_test_scaled)
            
        elif strategy == "gp_ei":
            # Basic GP (Tier 1 baseline)
            model = GPModel(
                torch.tensor(X_labeled_scaled, dtype=torch.float64),
                torch.tensor(y_labeled.values, dtype=torch.float64)
            )
            model.fit()
            y_pred_test, _ = model.predict(X_test_scaled)
            
        else:  # random
            # Dummy prediction for random (not used for selection)
            y_pred_test = np.full(len(X_test), y_labeled.mean())
        
        # Evaluate
        rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
        rmse_history.append(rmse)
        
        if round_idx % 5 == 0 or round_idx == num_rounds - 1:
            logger.info(f"   Round {round_idx:2d}: n_labeled={len(X_labeled):4d}, RMSE={rmse:.2f} K")
        
        # Select new samples (except on last round)
        if round_idx < num_rounds - 1:
            num_to_select = min(batch_size, len(X_unlabeled))
            if num_to_select == 0:
                logger.info("   No more unlabeled samples.")
                break
            
            if strategy in ["dkl_ei", "gp_ei"]:
                # Expected Improvement acquisition
                best_f = y_labeled.max()  # Maximize Tc
                X_unlabeled_tensor = torch.tensor(X_unlabeled_scaled, dtype=torch.float64)
                ei_values = expected_improvement_acquisition(model, X_unlabeled_tensor, best_f)
                acquisition_indices = ei_values.squeeze().argsort(descending=True)[:num_to_select].cpu().numpy()
                selected_indices = X_unlabeled.iloc[acquisition_indices].index
                
            else:  # random
                selected_indices = np.random.choice(X_unlabeled.index, num_to_select, replace=False)
            
            # Move samples from unlabeled to labeled
            X_labeled = pd.concat([X_labeled, X_unlabeled.loc[selected_indices]])
            y_labeled = pd.concat([y_labeled, y_unlabeled.loc[selected_indices]])
            X_unlabeled = X_unlabeled.drop(selected_indices)
            y_unlabeled = y_unlabeled.drop(selected_indices)
            
            # Re-normalize with updated training set
            X_labeled_scaled = scaler.fit_transform(X_labeled.values)
            X_unlabeled_scaled = scaler.transform(X_unlabeled.values)
    
    logger.info(f"✅ {strategy}: Final RMSE = {rmse_history[-1]:.2f} K")
    
    return {
        'strategy': strategy,
        'seed': random_seed,
        'rmse_history': rmse_history,
        'final_rmse': rmse_history[-1]
    }


def main():
    output_dir = Path("evidence/phase10/tier2_results")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info("=" * 70)
    logger.info("PHASE 10 TIER 2: DKL vs GP vs RANDOM BENCHMARK")
    logger.info("=" * 70)
    
    # Load UCI data
    logger.info("\n📂 Loading UCI Superconductivity dataset...")
    train_df, val_df, test_df = load_uci_superconductor()
    
    feature_cols = [col for col in train_df.columns if col != 'Tc']
    X_train = train_df[feature_cols]
    y_train = train_df['Tc']
    X_test = test_df[feature_cols]
    y_test = test_df['Tc']
    
    # Combine train+val for pool
    X_pool = pd.concat([X_train, val_df[feature_cols]])
    y_pool = pd.concat([y_train, val_df['Tc']])
    
    logger.info(f"✅ Data loaded: Pool={len(X_pool)}, Test={len(X_test)}")
    
    # Benchmark parameters
    strategies = ["dkl_ei", "gp_ei", "random"]
    seeds = [42, 43, 44, 45, 46]  # 5 seeds for statistical robustness
    initial_samples = 100
    num_rounds = 20
    batch_size = 20
    
    logger.info(f"\n🔬 Benchmark: {len(strategies)} strategies × {len(seeds)} seeds × {num_rounds} rounds")
    logger.info(f"   Initial: {initial_samples}, Batch: {batch_size}")
    
    # Run benchmarks
    all_results = []
    
    for strategy in strategies:
        logger.info(f"\n{'=' * 70}")
        logger.info(f"STRATEGY: {strategy.upper()}")
        logger.info('=' * 70)
        
        for seed in seeds:
            result = run_active_learning(
                X_pool.copy(), y_pool.copy(),
                X_test.copy(), y_test.copy(),
                strategy=strategy,
                initial_samples=initial_samples,
                num_rounds=num_rounds,
                batch_size=batch_size,
                random_seed=seed
            )
            all_results.append(result)
    
    # Aggregate results
    logger.info(f"\n{'=' * 70}")
    logger.info("RESULTS SUMMARY")
    logger.info('=' * 70)
    
    results_by_strategy = {}
    for strategy in strategies:
        strategy_results = [r for r in all_results if r['strategy'] == strategy]
        final_rmses = [r['final_rmse'] for r in strategy_results]
        rmse_histories = [r['rmse_history'] for r in strategy_results]
        
        results_by_strategy[strategy] = {
            'mean_rmse': float(np.mean(final_rmses)),
            'std_rmse': float(np.std(final_rmses)),
            'min_rmse': float(np.min(final_rmses)),
            'max_rmse': float(np.max(final_rmses)),
            'rmse_histories': rmse_histories
        }
        
        logger.info(f"\n{strategy.upper()}:")
        logger.info(f"  Mean RMSE: {np.mean(final_rmses):.2f} ± {np.std(final_rmses):.2f} K")
        logger.info(f"  Range:     [{np.min(final_rmses):.2f}, {np.max(final_rmses):.2f}] K")
    
    # Statistical tests
    dkl_rmses = [r['final_rmse'] for r in all_results if r['strategy'] == 'dkl_ei']
    gp_rmses = [r['final_rmse'] for r in all_results if r['strategy'] == 'gp_ei']
    random_rmses = [r['final_rmse'] for r in all_results if r['strategy'] == 'random']
    
    # DKL vs Random
    t_stat_dkl_vs_random, p_val_dkl_vs_random = stats.ttest_ind(dkl_rmses, random_rmses)
    improvement_dkl_vs_random = ((np.mean(random_rmses) - np.mean(dkl_rmses)) / np.mean(random_rmses)) * 100
    
    # DKL vs GP (Tier 1)
    t_stat_dkl_vs_gp, p_val_dkl_vs_gp = stats.ttest_ind(dkl_rmses, gp_rmses)
    improvement_dkl_vs_gp = ((np.mean(gp_rmses) - np.mean(dkl_rmses)) / np.mean(gp_rmses)) * 100
    
    logger.info(f"\n{'=' * 70}")
    logger.info("STATISTICAL ANALYSIS")
    logger.info('=' * 70)
    
    logger.info(f"\nDKL vs RANDOM:")
    logger.info(f"  Improvement: {improvement_dkl_vs_random:+.1f}%")
    logger.info(f"  p-value: {p_val_dkl_vs_random:.4f}")
    logger.info(f"  Significant (p<0.01): {'✅ YES' if p_val_dkl_vs_random < 0.01 else '❌ NO'}")
    
    logger.info(f"\nDKL vs GP (Tier 1):")
    logger.info(f"  Improvement: {improvement_dkl_vs_gp:+.1f}%")
    logger.info(f"  p-value: {p_val_dkl_vs_gp:.4f}")
    logger.info(f"  Significant (p<0.01): {'✅ YES' if p_val_dkl_vs_gp < 0.01 else '❌ NO'}")
    
    # Success criteria
    logger.info(f"\n{'=' * 70}")
    logger.info("SUCCESS CRITERIA")
    logger.info('=' * 70)
    logger.info(f"\n  Target: ≥40% improvement vs random")
    logger.info(f"  Achieved: {improvement_dkl_vs_random:+.1f}%")
    if improvement_dkl_vs_random >= 40:
        logger.info("  ✅ SUCCESS: Above target!")
    elif improvement_dkl_vs_random >= 20:
        logger.info("  ⚠️  PARTIAL: 20-40% improvement")
    else:
        logger.info("  ❌ FAILURE: Below 20% improvement")
    
    # Visualization
    logger.info(f"\n📊 Generating plots...")
    
    fig, ax = plt.subplots(figsize=(12, 7))
    
    x_axis = np.arange(initial_samples, initial_samples + num_rounds * batch_size, batch_size)
    
    for strategy, color, label in [("dkl_ei", "blue", "DKL-EI (Tier 2)"),
                                     ("gp_ei", "orange", "GP-EI (Tier 1)"),
                                     ("random", "green", "Random")]:
        histories = np.array(results_by_strategy[strategy]['rmse_histories'])
        mean_history = np.mean(histories, axis=0)
        std_history = np.std(histories, axis=0)
        
        ax.plot(x_axis, mean_history, label=label, color=color, marker='o', linewidth=2)
        ax.fill_between(x_axis, mean_history - std_history, mean_history + std_history,
                        color=color, alpha=0.2)
    
    ax.set_xlabel('Number of Labeled Samples', fontsize=12)
    ax.set_ylabel('RMSE (K)', fontsize=12)
    ax.set_title('Phase 10 Tier 2: DKL vs GP vs Random (5 seeds)', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'tier2_dkl_vs_baselines.png', dpi=300, bbox_inches='tight')
    logger.info(f"   Saved: {output_dir / 'tier2_dkl_vs_baselines.png'}")
    
    # Save results JSON
    results_json = {
        'experiment': 'Phase 10 Tier 2: DKL Benchmark',
        'dataset': 'UCI Superconductivity',
        'n_seeds': len(seeds),
        'n_rounds': num_rounds,
        'batch_size': batch_size,
        'n_initial': initial_samples,
        'timestamp': datetime.now().isoformat(),
        'results': results_by_strategy,
        'comparisons': {
            'dkl_vs_random': {
                'improvement_percent': float(improvement_dkl_vs_random),
                'p_value': float(p_val_dkl_vs_random),
                'significant': bool(p_val_dkl_vs_random < 0.01)
            },
            'dkl_vs_gp': {
                'improvement_percent': float(improvement_dkl_vs_gp),
                'p_value': float(p_val_dkl_vs_gp),
                'significant': bool(p_val_dkl_vs_gp < 0.01)
            }
        },
        'success': {
            'target': 40.0,
            'achieved': float(improvement_dkl_vs_random),
            'met': bool(improvement_dkl_vs_random >= 40)
        }
    }
    
    with open(output_dir / 'tier2_results.json', 'w') as f:
        json.dump(results_json, f, indent=2)
    logger.info(f"   Saved: {output_dir / 'tier2_results.json'}")
    
    logger.info(f"\n{'=' * 70}")
    logger.info("✅ TIER 2 BENCHMARK COMPLETE")
    logger.info('=' * 70)
    logger.info(f"\nResults saved to: {output_dir}")
    logger.info("  - tier2_results.json")
    logger.info("  - tier2_dkl_vs_baselines.png")


if __name__ == "__main__":
    main()

