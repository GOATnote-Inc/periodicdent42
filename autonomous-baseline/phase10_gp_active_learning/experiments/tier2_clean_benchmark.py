"""
Tier 2 Clean Benchmark: DKL vs GP vs Random with proper BoTorch API

Uses official BoTorch acquisition functions with properly wrapped models.

Author: GOATnote Autonomous Research Lab Initiative
Contact: b@thegoatnote.com
License: MIT
"""

import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt
import logging
import sys
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from scipy import stats
import json
from datetime import datetime

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from phase10_gp_active_learning.data.uci_loader import load_uci_superconductor
from phase10_gp_active_learning.models.dkl_model import create_dkl_model
from phase10_gp_active_learning.models.botorch_dkl import BoTorchDKL

# BoTorch imports
from botorch.models import SingleTaskGP
from botorch.fit import fit_gpytorch_mll
from gpytorch.mlls import ExactMarginalLogLikelihood
from botorch.acquisition import ExpectedImprovement
from botorch.optim import optimize_acqf

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def set_seed(seed):
    """Set all random seeds for reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def run_active_learning_clean(
    X_pool: np.ndarray,
    y_pool: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    strategy: str = "dkl",
    initial_samples: int = 100,
    num_rounds: int = 20,
    batch_size: int = 20,
    random_seed: int = 42
) -> dict:
    """
    Run AL with clean BoTorch API.
    
    Args:
        X_pool, y_pool: Pool data (numpy arrays)
        X_test, y_test: Test data (numpy arrays)
        strategy: "dkl", "gp", or "random"
        initial_samples: Initial labeled samples
        num_rounds: AL rounds
        batch_size: Samples per round
        random_seed: Seed
    
    Returns:
        Dict with RMSE history
    """
    set_seed(random_seed)
    logger.info(f"🚀 Running {strategy.upper()} (seed={random_seed})")
    
    # Normalize (critical for GP/DKL)
    scaler = StandardScaler()
    X_pool_scaled = scaler.fit_transform(X_pool)
    X_test_scaled = scaler.transform(X_test)
    
    # Initial random selection
    indices = np.random.choice(len(X_pool), initial_samples, replace=False)
    labeled_mask = np.zeros(len(X_pool), dtype=bool)
    labeled_mask[indices] = True
    
    rmse_history = []
    
    for round_idx in range(num_rounds):
        # Get current labeled data
        X_labeled = X_pool_scaled[labeled_mask]
        y_labeled = y_pool[labeled_mask]
        
        # Convert to torch tensors (float64 for numerical stability)
        X_train_t = torch.tensor(X_labeled, dtype=torch.float64)
        y_train_t = torch.tensor(y_labeled, dtype=torch.float64)
        X_test_t = torch.tensor(X_test_scaled, dtype=torch.float64)
        
        # Train model based on strategy
        if strategy == "dkl":
            # Train DKL
            dkl = create_dkl_model(
                X_labeled, y_labeled,
                input_dim=X_labeled.shape[1],
                n_epochs=50,  # Fast for benchmarking
                lr=0.001,
                verbose=False
            )
            model = BoTorchDKL(dkl)
            
        elif strategy == "gp":
            # Train baseline GP (BoTorch SingleTaskGP)
            model = SingleTaskGP(X_train_t, y_train_t.unsqueeze(-1))
            mll = ExactMarginalLogLikelihood(model.likelihood, model)
            fit_gpytorch_mll(mll)
            
        else:  # random
            model = None
        
        # Evaluate on test set
        if model is not None:
            model.eval()
            with torch.no_grad():
                posterior = model.posterior(X_test_t)
                y_pred = posterior.mean.squeeze().cpu().numpy()
        else:
            # Random: use mean prediction
            y_pred = np.full(len(y_test), y_labeled.mean())
        
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        rmse_history.append(rmse)
        
        if round_idx % 5 == 0 or round_idx == num_rounds - 1:
            logger.info(f"   Round {round_idx:2d}: n_labeled={labeled_mask.sum():4d}, RMSE={rmse:.2f} K")
        
        # Select new samples (except last round)
        if round_idx < num_rounds - 1:
            unlabeled_indices = np.where(~labeled_mask)[0]
            num_to_select = min(batch_size, len(unlabeled_indices))
            
            if num_to_select == 0:
                break
            
            if strategy in ["dkl", "gp"]:
                # Use ExpectedImprovement
                best_f = y_labeled.max()
                
                # Prepare unlabeled candidates
                X_unlabeled_t = torch.tensor(
                    X_pool_scaled[unlabeled_indices],
                    dtype=torch.float64
                )
                
                # BoTorch EI
                acq = ExpectedImprovement(model=model, best_f=best_f)
                
                # Evaluate EI (add q dimension)
                X_candidates_q = X_unlabeled_t.unsqueeze(1)  # (N, 1, D)
                with torch.no_grad():
                    ei_values = acq(X_candidates_q).cpu().numpy()
                
                # Select top EI samples
                top_indices = np.argsort(ei_values)[-num_to_select:]
                selected = unlabeled_indices[top_indices]
                
            else:  # random
                selected = np.random.choice(unlabeled_indices, num_to_select, replace=False)
            
            # Mark as labeled
            labeled_mask[selected] = True
    
    logger.info(f"✅ {strategy.upper()}: Final RMSE = {rmse_history[-1]:.2f} K")
    
    return {
        'strategy': strategy,
        'seed': random_seed,
        'rmse_history': rmse_history
    }


def main():
    output_dir = Path("evidence/phase10/tier2_clean")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info("=" * 70)
    logger.info("PHASE 10 TIER 2: CLEAN DKL vs GP vs RANDOM BENCHMARK")
    logger.info("=" * 70)
    
    # Load data
    logger.info("\n📂 Loading UCI dataset...")
    train_df, val_df, test_df = load_uci_superconductor()
    
    feature_cols = [col for col in train_df.columns if col != 'Tc']
    
    # Pool = train + val
    X_pool = pd.concat([train_df[feature_cols], val_df[feature_cols]]).values
    y_pool = pd.concat([train_df['Tc'], val_df['Tc']]).values
    X_test = test_df[feature_cols].values
    y_test = test_df['Tc'].values
    
    logger.info(f"✅ Pool: {len(X_pool)}, Test: {len(X_test)}")
    
    # Benchmark parameters
    strategies = ["dkl", "gp", "random"]
    seeds = [42, 43, 44, 45, 46]  # 5 seeds
    initial_samples = 100
    num_rounds = 20
    batch_size = 20
    
    logger.info(f"\n🔬 Running: {len(strategies)} strategies × {len(seeds)} seeds × {num_rounds} rounds")
    
    # Run benchmarks
    all_results = []
    
    for strategy in strategies:
        logger.info(f"\n{'='*70}")
        logger.info(f"STRATEGY: {strategy.upper()}")
        logger.info('='*70)
        
        for seed in seeds:
            result = run_active_learning_clean(
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
    logger.info(f"\n{'='*70}")
    logger.info("RESULTS")
    logger.info('='*70)
    
    results_by_strategy = {}
    for strategy in strategies:
        strategy_results = [r for r in all_results if r['strategy'] == strategy]
        final_rmses = [r['rmse_history'][-1] for r in strategy_results]
        
        results_by_strategy[strategy] = {
            'mean_rmse': float(np.mean(final_rmses)),
            'std_rmse': float(np.std(final_rmses)),
            'min_rmse': float(np.min(final_rmses)),
            'max_rmse': float(np.max(final_rmses)),
            'rmse_histories': [r['rmse_history'] for r in strategy_results]
        }
        
        logger.info(f"\n{strategy.upper()}:")
        logger.info(f"  Mean: {np.mean(final_rmses):.2f} ± {np.std(final_rmses):.2f} K")
        logger.info(f"  Range: [{np.min(final_rmses):.2f}, {np.max(final_rmses):.2f}] K")
    
    # Statistical comparisons
    dkl_rmses = [r['rmse_history'][-1] for r in all_results if r['strategy'] == 'dkl']
    gp_rmses = [r['rmse_history'][-1] for r in all_results if r['strategy'] == 'gp']
    random_rmses = [r['rmse_history'][-1] for r in all_results if r['strategy'] == 'random']
    
    _, p_dkl_vs_random = stats.ttest_ind(dkl_rmses, random_rmses)
    _, p_dkl_vs_gp = stats.ttest_ind(dkl_rmses, gp_rmses)
    
    imp_dkl_vs_random = ((np.mean(random_rmses) - np.mean(dkl_rmses)) / np.mean(random_rmses)) * 100
    imp_dkl_vs_gp = ((np.mean(gp_rmses) - np.mean(dkl_rmses)) / np.mean(gp_rmses)) * 100
    
    logger.info(f"\n{'='*70}")
    logger.info("STATISTICAL ANALYSIS")
    logger.info('='*70)
    logger.info(f"\nDKL vs Random: {imp_dkl_vs_random:+.1f}% (p={p_dkl_vs_random:.4f})")
    logger.info(f"DKL vs GP:     {imp_dkl_vs_gp:+.1f}% (p={p_dkl_vs_gp:.4f})")
    
    # Visualization
    logger.info(f"\n📊 Generating plot...")
    fig, ax = plt.subplots(figsize=(12, 7))
    
    x_axis = np.arange(initial_samples, initial_samples + num_rounds * batch_size, batch_size)
    
    for strategy, color, label in [
        ("dkl", "blue", "DKL (Tier 2)"),
        ("gp", "orange", "GP Baseline"),
        ("random", "green", "Random")
    ]:
        histories = np.array(results_by_strategy[strategy]['rmse_histories'])
        mean_hist = np.mean(histories, axis=0)
        std_hist = np.std(histories, axis=0)
        
        ax.plot(x_axis, mean_hist, label=label, color=color, marker='o', linewidth=2)
        ax.fill_between(x_axis, mean_hist - std_hist, mean_hist + std_hist, 
                        color=color, alpha=0.2)
    
    ax.set_xlabel('Labeled Samples', fontsize=12)
    ax.set_ylabel('RMSE (K)', fontsize=12)
    ax.set_title('Phase 10 Tier 2: DKL vs GP vs Random (Clean BoTorch API)', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / 'clean_benchmark.png', dpi=300, bbox_inches='tight')
    logger.info(f"   Saved: {output_dir / 'clean_benchmark.png'}")
    
    # Save results
    results_json = {
        'experiment': 'Phase 10 Tier 2: Clean BoTorch Benchmark',
        'dataset': 'UCI Superconductivity',
        'n_seeds': len(seeds),
        'timestamp': datetime.now().isoformat(),
        'results': results_by_strategy,
        'comparisons': {
            'dkl_vs_random': {
                'improvement_percent': float(imp_dkl_vs_random),
                'p_value': float(p_dkl_vs_random),
                'significant': bool(p_dkl_vs_random < 0.01)
            },
            'dkl_vs_gp': {
                'improvement_percent': float(imp_dkl_vs_gp),
                'p_value': float(p_dkl_vs_gp),
                'significant': bool(p_dkl_vs_gp < 0.01)
            }
        }
    }
    
    with open(output_dir / 'results.json', 'w') as f:
        json.dump(results_json, f, indent=2)
    logger.info(f"   Saved: {output_dir / 'results.json'}")
    
    logger.info(f"\n{'='*70}")
    logger.info("✅ BENCHMARK COMPLETE")
    logger.info('='*70)


if __name__ == "__main__":
    main()

