"""
DKL Ablation Study: Isolate Feature Learning Contribution (REAL IMPLEMENTATION)

Addresses Critical Flaw #4: "DKL beats GP lacks depth"
- Compares 4 methods: DKL vs PCA+GP vs Random+GP vs GP-raw
- Isolates feature learning value: DKL - PCA
- Uses real active learning experiments (not placeholders)
- Reports wall-clock time for each method

Usage:
    python scripts/tier2_dkl_ablation_real.py --seeds 5 --rounds 20

References:
    - Wilson et al. (2016): "Deep Kernel Learning" (original DKL paper)
    - Rasmussen & Williams (2006): "Gaussian Processes for Machine Learning"
    - Tipping & Bishop (1999): "Probabilistic Principal Component Analysis"

Author: GOATnote Autonomous Research Lab Initiative
Contact: b@thegoatnote.com
License: MIT
"""

import argparse
import json
import logging
import numpy as np
import time
import torch
import sys
from pathlib import Path
from typing import Dict, List, Tuple
from sklearn.decomposition import PCA
from sklearn.random_projection import GaussianRandomProjection
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from scipy import stats as scipy_stats
from datetime import datetime

# Add paths
sys.path.insert(0, str(Path(__file__).parent.parent))

from phase10_gp_active_learning.data.uci_loader import load_uci_superconductor
from phase10_gp_active_learning.models.dkl_model import create_dkl_model
from phase10_gp_active_learning.models.botorch_dkl import BoTorchDKL

# BoTorch imports
from botorch.models import SingleTaskGP
from botorch.fit import fit_gpytorch_mll
from gpytorch.mlls import ExactMarginalLogLikelihood
from botorch.acquisition import ExpectedImprovement

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("dkl_ablation")


def set_seed(seed):
    """Set all random seeds for reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def run_ablation_experiment(
    method: str,
    X_pool: np.ndarray,
    y_pool: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    seed: int,
    latent_dim: int = 16,
    initial_samples: int = 100,
    num_rounds: int = 20,
    batch_size: int = 20
) -> Dict:
    """
    Run single ablation experiment with REAL active learning
    
    Args:
        method: 'dkl', 'pca_gp', 'random_gp', or 'gp_raw'
        X_pool, y_pool: Unlabeled pool
        X_test, y_test: Test set
        seed: Random seed
        latent_dim: Latent dimensionality (for PCA/Random projection)
        initial_samples: Initial labeled samples
        num_rounds: Number of AL rounds
        batch_size: Samples per round
    
    Returns:
        Dictionary with results
    """
    set_seed(seed)
    
    logger.info(f"  Running {method} (latent_dim={latent_dim}, seed={seed})...")
    
    start_time = time.time()
    
    # Normalize features
    scaler = StandardScaler()
    X_pool_scaled = scaler.fit_transform(X_pool)
    X_test_scaled = scaler.transform(X_test)
    
    # Apply dimensionality reduction if needed
    if method == 'pca_gp':
        # PCA to latent_dim dimensions
        pca = PCA(n_components=latent_dim, random_state=seed)
        X_pool_reduced = pca.fit_transform(X_pool_scaled)
        X_test_reduced = pca.transform(X_test_scaled)
        explained_var = pca.explained_variance_ratio_.sum()
        logger.info(f"    PCA: {latent_dim}D explains {explained_var:.3f} variance")
        
    elif method == 'random_gp':
        # Random projection to latent_dim dimensions
        projector = GaussianRandomProjection(n_components=latent_dim, random_state=seed)
        X_pool_reduced = projector.fit_transform(X_pool_scaled)
        X_test_reduced = projector.transform(X_test_scaled)
        logger.info(f"    Random projection: {latent_dim}D")
        
    elif method == 'gp_raw':
        # No dimensionality reduction - use all 81 features
        X_pool_reduced = X_pool_scaled
        X_test_reduced = X_test_scaled
        logger.info(f"    GP on raw features: {X_pool_scaled.shape[1]}D")
        
    elif method == 'dkl':
        # DKL learns its own features - start with full data
        X_pool_reduced = X_pool_scaled
        X_test_reduced = X_test_scaled
        logger.info(f"    DKL with {latent_dim}D latent space")
    
    # Initial random selection
    indices = np.random.choice(len(X_pool_reduced), initial_samples, replace=False)
    labeled_mask = np.zeros(len(X_pool_reduced), dtype=bool)
    labeled_mask[indices] = True
    
    rmse_history = []
    training_times = []
    
    # Active learning loop
    for round_idx in range(num_rounds):
        # Get current labeled data
        X_labeled = X_pool_reduced[labeled_mask]
        y_labeled = y_pool[labeled_mask]
        
        # Convert to torch tensors
        X_train_t = torch.tensor(X_labeled, dtype=torch.float64)
        y_train_t = torch.tensor(y_labeled, dtype=torch.float64)
        X_test_t = torch.tensor(X_test_reduced, dtype=torch.float64)
        
        # Train model
        train_start = time.time()
        
        if method == 'dkl':
            # Train DKL (learns features)
            dkl = create_dkl_model(
                X_labeled, y_labeled,
                input_dim=X_labeled.shape[1],
                output_dim=latent_dim,  # Use output_dim parameter
                n_epochs=50,
                lr=0.001,
                verbose=False
            )
            model = BoTorchDKL(dkl)
            
        else:  # pca_gp, random_gp, gp_raw
            # Train standard GP on reduced features
            model = SingleTaskGP(X_train_t, y_train_t.unsqueeze(-1))
            mll = ExactMarginalLogLikelihood(model.likelihood, model)
            fit_gpytorch_mll(mll)
        
        train_time = time.time() - train_start
        training_times.append(train_time)
        
        # Evaluate on test set
        model.eval()
        with torch.no_grad():
            posterior = model.posterior(X_test_t)
            y_pred = posterior.mean.squeeze().cpu().numpy()
        
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        rmse_history.append(rmse)
        
        if round_idx % 5 == 0 or round_idx == num_rounds - 1:
            logger.info(f"    Round {round_idx:2d}: n_labeled={labeled_mask.sum():4d}, RMSE={rmse:.2f} K, train_time={train_time:.1f}s")
        
        # Select new samples (using random selection for fair comparison)
        # Note: Could use EI, but random is simpler and fair across methods
        if round_idx < num_rounds - 1:
            unlabeled_indices = np.where(~labeled_mask)[0]
            num_to_select = min(batch_size, len(unlabeled_indices))
            
            if num_to_select == 0:
                break
            
            # Random selection (fair baseline)
            new_indices = np.random.choice(unlabeled_indices, num_to_select, replace=False)
            labeled_mask[new_indices] = True
    
    total_time = time.time() - start_time
    final_rmse = rmse_history[-1]
    
    logger.info(f"    FINAL: RMSE={final_rmse:.2f} K, Total time={total_time:.1f}s")
    
    return {
        'method': method,
        'latent_dim': latent_dim,
        'seed': seed,
        'rmse_history': [float(r) for r in rmse_history],
        'final_rmse': float(final_rmse),
        'training_times': [float(t) for t in training_times],
        'total_time': float(total_time),
        'avg_train_time': float(np.mean(training_times)),
        'num_rounds': num_rounds,
        'initial_samples': initial_samples
    }


def compute_statistics(results: List[Dict], method: str) -> Dict:
    """Compute aggregate statistics across seeds"""
    final_rmses = [r['final_rmse'] for r in results]
    total_times = [r['total_time'] for r in results]
    
    return {
        'method': method,
        'n_seeds': len(results),
        'rmse_mean': float(np.mean(final_rmses)),
        'rmse_std': float(np.std(final_rmses)),
        'rmse_ci_lower': float(np.percentile(final_rmses, 2.5)),
        'rmse_ci_upper': float(np.percentile(final_rmses, 97.5)),
        'time_mean': float(np.mean(total_times)),
        'time_std': float(np.std(total_times))
    }


def main():
    parser = argparse.ArgumentParser(description="DKL ablation study (REAL experiments)")
    parser.add_argument(
        '--seeds',
        type=int,
        default=5,
        help='Number of seeds to run'
    )
    parser.add_argument(
        '--latent-dim',
        type=int,
        default=16,
        help='Latent dimensionality for PCA/Random/DKL'
    )
    parser.add_argument(
        '--rounds',
        type=int,
        default=20,
        help='Number of AL rounds'
    )
    parser.add_argument(
        '--output',
        type=Path,
        default=Path('experiments/ablations'),
        help='Output directory'
    )
    args = parser.parse_args()
    
    # Parse arguments
    seeds = list(range(42, 42 + args.seeds))
    methods = ['dkl', 'pca_gp', 'random_gp', 'gp_raw']
    
    logger.info("=" * 80)
    logger.info("DKL ABLATION STUDY (REAL EXPERIMENTS)")
    logger.info("=" * 80)
    logger.info(f"Methods: {', '.join(methods)}")
    logger.info(f"Latent dim: {args.latent_dim}D")
    logger.info(f"Seeds: {len(seeds)} ({seeds[0]}-{seeds[-1]})")
    logger.info(f"AL rounds: {args.rounds}")
    logger.info("=" * 80)
    
    # Load data
    logger.info("Loading UCI Superconductor dataset...")
    train_df, val_df, test_df = load_uci_superconductor()
    
    # Extract features and targets
    feature_cols = [c for c in train_df.columns if c != 'Tc']
    X_train = train_df[feature_cols].values
    y_train = train_df['Tc'].values
    X_val = val_df[feature_cols].values
    y_val = val_df['Tc'].values
    X_test = test_df[feature_cols].values
    y_test = test_df['Tc'].values
    
    # Use validation as pool for AL
    X_pool = X_val
    y_pool = y_val
    
    logger.info(f"  Train: {len(X_train)} samples (unused in AL)")
    logger.info(f"  Pool: {len(X_pool)} samples (for AL)")
    logger.info(f"  Test: {len(X_test)} samples (for evaluation)")
    logger.info("=" * 80)
    
    # Run experiments
    all_results = {}
    
    for method in methods:
        logger.info(f"Method: {method.upper()}")
        method_results = []
        
        for seed in seeds:
            result = run_ablation_experiment(
                method=method,
                X_pool=X_pool,
                y_pool=y_pool,
                X_test=X_test,
                y_test=y_test,
                seed=seed,
                latent_dim=args.latent_dim,
                num_rounds=args.rounds
            )
            method_results.append(result)
        
        # Compute statistics
        stats = compute_statistics(method_results, method)
        logger.info(f"  {method:12s}: RMSE = {stats['rmse_mean']:.2f} ± {stats['rmse_std']:.2f} K, Time = {stats['time_mean']:.1f} ± {stats['time_std']:.1f} s")
        
        all_results[method] = {
            'statistics': stats,
            'seed_results': method_results
        }
        
        logger.info("")
    
    # Statistical comparisons
    logger.info("=" * 80)
    logger.info("STATISTICAL COMPARISONS (Paired t-tests)")
    logger.info("=" * 80)
    
    dkl_rmses = [r['final_rmse'] for r in all_results['dkl']['seed_results']]
    
    for method in ['pca_gp', 'random_gp', 'gp_raw']:
        method_rmses = [r['final_rmse'] for r in all_results[method]['seed_results']]
        
        # Paired t-test
        t_stat, p_value = scipy_stats.ttest_rel(dkl_rmses, method_rmses)
        delta_rmse = np.mean(dkl_rmses) - np.mean(method_rmses)
        cohens_d = delta_rmse / np.std(np.array(dkl_rmses) - np.array(method_rmses))
        
        logger.info(f"DKL vs {method:12s}: Δ RMSE = {delta_rmse:+.2f} K, p = {p_value:.4f}, Cohen's d = {cohens_d:.3f}")
        
        all_results[method]['comparison_vs_dkl'] = {
            'delta_rmse': float(delta_rmse),
            'p_value': float(p_value),
            'cohens_d': float(cohens_d),
            'significant': bool(p_value < 0.05)
        }
    
    logger.info("=" * 80)
    
    # Save results
    args.output.mkdir(parents=True, exist_ok=True)
    output_path = args.output / 'dkl_ablation_results_real.json'
    
    with open(output_path, 'w') as f:
        json.dump({
            'metadata': {
                'script': 'tier2_dkl_ablation_real.py',
                'timestamp': datetime.now().isoformat(),
                'methods': methods,
                'latent_dim': args.latent_dim,
                'n_seeds': len(seeds),
                'seeds': seeds,
                'n_rounds': args.rounds
            },
            'results': all_results
        }, f, indent=2)
    
    logger.info(f"✅ Saved: {output_path}")
    logger.info("=" * 80)
    logger.info("ABLATION STUDY COMPLETE")
    logger.info("=" * 80)


if __name__ == '__main__':
    main()

