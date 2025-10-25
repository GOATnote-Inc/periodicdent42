#!/usr/bin/env python3
"""
Compare different acquisition functions for active learning.

Tests: Expected Improvement (EI) vs Probability of Improvement (PI) vs Upper Confidence Bound (UCB)

Usage:
    python scripts/compare_acquisitions.py --methods EI,PI,UCB --seeds 5
"""

import argparse
import json
import logging
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from scipy import stats
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from phase10_gp_active_learning.data.uci_loader import load_uci_superconductor
from phase10_gp_active_learning.models.dkl_model import create_dkl_model
from phase10_gp_active_learning.models.botorch_dkl import BoTorchDKL
from botorch.acquisition import ExpectedImprovement, ProbabilityOfImprovement, UpperConfidenceBound
from botorch.optim import optimize_acqf
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def run_active_learning_with_acquisition(
    acquisition_name: str,
    X_pool: pd.DataFrame,
    y_pool: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    initial_samples: int = 100,
    num_rounds: int = 20,
    batch_size: int = 1,  # q=1 for analytic acquisition
    random_seed: int = 42
) -> list:
    """
    Run active learning with specified acquisition function.
    
    Args:
        acquisition_name: 'EI', 'PI', or 'UCB'
        ...
    
    Returns:
        rmse_history: List of RMSE values per round
    """
    set_seed(random_seed)
    logger.info(f"🚀 Running {acquisition_name} (seed={random_seed})")
    
    # Initialize labeled and unlabeled sets
    initial_indices = np.random.choice(X_pool.index, initial_samples, replace=False)
    X_labeled = X_pool.loc[initial_indices]
    y_labeled = y_pool.loc[initial_indices]
    X_unlabeled = X_pool.drop(initial_indices)
    y_unlabeled = y_pool.drop(initial_indices)
    
    rmse_history = []
    
    # Feature scaling
    X_scaler = StandardScaler()
    y_scaler = StandardScaler()
    
    for i in range(num_rounds):
        # Scale data
        X_labeled_scaled = X_scaler.fit_transform(X_labeled)
        y_labeled_scaled = y_scaler.fit_transform(y_labeled.values.reshape(-1, 1)).ravel()
        X_test_scaled = X_scaler.transform(X_test)
        X_unlabeled_scaled = X_scaler.transform(X_unlabeled)
        
        # Train DKL model
        dkl = create_dkl_model(
            X_labeled_scaled, y_labeled_scaled,
            input_dim=X_labeled_scaled.shape[1],
            n_epochs=20,
            verbose=False
        )
        
        # Wrap for BoTorch
        model = BoTorchDKL(dkl)
        
        # Predict on test set
        X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float64)
        with torch.no_grad():
            posterior = model.posterior(X_test_tensor)
            y_pred_scaled = posterior.mean.cpu().numpy().ravel()
        
        y_pred = y_scaler.inverse_transform(y_pred_scaled.reshape(-1, 1)).ravel()
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        rmse_history.append(rmse)
        
        logger.info(f"   Round {i:2d}: n_labeled={len(X_labeled):4d}, RMSE={rmse:.2f} K")
        
        # Select new sample using acquisition function
        if i < num_rounds - 1 and len(X_unlabeled) > 0:
            best_f = y_labeled_scaled.max()  # Maximize Tc
            X_unlabeled_tensor = torch.tensor(X_unlabeled_scaled, dtype=torch.float64)
            
            # Create acquisition function
            if acquisition_name == 'EI':
                acq = ExpectedImprovement(model=model, best_f=best_f)
            elif acquisition_name == 'PI':
                acq = ProbabilityOfImprovement(model=model, best_f=best_f)
            elif acquisition_name == 'UCB':
                acq = UpperConfidenceBound(model=model, beta=2.0)  # Standard beta
            else:
                raise ValueError(f"Unknown acquisition: {acquisition_name}")
            
            # Optimize acquisition (find best point in unlabeled pool)
            # For efficiency, evaluate on unlabeled pool rather than full optimization
            with torch.no_grad():
                if X_unlabeled_tensor.dim() == 2:
                    X_unlabeled_tensor = X_unlabeled_tensor.unsqueeze(1)  # (N, 1, D)
                acq_values = acq(X_unlabeled_tensor).squeeze()
            
            # Select point with highest acquisition value
            best_idx = acq_values.argmax().item()
            selected_index = X_unlabeled.iloc[[best_idx]].index[0]
            
            # Move to labeled set
            X_labeled = pd.concat([X_labeled, X_unlabeled.loc[[selected_index]]])
            y_labeled = pd.concat([y_labeled, y_unlabeled.loc[[selected_index]]])
            X_unlabeled = X_unlabeled.drop(selected_index)
            y_unlabeled = y_unlabeled.drop(selected_index)
    
    logger.info(f"✅ {acquisition_name}: Final RMSE = {rmse_history[-1]:.2f} K")
    return rmse_history

def main():
    parser = argparse.ArgumentParser(description='Compare acquisition functions')
    parser.add_argument('--methods', type=str, default='EI,PI,UCB',
                       help='Comma-separated: EI,PI,UCB')
    parser.add_argument('--seeds', type=int, default=5, help='Number of seeds')
    parser.add_argument('--initial', type=int, default=100)
    parser.add_argument('--rounds', type=int, default=20)
    parser.add_argument('--batch', type=int, default=1, help='q=1 for analytic')
    parser.add_argument('--output', type=Path, default=Path('evidence/phase10/acquisitions'))
    args = parser.parse_args()
    
    logger.info("="*70)
    logger.info("ACQUISITION FUNCTION COMPARISON")
    logger.info("="*70)
    
    # Load data
    logger.info("\n📂 Loading UCI dataset...")
    train_df, val_df, test_df = load_uci_superconductor()
    
    feature_cols = [col for col in train_df.columns if col != 'Tc']
    X_train = train_df[feature_cols]
    y_train = train_df['Tc']
    X_test = test_df[feature_cols]
    y_test = test_df['Tc']
    
    X_pool = pd.concat([X_train, val_df[feature_cols]])
    y_pool = pd.concat([y_train, val_df['Tc']])
    
    methods = args.methods.split(',')
    seeds = list(range(42, 42 + args.seeds))
    
    all_results = {}
    
    # Run each acquisition function
    for method in methods:
        logger.info(f"\n{'='*70}")
        logger.info(f"TESTING: {method}")
        logger.info(f"{'='*70}")
        
        histories = []
        for seed in seeds:
            history = run_active_learning_with_acquisition(
                method,
                X_pool.copy(), y_pool.copy(),
                X_test.copy(), y_test.copy(),
                initial_samples=args.initial,
                num_rounds=args.rounds,
                batch_size=args.batch,
                random_seed=seed
            )
            histories.append(history)
        
        # Compute statistics
        final_rmses = [h[-1] for h in histories]
        all_results[method] = {
            'mean_rmse': float(np.mean(final_rmses)),
            'std_rmse': float(np.std(final_rmses)),
            'min_rmse': float(np.min(final_rmses)),
            'max_rmse': float(np.max(final_rmses)),
            'rmse_histories': histories
        }
        
        logger.info(f"\n{method} Summary:")
        logger.info(f"  Mean RMSE: {all_results[method]['mean_rmse']:.2f} ± {all_results[method]['std_rmse']:.2f} K")
    
    # Statistical comparisons
    logger.info(f"\n{'='*70}")
    logger.info("STATISTICAL COMPARISONS")
    logger.info(f"{'='*70}")
    
    comparisons = {}
    methods_list = list(all_results.keys())
    for i, method1 in enumerate(methods_list):
        for method2 in methods_list[i+1:]:
            rmse1 = [h[-1] for h in all_results[method1]['rmse_histories']]
            rmse2 = [h[-1] for h in all_results[method2]['rmse_histories']]
            
            t_stat, p_value = stats.ttest_ind(rmse1, rmse2, equal_var=False)
            improvement = ((np.mean(rmse2) - np.mean(rmse1)) / np.mean(rmse2)) * 100
            
            comparisons[f"{method1}_vs_{method2}"] = {
                'improvement_percent': float(improvement),
                'p_value': float(p_value),
                'significant': bool(p_value < 0.05)
            }
            
            sig_str = "significant" if p_value < 0.05 else "not significant"
            logger.info(f"\n{method1} vs {method2}:")
            logger.info(f"  Improvement: {improvement:+.1f}%")
            logger.info(f"  p-value: {p_value:.4f} ({sig_str})")
    
    # Save results
    args.output.mkdir(parents=True, exist_ok=True)
    output_file = args.output / 'acquisition_comparison.json'
    
    with open(output_file, 'w') as f:
        json.dump({
            'experiment': 'Acquisition Function Comparison',
            'dataset': 'UCI Superconductivity',
            'n_seeds': args.seeds,
            'results': all_results,
            'comparisons': comparisons
        }, f, indent=2)
    
    logger.info(f"\n✅ Results saved to: {output_file}")
    logger.info("\n" + "="*70)
    logger.info("ACQUISITION COMPARISON COMPLETE")
    logger.info("="*70)
    
    # Print recommendation
    best_method = min(all_results.keys(), key=lambda k: all_results[k]['mean_rmse'])
    logger.info(f"\n✅ Best acquisition: {best_method} (RMSE = {all_results[best_method]['mean_rmse']:.2f} K)")

if __name__ == '__main__':
    main()

