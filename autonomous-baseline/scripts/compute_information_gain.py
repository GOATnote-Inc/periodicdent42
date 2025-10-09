#!/usr/bin/env python3
"""
Compute epistemic efficiency: information gain per query.

Measures how much uncertainty is reduced per active learning query.

Usage:
    python scripts/compute_information_gain.py \
        --strategy dkl_ei \
        --seeds 3
"""

import argparse
import json
import logging
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from sklearn.preprocessing import StandardScaler
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from phase10_gp_active_learning.data.uci_loader import load_uci_data
from phase10_gp_active_learning.models.dkl_model import create_dkl_model
from phase10_gp_active_learning.models.botorch_dkl import BoTorchDKL
from botorch.acquisition import ExpectedImprovement

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)

def compute_entropy_reduction(
    model_before,
    model_after,
    X_candidates: torch.Tensor
) -> float:
    """
    Compute entropy reduction: H(before) - H(after).
    
    Uses log variance as a proxy for entropy (for Gaussian posteriors).
    
    Args:
        model_before: Model before new observation
        model_after: Model after new observation
        X_candidates: Remaining candidate points
    
    Returns:
        entropy_reduction: ΔH in nats (natural log units)
    """
    with torch.no_grad():
        # Entropy before (higher variance = higher entropy)
        posterior_before = model_before.posterior(X_candidates)
        log_var_before = torch.log(posterior_before.variance + 1e-8)
        H_before = log_var_before.mean().item()
        
        # Entropy after
        posterior_after = model_after.posterior(X_candidates)
        log_var_after = torch.log(posterior_after.variance + 1e-8)
        H_after = log_var_after.mean().item()
        
        # Reduction (positive = we learned something)
        delta_H = H_before - H_after
    
    return delta_H

def run_active_learning_with_information_gain(
    X_pool: pd.DataFrame,
    y_pool: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    initial_samples: int = 100,
    num_rounds: int = 20,
    batch_size: int = 1,
    random_seed: int = 42
) -> dict:
    """
    Run active learning while tracking information gain per query.
    
    Returns:
        results: Dict with RMSE history, information gain history, epistemic efficiency
    """
    set_seed(random_seed)
    logger.info(f"🚀 Running AL with information gain tracking (seed={random_seed})")
    
    # Initialize
    initial_indices = np.random.choice(X_pool.index, initial_samples, replace=False)
    X_labeled = X_pool.loc[initial_indices]
    y_labeled = y_pool.loc[initial_indices]
    X_unlabeled = X_pool.drop(initial_indices)
    y_unlabeled = y_pool.drop(initial_indices)
    
    rmse_history = []
    information_gain_history = []
    
    X_scaler = StandardScaler()
    y_scaler = StandardScaler()
    
    for i in range(num_rounds):
        # Scale data
        X_labeled_scaled = X_scaler.fit_transform(X_labeled)
        y_labeled_scaled = y_scaler.fit_transform(y_labeled.values.reshape(-1, 1)).ravel()
        X_test_scaled = X_scaler.transform(X_test)
        X_unlabeled_scaled = X_scaler.transform(X_unlabeled)
        
        # Train DKL model (BEFORE adding new sample)
        dkl_before = create_dkl_model(
            X_labeled_scaled, y_labeled_scaled,
            input_dim=X_labeled_scaled.shape[1],
            n_epochs=20,
            verbose=False
        )
        model_before = BoTorchDKL(dkl_before)
        
        # Evaluate RMSE
        X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float64)
        with torch.no_grad():
            posterior = model_before.posterior(X_test_tensor)
            y_pred_scaled = posterior.mean.cpu().numpy().ravel()
        
        y_pred = y_scaler.inverse_transform(y_pred_scaled.reshape(-1, 1)).ravel()
        from sklearn.metrics import mean_squared_error
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        rmse_history.append(rmse)
        
        # Select new sample
        if i < num_rounds - 1 and len(X_unlabeled) > 0:
            best_f = y_labeled_scaled.max()
            X_unlabeled_tensor = torch.tensor(X_unlabeled_scaled, dtype=torch.float64)
            
            # Use Expected Improvement
            acq = ExpectedImprovement(model=model_before, best_f=best_f)
            with torch.no_grad():
                if X_unlabeled_tensor.dim() == 2:
                    X_unlabeled_tensor = X_unlabeled_tensor.unsqueeze(1)
                acq_values = acq(X_unlabeled_tensor).squeeze()
            
            best_idx = acq_values.argmax().item()
            selected_index = X_unlabeled.iloc[[best_idx]].index[0]
            
            # Add selected sample
            X_new = X_unlabeled.loc[[selected_index]]
            y_new = y_unlabeled.loc[[selected_index]]
            
            X_labeled_new = pd.concat([X_labeled, X_new])
            y_labeled_new = pd.concat([y_labeled, y_new])
            
            # Retrain model (AFTER adding new sample)
            X_labeled_new_scaled = X_scaler.fit_transform(X_labeled_new)
            y_labeled_new_scaled = y_scaler.fit_transform(y_labeled_new.values.reshape(-1, 1)).ravel()
            X_unlabeled_new_scaled = X_scaler.transform(X_unlabeled.drop(selected_index))
            
            dkl_after = create_dkl_model(
                X_labeled_new_scaled, y_labeled_new_scaled,
                input_dim=X_labeled_new_scaled.shape[1],
                n_epochs=20,
                verbose=False
            )
            model_after = BoTorchDKL(dkl_after)
            
            # Compute information gain on remaining unlabeled pool
            X_remaining_tensor = torch.tensor(X_unlabeled_new_scaled, dtype=torch.float64)
            if len(X_remaining_tensor) > 0:
                delta_H = compute_entropy_reduction(
                    model_before, model_after, X_remaining_tensor
                )
                information_gain_history.append(delta_H)
                
                logger.info(f"   Round {i:2d}: RMSE={rmse:.2f} K, ΔH={delta_H:.4f} nats")
            else:
                information_gain_history.append(0.0)
                logger.info(f"   Round {i:2d}: RMSE={rmse:.2f} K (no candidates left)")
            
            # Update for next round
            X_labeled = X_labeled_new
            y_labeled = y_labeled_new
            X_unlabeled = X_unlabeled.drop(selected_index)
            y_unlabeled = y_unlabeled.drop(selected_index)
        else:
            logger.info(f"   Round {i:2d}: RMSE={rmse:.2f} K (final round)")
    
    # Compute epistemic efficiency
    total_info_gain = sum(information_gain_history)
    avg_info_gain_per_query = total_info_gain / len(information_gain_history) if information_gain_history else 0.0
    
    logger.info(f"\n✅ Total information gain: {total_info_gain:.3f} nats")
    logger.info(f"✅ Avg per query: {avg_info_gain_per_query:.4f} nats/query")
    
    return {
        'rmse_history': rmse_history,
        'information_gain_history': information_gain_history,
        'total_information_gain': float(total_info_gain),
        'avg_information_gain_per_query': float(avg_info_gain_per_query),
        'final_rmse': float(rmse_history[-1])
    }

def main():
    parser = argparse.ArgumentParser(description='Compute epistemic efficiency')
    parser.add_argument('--strategy', type=str, default='dkl_ei')
    parser.add_argument('--seeds', type=int, default=3)
    parser.add_argument('--initial', type=int, default=100)
    parser.add_argument('--rounds', type=int, default=20)
    parser.add_argument('--output', type=Path, default=Path('evidence/phase10/epistemic'))
    args = parser.parse_args()
    
    logger.info("="*70)
    logger.info("EPISTEMIC EFFICIENCY ANALYSIS")
    logger.info("="*70)
    
    # Load data
    logger.info("\n📂 Loading UCI dataset...")
    train_df, val_df, test_df = load_uci_data()
    
    feature_cols = [col for col in train_df.columns if col != 'Tc']
    X_train = train_df[feature_cols]
    y_train = train_df['Tc']
    X_test = test_df[feature_cols]
    y_test = test_df['Tc']
    
    X_pool = pd.concat([X_train, val_df[feature_cols]])
    y_pool = pd.concat([y_train, val_df['Tc']])
    
    seeds = list(range(42, 42 + args.seeds))
    
    all_results = []
    for seed in seeds:
        result = run_active_learning_with_information_gain(
            X_pool.copy(), y_pool.copy(),
            X_test.copy(), y_test.copy(),
            initial_samples=args.initial,
            num_rounds=args.rounds,
            batch_size=1,
            random_seed=seed
        )
        all_results.append(result)
    
    # Aggregate statistics
    avg_info_gains = [r['avg_information_gain_per_query'] for r in all_results]
    
    summary = {
        'experiment': 'Epistemic Efficiency Analysis',
        'strategy': args.strategy,
        'n_seeds': args.seeds,
        'mean_info_gain_per_query': float(np.mean(avg_info_gains)),
        'std_info_gain_per_query': float(np.std(avg_info_gains)),
        'results_per_seed': all_results
    }
    
    # Save
    args.output.mkdir(parents=True, exist_ok=True)
    output_file = args.output / 'information_gain.json'
    
    with open(output_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    logger.info(f"\n✅ Results saved to: {output_file}")
    logger.info("\n" + "="*70)
    logger.info("EPISTEMIC ANALYSIS COMPLETE")
    logger.info("="*70)
    logger.info(f"\nAverage information gain: {summary['mean_info_gain_per_query']:.4f} ± {summary['std_info_gain_per_query']:.4f} nats/query")

if __name__ == '__main__':
    main()

