#!/usr/bin/env python3
"""
Add external baseline comparisons (XGBoost, Random Forest).

Usage:
    python scripts/add_baselines.py --strategies xgboost,random_forest --seeds 5
"""

import argparse
import json
import logging
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from xgboost import XGBRegressor
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from phase10_gp_active_learning.data.uci_loader import load_uci_data

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def set_seed(seed):
    """Set random seeds for reproducibility"""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def run_active_learning_sklearn(
    model,
    X_pool: pd.DataFrame,
    y_pool: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    initial_samples: int = 100,
    num_rounds: int = 20,
    batch_size: int = 20,
    strategy: str = "random",  # sklearn models use random sampling for now
    random_seed: int = 42
) -> list:
    """
    Run active learning with sklearn model.
    
    Note: For simplicity, using random sampling for sklearn models.
    To use true AL, would need uncertainty estimates (e.g., RF variance, quantile regression).
    """
    set_seed(random_seed)
    logger.info(f"🚀 Running {strategy} (seed={random_seed})")
    logger.info(f"   Pool: {len(X_pool)}, Test: {len(X_test)}")
    
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
        
        # Train model
        model_copy = type(model)(**model.get_params())  # Fresh model each round
        model_copy.fit(X_labeled_scaled, y_labeled_scaled)
        
        # Predict
        y_pred_scaled = model_copy.predict(X_test_scaled)
        y_pred = y_scaler.inverse_transform(y_pred_scaled.reshape(-1, 1)).ravel()
        
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        rmse_history.append(rmse)
        logger.info(f"   Round {i:2d}: n_labeled={len(X_labeled):4d}, RMSE={rmse:.2f} K")
        
        # Select new samples (random for simplicity)
        if i < num_rounds - 1 and len(X_unlabeled) > 0:
            num_to_select = min(batch_size, len(X_unlabeled))
            selected_indices = np.random.choice(X_unlabeled.index, num_to_select, replace=False)
            
            # Move to labeled set
            X_labeled = pd.concat([X_labeled, X_unlabeled.loc[selected_indices]])
            y_labeled = pd.concat([y_labeled, y_unlabeled.loc[selected_indices]])
            X_unlabeled = X_unlabeled.drop(selected_indices)
            y_unlabeled = y_unlabeled.drop(selected_indices)
    
    logger.info(f"✅ {strategy}: Final RMSE = {rmse_history[-1]:.2f} K")
    return rmse_history

def benchmark_xgboost(
    X_pool, y_pool, X_test, y_test,
    seeds: list,
    initial: int = 100,
    rounds: int = 20,
    batch: int = 20
) -> dict:
    """Benchmark XGBoost with active learning"""
    
    logger.info("\n" + "="*70)
    logger.info("XGBOOST BENCHMARK")
    logger.info("="*70)
    
    model = XGBRegressor(
        n_estimators=500,
        max_depth=8,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=1.0,
        random_state=42,
        n_jobs=-1,
        verbosity=0
    )
    
    all_histories = []
    for seed in seeds:
        history = run_active_learning_sklearn(
            model,
            X_pool.copy(), y_pool.copy(),
            X_test.copy(), y_test.copy(),
            initial_samples=initial,
            num_rounds=rounds,
            batch_size=batch,
            strategy='xgboost',
            random_seed=seed
        )
        all_histories.append(history)
    
    # Compute statistics
    final_rmses = [h[-1] for h in all_histories]
    results = {
        'mean_rmse': float(np.mean(final_rmses)),
        'std_rmse': float(np.std(final_rmses)),
        'min_rmse': float(np.min(final_rmses)),
        'max_rmse': float(np.max(final_rmses)),
        'rmse_histories': all_histories
    }
    
    logger.info(f"\nXGBoost Summary:")
    logger.info(f"  Mean RMSE: {results['mean_rmse']:.2f} ± {results['std_rmse']:.2f} K")
    logger.info(f"  Range: [{results['min_rmse']:.2f}, {results['max_rmse']:.2f}] K")
    
    return results

def benchmark_random_forest(
    X_pool, y_pool, X_test, y_test,
    seeds: list,
    initial: int = 100,
    rounds: int = 20,
    batch: int = 20
) -> dict:
    """Benchmark Random Forest with active learning"""
    
    logger.info("\n" + "="*70)
    logger.info("RANDOM FOREST BENCHMARK")
    logger.info("="*70)
    
    model = RandomForestRegressor(
        n_estimators=500,
        max_depth=None,
        min_samples_split=5,
        min_samples_leaf=2,
        max_features='sqrt',
        bootstrap=True,
        random_state=42,
        n_jobs=-1,
        verbose=0
    )
    
    all_histories = []
    for seed in seeds:
        history = run_active_learning_sklearn(
            model,
            X_pool.copy(), y_pool.copy(),
            X_test.copy(), y_test.copy(),
            initial_samples=initial,
            num_rounds=rounds,
            batch_size=batch,
            strategy='random_forest',
            random_seed=seed
        )
        all_histories.append(history)
    
    # Compute statistics
    final_rmses = [h[-1] for h in all_histories]
    results = {
        'mean_rmse': float(np.mean(final_rmses)),
        'std_rmse': float(np.std(final_rmses)),
        'min_rmse': float(np.min(final_rmses)),
        'max_rmse': float(np.max(final_rmses)),
        'rmse_histories': all_histories
    }
    
    logger.info(f"\nRandom Forest Summary:")
    logger.info(f"  Mean RMSE: {results['mean_rmse']:.2f} ± {results['std_rmse']:.2f} K")
    logger.info(f"  Range: [{results['min_rmse']:.2f}, {results['max_rmse']:.2f}] K")
    
    return results

def main():
    parser = argparse.ArgumentParser(description='Add external baseline comparisons')
    parser.add_argument('--strategies', type=str, default='xgboost,random_forest',
                       help='Comma-separated list: xgboost,random_forest')
    parser.add_argument('--seeds', type=int, default=5, help='Number of random seeds')
    parser.add_argument('--initial', type=int, default=100, help='Initial labeled samples')
    parser.add_argument('--rounds', type=int, default=20, help='Number of AL rounds')
    parser.add_argument('--batch', type=int, default=20, help='Batch size per round')
    parser.add_argument('--output', type=Path, default=Path('evidence/phase10/baselines'),
                       help='Output directory')
    args = parser.parse_args()
    
    # Load data
    logger.info("📂 Loading UCI dataset...")
    train_df, val_df, test_df = load_uci_data()
    
    feature_cols = [col for col in train_df.columns if col != 'Tc']
    X_train = train_df[feature_cols]
    y_train = train_df['Tc']
    X_test = test_df[feature_cols]
    y_test = test_df['Tc']
    
    # Combine train and val for pool
    X_pool = pd.concat([X_train, val_df[feature_cols]])
    y_pool = pd.concat([y_train, val_df['Tc']])
    
    seeds = list(range(42, 42 + args.seeds))
    strategies = args.strategies.split(',')
    
    all_results = {}
    
    # Run benchmarks
    if 'xgboost' in strategies:
        all_results['xgboost'] = benchmark_xgboost(
            X_pool, y_pool, X_test, y_test,
            seeds, args.initial, args.rounds, args.batch
        )
    
    if 'random_forest' in strategies:
        all_results['random_forest'] = benchmark_random_forest(
            X_pool, y_pool, X_test, y_test,
            seeds, args.initial, args.rounds, args.batch
        )
    
    # Save results
    args.output.mkdir(parents=True, exist_ok=True)
    output_file = args.output / 'baseline_results.json'
    
    with open(output_file, 'w') as f:
        json.dump({
            'experiment': 'External Baseline Comparison',
            'dataset': 'UCI Superconductivity',
            'n_seeds': args.seeds,
            'results': all_results
        }, f, indent=2)
    
    logger.info(f"\n✅ Results saved to: {output_file}")
    logger.info("\n" + "="*70)
    logger.info("BASELINE COMPARISON COMPLETE")
    logger.info("="*70)

if __name__ == '__main__':
    main()

