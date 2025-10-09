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
    strategy: str = "random",
    random_seed: int = 42,
    uncertainty_aware: bool = True
) -> dict:
    """
    Run uncertainty-aware active learning with sklearn model.
    
    For RF: Select by ensemble variance across trees
    For XGB: Train quantile models and select by PI width
    """
    set_seed(random_seed)
    logger.info(f"🚀 Running {strategy} (seed={random_seed}, uncertainty_aware={uncertainty_aware})")
    logger.info(f"   Pool: {len(X_pool)}, Test: {len(X_test)}")
    
    # Initialize labeled and unlabeled sets
    initial_indices = np.random.choice(X_pool.index, initial_samples, replace=False)
    X_labeled = X_pool.loc[initial_indices]
    y_labeled = y_pool.loc[initial_indices]
    X_unlabeled = X_pool.drop(initial_indices)
    y_unlabeled = y_pool.drop(initial_indices)
    
    rmse_history = []
    coverage_80_history = []
    coverage_90_history = []
    pi_width_history = []
    
    # Feature scaling
    X_scaler = StandardScaler()
    y_scaler = StandardScaler()
    
    for i in range(num_rounds):
        # Scale data
        X_labeled_scaled = X_scaler.fit_transform(X_labeled)
        y_labeled_scaled = y_scaler.fit_transform(y_labeled.values.reshape(-1, 1)).ravel()
        X_test_scaled = X_scaler.transform(X_test)
        X_unlabeled_scaled = X_scaler.transform(X_unlabeled) if len(X_unlabeled) > 0 else None
        
        # Train model
        model_copy = type(model)(**model.get_params())
        model_copy.fit(X_labeled_scaled, y_labeled_scaled)
        
        # Predict on test set
        y_pred_scaled = model_copy.predict(X_test_scaled)
        y_pred = y_scaler.inverse_transform(y_pred_scaled.reshape(-1, 1)).ravel()
        
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        rmse_history.append(rmse)
        
        # Compute uncertainty estimates and coverage
        if isinstance(model_copy, RandomForestRegressor):
            # RF: Ensemble variance
            predictions = np.array([tree.predict(X_test_scaled) for tree in model_copy.estimators_])
            y_pred_std_scaled = predictions.std(axis=0)
            y_pred_std = y_pred_std_scaled * y_scaler.scale_[0]  # Unscale
            
            # Coverage at 80% and 90% (1.28σ and 1.645σ)
            lower_80 = y_pred - 1.28 * y_pred_std
            upper_80 = y_pred + 1.28 * y_pred_std
            coverage_80 = np.mean((y_test >= lower_80) & (y_test <= upper_80))
            
            lower_90 = y_pred - 1.645 * y_pred_std
            upper_90 = y_pred + 1.645 * y_pred_std
            coverage_90 = np.mean((y_test >= lower_90) & (y_test <= upper_90))
            
            pi_width = (upper_90 - lower_90).mean()
            
        elif 'XGB' in type(model_copy).__name__:
            # XGBoost: Train quantile models
            from xgboost import XGBRegressor
            
            # Train quantile models for 90% PI (α=0.05, 0.95)
            model_lower = XGBRegressor(**{**model.get_params(), 'objective': 'reg:quantileerror', 'quantile_alpha': 0.05})
            model_upper = XGBRegressor(**{**model.get_params(), 'objective': 'reg:quantileerror', 'quantile_alpha': 0.95})
            model_lower.fit(X_labeled_scaled, y_labeled_scaled, verbose=False)
            model_upper.fit(X_labeled_scaled, y_labeled_scaled, verbose=False)
            
            lower_90_scaled = model_lower.predict(X_test_scaled)
            upper_90_scaled = model_upper.predict(X_test_scaled)
            lower_90 = y_scaler.inverse_transform(lower_90_scaled.reshape(-1, 1)).ravel()
            upper_90 = y_scaler.inverse_transform(upper_90_scaled.reshape(-1, 1)).ravel()
            
            coverage_90 = np.mean((y_test >= lower_90) & (y_test <= upper_90))
            pi_width = (upper_90 - lower_90).mean()
            
            # Approximate 80% using 0.1, 0.9 quantiles
            model_lower_80 = XGBRegressor(**{**model.get_params(), 'objective': 'reg:quantileerror', 'quantile_alpha': 0.1})
            model_upper_80 = XGBRegressor(**{**model.get_params(), 'objective': 'reg:quantileerror', 'quantile_alpha': 0.9})
            model_lower_80.fit(X_labeled_scaled, y_labeled_scaled, verbose=False)
            model_upper_80.fit(X_labeled_scaled, y_labeled_scaled, verbose=False)
            
            lower_80_scaled = model_lower_80.predict(X_test_scaled)
            upper_80_scaled = model_upper_80.predict(X_test_scaled)
            lower_80 = y_scaler.inverse_transform(lower_80_scaled.reshape(-1, 1)).ravel()
            upper_80 = y_scaler.inverse_transform(upper_80_scaled.reshape(-1, 1)).ravel()
            
            coverage_80 = np.mean((y_test >= lower_80) & (y_test <= upper_80))
        
        coverage_80_history.append(coverage_80)
        coverage_90_history.append(coverage_90)
        pi_width_history.append(pi_width)
        
        logger.info(f"   Round {i:2d}: n_labeled={len(X_labeled):4d}, RMSE={rmse:.2f} K, "
                   f"Cov@80={coverage_80:.2f}, Cov@90={coverage_90:.2f}, PI_width={pi_width:.1f} K")
        
        # Select new samples (uncertainty-aware or random)
        if i < num_rounds - 1 and len(X_unlabeled) > 0:
            num_to_select = min(batch_size, len(X_unlabeled))
            
            if uncertainty_aware:
                # Compute uncertainty on unlabeled pool
                if isinstance(model_copy, RandomForestRegressor):
                    predictions_pool = np.array([tree.predict(X_unlabeled_scaled) for tree in model_copy.estimators_])
                    uncertainties = predictions_pool.std(axis=0)
                elif 'XGB' in type(model_copy).__name__:
                    # Use PI width as uncertainty proxy
                    lower_pool_scaled = model_lower.predict(X_unlabeled_scaled)
                    upper_pool_scaled = model_upper.predict(X_unlabeled_scaled)
                    lower_pool = y_scaler.inverse_transform(lower_pool_scaled.reshape(-1, 1)).ravel()
                    upper_pool = y_scaler.inverse_transform(upper_pool_scaled.reshape(-1, 1)).ravel()
                    uncertainties = upper_pool - lower_pool
                
                # Select points with highest uncertainty
                top_uncertain_indices = np.argsort(uncertainties)[-num_to_select:]
                selected_indices = X_unlabeled.iloc[top_uncertain_indices].index
            else:
                # Random selection
                selected_indices = np.random.choice(X_unlabeled.index, num_to_select, replace=False)
            
            # Move to labeled set
            X_labeled = pd.concat([X_labeled, X_unlabeled.loc[selected_indices]])
            y_labeled = pd.concat([y_labeled, y_unlabeled.loc[selected_indices]])
            X_unlabeled = X_unlabeled.drop(selected_indices)
            y_unlabeled = y_unlabeled.drop(selected_indices)
    
    logger.info(f"✅ {strategy}: Final RMSE = {rmse_history[-1]:.2f} K, "
               f"Cov@90 = {coverage_90_history[-1]:.2f}")
    
    return {
        'rmse_history': rmse_history,
        'coverage_80_history': coverage_80_history,
        'coverage_90_history': coverage_90_history,
        'pi_width_history': pi_width_history
    }

def benchmark_xgboost(
    X_pool, y_pool, X_test, y_test,
    seeds: list,
    initial: int = 100,
    rounds: int = 20,
    batch: int = 20,
    uncertainty_aware: bool = True
) -> dict:
    """Benchmark XGBoost with uncertainty-aware active learning"""
    
    logger.info("\n" + "="*70)
    logger.info(f"XGBOOST BENCHMARK (uncertainty_aware={uncertainty_aware})")
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
    
    all_results = []
    for seed in seeds:
        result = run_active_learning_sklearn(
            model,
            X_pool.copy(), y_pool.copy(),
            X_test.copy(), y_test.copy(),
            initial_samples=initial,
            num_rounds=rounds,
            batch_size=batch,
            strategy='xgboost',
            random_seed=seed,
            uncertainty_aware=uncertainty_aware
        )
        all_results.append(result)
    
    # Aggregate statistics
    final_rmses = [r['rmse_history'][-1] for r in all_results]
    final_cov80 = [r['coverage_80_history'][-1] for r in all_results]
    final_cov90 = [r['coverage_90_history'][-1] for r in all_results]
    final_pi_width = [r['pi_width_history'][-1] for r in all_results]
    
    results = {
        'mean_rmse': float(np.mean(final_rmses)),
        'std_rmse': float(np.std(final_rmses)),
        'min_rmse': float(np.min(final_rmses)),
        'max_rmse': float(np.max(final_rmses)),
        'mean_coverage_80': float(np.mean(final_cov80)),
        'mean_coverage_90': float(np.mean(final_cov90)),
        'mean_pi_width': float(np.mean(final_pi_width)),
        'rmse_histories': [r['rmse_history'] for r in all_results],
        'coverage_80_histories': [r['coverage_80_history'] for r in all_results],
        'coverage_90_histories': [r['coverage_90_history'] for r in all_results],
        'pi_width_histories': [r['pi_width_history'] for r in all_results]
    }
    
    logger.info(f"\nXGBoost Summary:")
    logger.info(f"  Mean RMSE: {results['mean_rmse']:.2f} ± {results['std_rmse']:.2f} K")
    logger.info(f"  Coverage@80: {results['mean_coverage_80']:.2f}")
    logger.info(f"  Coverage@90: {results['mean_coverage_90']:.2f}")
    logger.info(f"  PI Width: {results['mean_pi_width']:.1f} K")
    
    return results

def benchmark_random_forest(
    X_pool, y_pool, X_test, y_test,
    seeds: list,
    initial: int = 100,
    rounds: int = 20,
    batch: int = 20,
    uncertainty_aware: bool = True
) -> dict:
    """Benchmark Random Forest with uncertainty-aware active learning"""
    
    logger.info("\n" + "="*70)
    logger.info(f"RANDOM FOREST BENCHMARK (uncertainty_aware={uncertainty_aware})")
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
    
    all_results = []
    for seed in seeds:
        result = run_active_learning_sklearn(
            model,
            X_pool.copy(), y_pool.copy(),
            X_test.copy(), y_test.copy(),
            initial_samples=initial,
            num_rounds=rounds,
            batch_size=batch,
            strategy='random_forest',
            random_seed=seed,
            uncertainty_aware=uncertainty_aware
        )
        all_results.append(result)
    
    # Aggregate statistics
    final_rmses = [r['rmse_history'][-1] for r in all_results]
    final_cov80 = [r['coverage_80_history'][-1] for r in all_results]
    final_cov90 = [r['coverage_90_history'][-1] for r in all_results]
    final_pi_width = [r['pi_width_history'][-1] for r in all_results]
    
    results = {
        'mean_rmse': float(np.mean(final_rmses)),
        'std_rmse': float(np.std(final_rmses)),
        'min_rmse': float(np.min(final_rmses)),
        'max_rmse': float(np.max(final_rmses)),
        'mean_coverage_80': float(np.mean(final_cov80)),
        'mean_coverage_90': float(np.mean(final_cov90)),
        'mean_pi_width': float(np.mean(final_pi_width)),
        'rmse_histories': [r['rmse_history'] for r in all_results],
        'coverage_80_histories': [r['coverage_80_history'] for r in all_results],
        'coverage_90_histories': [r['coverage_90_history'] for r in all_results],
        'pi_width_histories': [r['pi_width_history'] for r in all_results]
    }
    
    logger.info(f"\nRandom Forest Summary:")
    logger.info(f"  Mean RMSE: {results['mean_rmse']:.2f} ± {results['std_rmse']:.2f} K")
    logger.info(f"  Coverage@80: {results['mean_coverage_80']:.2f}")
    logger.info(f"  Coverage@90: {results['mean_coverage_90']:.2f}")
    logger.info(f"  PI Width: {results['mean_pi_width']:.1f} K")
    
    return results

def main():
    parser = argparse.ArgumentParser(description='Add external baseline comparisons')
    parser.add_argument('--strategies', type=str, default='xgboost,random_forest',
                       help='Comma-separated list: xgboost,random_forest')
    parser.add_argument('--seeds', type=int, default=5, help='Number of random seeds')
    parser.add_argument('--initial', type=int, default=100, help='Initial labeled samples')
    parser.add_argument('--rounds', type=int, default=20, help='Number of AL rounds')
    parser.add_argument('--batch', type=int, default=20, help='Batch size per round')
    parser.add_argument('--uncertainty-aware', action='store_true', default=True,
                       help='Use uncertainty-aware selection (RF variance, XGB quantiles)')
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
            seeds, args.initial, args.rounds, args.batch,
            uncertainty_aware=args.uncertainty_aware
        )
    
    if 'random_forest' in strategies:
        all_results['random_forest'] = benchmark_random_forest(
            X_pool, y_pool, X_test, y_test,
            seeds, args.initial, args.rounds, args.batch,
            uncertainty_aware=args.uncertainty_aware
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

