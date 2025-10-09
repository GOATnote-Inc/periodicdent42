#!/usr/bin/env python3
"""
Noise Sensitivity Study for Conformal-EI vs Vanilla EI

Tests hypothesis: Conformal-EI helps in high-noise regimes where calibrated 
uncertainty becomes more valuable for risk-aware acquisition.

Author: GOATnote Autonomous Research Lab Initiative
Contact: b@thegoatnote.com
"""

import numpy as np
import pandas as pd
import json
import logging
from pathlib import Path
from scipy import stats
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from phase10_gp_active_learning.data.uci_loader import load_uci_superconductor
from experiments.novelty.conformal_ei import run_active_learning

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("noise_sensitivity")


def add_noise(y: np.ndarray, sigma: float, seed: int = None) -> np.ndarray:
    """Add Gaussian noise to targets"""
    if seed is not None:
        np.random.seed(seed)
    return y + np.random.normal(0, sigma, size=len(y))


def main():
    logger.info("="*70)
    logger.info("NOISE SENSITIVITY STUDY")
    logger.info("="*70)
    
    # Noise levels to test (K)
    sigmas = [0, 2, 5, 10, 20, 50]
    
    # Methods to compare
    methods = {
        "vanilla_ei": "vanilla_ei",
        "conformal_ei": "conformal_ei"
    }
    
    # Seeds for statistical power
    seeds = list(range(42, 52))  # 10 seeds per condition (faster than 20)
    num_seeds = len(seeds)
    
    # Output directory
    outdir = Path("experiments/novelty/noise_sensitivity")
    outdir.mkdir(parents=True, exist_ok=True)
    
    # Load clean data
    logger.info("\n📂 Loading clean UCI dataset...")
    train_df, val_df, test_df = load_uci_superconductor()
    feature_cols = [c for c in train_df.columns if c != "Tc"]
    
    X_pool_clean = pd.concat([train_df[feature_cols], val_df[feature_cols]]).values
    y_pool_clean = pd.concat([train_df["Tc"], val_df["Tc"]]).values
    X_test_clean = test_df[feature_cols].values
    y_test_clean = test_df["Tc"].values
    
    logger.info(f"✅ Pool: {len(X_pool_clean)}, Test: {len(X_test_clean)}")
    logger.info(f"✅ Testing {len(sigmas)} noise levels × {num_seeds} seeds × {len(methods)} methods")
    
    # Results storage
    all_results = {}
    
    for sigma in sigmas:
        logger.info(f"\n{'='*70}")
        logger.info(f"NOISE LEVEL: σ = {sigma} K")
        logger.info(f"{'='*70}")
        
        sigma_results = {}
        
        methods_metrics = {}  # Store per method
        
        for method_name, method_key in methods.items():
            logger.info(f"\n🔬 Running {method_name.upper()}...")
            
            method_metrics = []
            
            for i, seed in enumerate(seeds, 1):
                # Add noise with deterministic seed
                y_pool = add_noise(y_pool_clean, sigma, seed=seed * 1000)
                y_test = add_noise(y_test_clean, sigma, seed=seed * 1000 + 1)
                
                # Run active learning
                metrics = run_active_learning(
                    X_pool_clean.copy(), y_pool.copy(),
                    X_test_clean.copy(), y_test.copy(),
                    method=method_key,
                    initial_samples=100,
                    num_rounds=10,
                    alpha=0.1,
                    credibility_weight=1.0,
                    random_seed=seed
                )
                
                method_metrics.append(metrics)
                
                logger.info(f"   Seed {seed}: RMSE={metrics['final_rmse']:.2f} K, "
                           f"Regret={metrics['mean_regret']:.2f} K")
            
            methods_metrics[method_name] = method_metrics  # Store for later
            
            # Aggregate statistics
            rmses = [m['final_rmse'] for m in method_metrics]
            regrets = [m['mean_regret'] for m in method_metrics]
            
            if method_key == "conformal_ei":
                coverages_80 = [m['mean_coverage_80'] for m in method_metrics]
                coverages_90 = [m['mean_coverage_90'] for m in method_metrics]
                pi_widths = [m['mean_pi_width'] for m in method_metrics]
                
                sigma_results[method_name] = {
                    'rmse_mean': float(np.mean(rmses)),
                    'rmse_std': float(np.std(rmses)),
                    'regret_mean': float(np.mean(regrets)),
                    'regret_std': float(np.std(regrets)),
                    'coverage_80_mean': float(np.nanmean(coverages_80)),
                    'coverage_90_mean': float(np.nanmean(coverages_90)),
                    'pi_width_mean': float(np.nanmean(pi_widths))
                }
            else:
                sigma_results[method_name] = {
                    'rmse_mean': float(np.mean(rmses)),
                    'rmse_std': float(np.std(rmses)),
                    'regret_mean': float(np.mean(regrets)),
                    'regret_std': float(np.std(regrets))
                }
            
            logger.info(f"   ✅ {method_name}: RMSE = {np.mean(rmses):.2f} ± {np.std(rmses):.2f} K, "
                       f"Regret = {np.mean(regrets):.2f} ± {np.std(regrets):.2f} K")
        
        # Paired comparison (proper)
        conformal_rmses = [m['final_rmse'] for m in methods_metrics['conformal_ei']]
        vanilla_rmses = [m['final_rmse'] for m in methods_metrics['vanilla_ei']]
        conformal_regrets = [m['mean_regret'] for m in methods_metrics['conformal_ei']]
        vanilla_regrets = [m['mean_regret'] for m in methods_metrics['vanilla_ei']]
        
        # Statistical test (paired t-test)
        if len(conformal_rmses) == len(vanilla_rmses):
            t_stat_rmse, p_val_rmse = stats.ttest_rel(conformal_rmses, vanilla_rmses)
            t_stat_regret, p_val_regret = stats.ttest_rel(conformal_regrets, vanilla_regrets)
            
            sigma_results['comparison'] = {
                'delta_rmse_mean': float(np.mean(np.array(conformal_rmses) - np.array(vanilla_rmses))),
                'p_value_rmse': float(p_val_rmse),
                'delta_regret_mean': float(np.mean(np.array(vanilla_regrets) - np.array(conformal_regrets))),
                'p_value_regret': float(p_val_regret),
                'significant': bool(p_val_regret < 0.05)
            }
        else:
            sigma_results['comparison'] = {
                'note': 'Unequal sample sizes'
            }
        
        all_results[float(sigma)] = sigma_results
        
        logger.info(f"\n📊 Summary for σ={sigma} K:")
        logger.info(f"   Conformal-EI: RMSE = {sigma_results['conformal_ei']['rmse_mean']:.2f} K")
        logger.info(f"   Vanilla EI:   RMSE = {sigma_results['vanilla_ei']['rmse_mean']:.2f} K")
    
    # Save results
    results_file = outdir / "noise_sensitivity_results.json"
    with open(results_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    logger.info(f"\n✅ Results saved to: {results_file}")
    
    # Generate summary
    logger.info(f"\n{'='*70}")
    logger.info("NOISE SENSITIVITY SUMMARY")
    logger.info(f"{'='*70}")
    
    for sigma in sigmas:
        cei_rmse = all_results[float(sigma)]['conformal_ei']['rmse_mean']
        ei_rmse = all_results[float(sigma)]['vanilla_ei']['rmse_mean']
        delta = cei_rmse - ei_rmse
        logger.info(f"σ={sigma:3d} K: CEI={cei_rmse:.2f} K, EI={ei_rmse:.2f} K, Δ={delta:+.2f} K")
    
    logger.info(f"\n{'='*70}")
    logger.info("✅ NOISE SENSITIVITY STUDY COMPLETE")
    logger.info(f"{'='*70}")


if __name__ == "__main__":
    main()

