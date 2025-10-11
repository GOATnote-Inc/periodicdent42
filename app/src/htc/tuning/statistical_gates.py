"""
Statistical validation gates for v0.5.0 accuracy tuning.

Bootstrap CI analysis and MAE decision aid for ΔMAPE validation.

© 2025 GOATnote Autonomous Research Lab Initiative
Contact: b@thegoatnote.com
"""
import numpy as np
import pandas as pd
from typing import Dict
import logging

logger = logging.getLogger(__name__)


def compute_delta_mape_ci(
    current_results: pd.DataFrame,
    baseline_results: pd.DataFrame,
    n_bootstrap: int = 1000,
    alpha: float = 0.05,
    seed: int = 42,
    stratify_by_tier: bool = True
) -> Dict:
    """
    Bootstrap 95% CI on ΔMAPE (stratified by tier).
    
    Args:
        current_results: Current predictions with 'material', 'tier', 'rel_error_pct', 'abs_error'
        baseline_results: Baseline predictions (same schema)
        n_bootstrap: Number of bootstrap resamples
        alpha: Significance level (0.05 for 95% CI)
        seed: Random seed for reproducibility
        stratify_by_tier: Stratify sampling by tier (recommended)
    
    Returns:
        {
            'delta_mape_mean': float,
            'delta_mape_std': float,
            'delta_mape_ci_lower': float,
            'delta_mape_ci_upper': float,
            'delta_mape_excludes_zero': bool,
            'delta_mape_p_value': float,
            'delta_mae_mean': float,
            'delta_mae_ci_lower': float,
            'delta_mae_ci_upper': float,
            'delta_mae_excludes_zero': bool,
            'n_bootstrap': int,
            'alpha': float,
        }
    """
    # Merge on material
    merged = current_results[['material', 'tier', 'rel_error_pct', 'abs_error']].merge(
        baseline_results[['material', 'rel_error_pct', 'abs_error']],
        on='material',
        suffixes=('_curr', '_base')
    ).dropna()
    
    if len(merged) == 0:
        raise ValueError("No overlapping materials between current and baseline")
    
    rng = np.random.default_rng(seed)
    delta_mapes = []
    delta_maes = []
    
    for _ in range(n_bootstrap):
        if stratify_by_tier:
            sample_indices = []
            for tier in merged['tier'].unique():
                tier_mask = merged['tier'] == tier
                tier_indices = merged[tier_mask].index.values
                if len(tier_indices) > 0:
                    sample_tier = rng.choice(
                        tier_indices,
                        size=len(tier_indices),
                        replace=True
                    )
                    sample_indices.extend(sample_tier)
        else:
            sample_indices = rng.choice(
                len(merged),
                size=len(merged),
                replace=True
            )
        
        sample = merged.iloc[sample_indices]
        delta_mape = (sample['rel_error_pct_curr'].mean() - 
                      sample['rel_error_pct_base'].mean())
        delta_mae = (sample['abs_error_curr'].mean() -
                     sample['abs_error_base'].mean())
        delta_mapes.append(delta_mape)
        delta_maes.append(delta_mae)
    
    delta_mapes = np.array(delta_mapes)
    delta_maes = np.array(delta_maes)
    
    # Compute CIs
    ci_lower_mape = np.percentile(delta_mapes, alpha/2 * 100)
    ci_upper_mape = np.percentile(delta_mapes, (1 - alpha/2) * 100)
    ci_lower_mae = np.percentile(delta_maes, alpha/2 * 100)
    ci_upper_mae = np.percentile(delta_maes, (1 - alpha/2) * 100)
    
    # Check if excludes zero
    mape_excludes_zero = (ci_upper_mape < 0) or (ci_lower_mape > 0)
    mae_excludes_zero = (ci_upper_mae < 0) or (ci_lower_mae > 0)
    
    # Approximate p-values (fraction of bootstrap samples ≥ 0)
    p_value_mape = (delta_mapes >= 0).sum() / len(delta_mapes)
    
    return {
        'delta_mape_mean': float(delta_mapes.mean()),
        'delta_mape_std': float(delta_mapes.std()),
        'delta_mape_ci_lower': float(ci_lower_mape),
        'delta_mape_ci_upper': float(ci_upper_mape),
        'delta_mape_excludes_zero': bool(mape_excludes_zero),
        'delta_mape_p_value': float(p_value_mape),
        
        'delta_mae_mean': float(delta_maes.mean()),
        'delta_mae_ci_lower': float(ci_lower_mae),
        'delta_mae_ci_upper': float(ci_upper_mae),
        'delta_mae_excludes_zero': bool(mae_excludes_zero),
        
        'n_bootstrap': n_bootstrap,
        'alpha': alpha,
        'n_materials': len(merged),
    }

