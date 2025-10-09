#!/usr/bin/env python3
"""
Conformal Expected Improvement (Conformal-EI)

Novel contribution: EI acquisition weighted by LOCALLY ADAPTIVE conformal credibility.
Fixes v1 bug: credibility now varies with x (local difficulty).

Key innovation:
- Scale nonconformity scores by local difficulty s(x) (posterior std or k-NN density)
- Calibrate on scaled scores → x-dependent intervals
- Credibility(x) = 1 / (1 + half_width(x) / median_half_width)

Citation alignment:
- Split conformal prediction (Vovk et al., 2005; Shafer & Vovk, 2008)
- Locally adaptive conformal (Romano et al., 2019; Lei et al., 2018)
- Conformal for scientific ML (Stanton et al., 2022; Cognac et al., 2023)
- Active learning + calibration (Cocheteux et al., 2025; Widmann et al., 2021)

Author: GOATnote Autonomous Research Lab Initiative
Contact: b@thegoatnote.com
License: MIT
"""

import numpy as np
import torch
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import mean_squared_error
from scipy import stats
import json
import matplotlib.pyplot as plt
import seaborn as sns
import sys
import logging

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from phase10_gp_active_learning.data.uci_loader import load_uci_superconductor
from phase10_gp_active_learning.models.dkl_model import create_dkl_model
from phase10_gp_active_learning.models.botorch_dkl import BoTorchDKL
from botorch.acquisition import ExpectedImprovement

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

sns.set_style('whitegrid')
plt.rcParams['figure.dpi'] = 300


class LocallyAdaptiveConformal:
    """
    Locally adaptive conformal prediction for regression.
    
    Key innovation: Scales nonconformity scores by local difficulty s(x).
    
    - If model provides posterior std → use it (heteroscedastic GP/DKL)
    - Otherwise → use k-NN distance as difficulty proxy
    
    Result: Intervals are μ(x) ± q * s(x) where s(x) varies with x.
    """
    
    def __init__(self, alpha=0.1, k=25, use_model_std=True):
        """
        Args:
            alpha: Miscoverage rate (0.1 = 90% coverage)
            k: Number of neighbors for k-NN fallback
            use_model_std: If True, use posterior std (if available)
        """
        self.alpha = alpha
        self.k = k
        self.use_model_std = use_model_std
        self.q = None
        self.nn = None
        self.X_calib = None
        
    def _local_scale(self, X, model_std=None):
        """
        Compute local difficulty s(x).
        
        Args:
            X: Features (N, D)
            model_std: Optional posterior std from GP/DKL (N,)
            
        Returns:
            s: Local scale (N,) - higher = more uncertain region
        """
        if self.use_model_std and model_std is not None:
            # Use posterior std directly (best for GP/DKL)
            return model_std
        
        # Fallback: k-NN distance as heteroscedasticity proxy
        if self.nn is None:
            raise ValueError("Must calibrate first (no kNN fitted)")
        
        dists, _ = self.nn.kneighbors(X, n_neighbors=self.k, return_distance=True)
        # Mean distance to k neighbors (avoid zeros)
        return dists.mean(axis=1) + 1e-6
    
    def calibrate(self, X_calib, y_calib, mu_calib, std_calib=None):
        """
        Calibrate on validation set.
        
        Args:
            X_calib: Calibration features (N, D)
            y_calib: True targets (N,)
            mu_calib: Model predictions (N,)
            std_calib: Optional posterior std (N,)
        """
        # Fit k-NN on calibration features (for local scale)
        self.nn = NearestNeighbors(n_neighbors=self.k).fit(X_calib)
        self.X_calib = X_calib
        
        # Compute local scale
        s_calib = self._local_scale(X_calib, std_calib)
        
        # Scaled nonconformity scores
        scores = np.abs(y_calib - mu_calib) / s_calib
        
        # Quantile at 1-alpha level
        n = len(scores)
        q_level = np.ceil((n + 1) * (1 - self.alpha)) / n
        self.q = np.quantile(scores, q_level)
        
        logger.info(f"Locally adaptive conformal: α={self.alpha}, q={self.q:.3f}")
        logger.info(f"  Scale range: [{s_calib.min():.3f}, {s_calib.max():.3f}]")
        
    def intervals(self, X, mu, std=None):
        """
        Generate conformal prediction intervals (locally adaptive).
        
        Args:
            X: Features (N, D)
            mu: Model predictions (N,)
            std: Optional posterior std (N,)
            
        Returns:
            lower, upper, half_width: PI bounds and half-widths (N,)
        """
        if self.q is None:
            raise ValueError("Must calibrate first")
        
        # Local scale
        s = self._local_scale(X, std)
        
        # Interval half-width (varies with x!)
        r = self.q * s
        
        return mu - r, mu + r, r
    
    def credibility(self, half_width):
        """
        Map interval half-widths to credibility scores ∈ [0, 1].
        
        Narrower intervals → higher credibility.
        Normalized by median half-width for scale-invariance.
        
        Args:
            half_width: Interval half-widths (N,)
            
        Returns:
            credibility: Scores ∈ [0, 1] (N,)
        """
        m = np.median(half_width)
        c = 1.0 / (1.0 + (half_width / (m + 1e-8)))
        return np.clip(c, 0.0, 1.0)
    
    def coverage(self, X, mu, y_true, std=None, nominal=0.90):
        """
        Compute empirical coverage at given nominal level.
        
        Args:
            X: Features (N, D)
            mu: Model predictions (N,)
            y_true: True targets (N,)
            std: Optional posterior std (N,)
            nominal: Target coverage (e.g., 0.90)
            
        Returns:
            coverage: Empirical coverage
        """
        lower, upper, _ = self.intervals(X, mu, std)
        covered = (y_true >= lower) & (y_true <= upper)
        return covered.mean()


def ece_regression(y_true, y_pred, n_bins=15):
    """
    Expected Calibration Error for regression (binned by prediction value).
    
    Crude but interpretable: bins by y_pred, compares bin MAE to global MAE.
    """
    bins = np.linspace(np.min(y_pred), np.max(y_pred), n_bins + 1)
    idx = np.digitize(y_pred, bins) - 1
    idx = np.clip(idx, 0, n_bins - 1)  # Handle edge case
    
    e = np.abs(y_true - y_pred)
    global_e = e.mean()
    
    ece = 0.0
    for b in range(n_bins):
        mask = (idx == b)
        if np.any(mask):
            e_bin = e[mask].mean()
            w_bin = mask.mean()
            ece += w_bin * np.abs(e_bin - global_e)
    
    return float(ece)


def conformal_ei_acquisition(
    model,
    X_candidates: torch.Tensor,
    best_f: float,
    X_candidates_np: np.ndarray,  # For kNN
    conformal_predictor: LocallyAdaptiveConformal,
    y_scaler: StandardScaler,
    credibility_weight: float = 1.0
):
    """
    Conformal Expected Improvement acquisition function (LOCALLY ADAPTIVE).
    
    CEI(x) = EI(x) * (1 + w * credibility(x))
    
    where credibility(x) varies per candidate (not constant!).
    
    Args:
        model: BoTorch-compatible model
        X_candidates: Candidate points (N, 1, D) [scaled, for model]
        best_f: Current best observation (scaled)
        X_candidates_np: Candidates (N, D) [scaled, for kNN]
        conformal_predictor: Calibrated locally adaptive conformal predictor
        y_scaler: For unscaling predictions
        credibility_weight: Weight for credibility term (0 = vanilla EI, 1 = full)
        
    Returns:
        acquisition_values: Conformal-EI scores (N,)
    """
    # Standard EI (scaled space)
    acq_ei = ExpectedImprovement(model=model, best_f=best_f)
    
    with torch.no_grad():
        ei_values = acq_ei(X_candidates).cpu().numpy()
        
        # Get predictions + posterior std
        posterior = model.posterior(X_candidates)
        mu_scaled = posterior.mean.cpu().numpy().ravel()
        std_scaled = posterior.variance.clamp_min(1e-12).sqrt().cpu().numpy().ravel()
    
    # Unscale to target space (K) for conformal intervals
    mu = y_scaler.inverse_transform(mu_scaled.reshape(-1, 1)).ravel()
    
    # Generate locally adaptive intervals
    lower, upper, half_width = conformal_predictor.intervals(X_candidates_np, mu, std_scaled)
    
    # Credibility (varies with x!)
    cred = conformal_predictor.credibility(half_width)
    
    # Weighted combination
    cei = ei_values * (1.0 + credibility_weight * cred)
    
    return torch.tensor(cei, dtype=torch.float64), cred, half_width


def run_active_learning(
    X_pool: np.ndarray,
    y_pool: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    method: str = "conformal_ei",  # "conformal_ei" or "vanilla_ei"
    initial_samples: int = 100,
    num_rounds: int = 10,  # Reduced for novelty experiment
    batch_size: int = 1,
    alpha: float = 0.1,  # Conformal miscoverage rate (90% coverage)
    credibility_weight: float = 1.0,
    random_seed: int = 42
):
    """
    Run active learning with Conformal-EI or vanilla EI.
    
    Returns:
        metrics: Dict with RMSE, coverage@80/90, PI width, ECE, regret
    """
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    
    logger.info(f"🚀 Running {method} (seed={random_seed})")
    
    # Split pool into train/calibration (20% calib)
    n_pool = len(X_pool)
    indices = np.random.permutation(n_pool)
    n_calib = int(0.2 * n_pool)
    calib_indices = indices[:n_calib]
    train_indices = indices[n_calib:]
    
    X_calib = X_pool[calib_indices]
    y_calib = y_pool[calib_indices]
    X_train_pool = X_pool[train_indices]
    y_train_pool = y_pool[train_indices]
    
    # Initialize labeled set from train pool
    initial_idx = np.random.choice(len(X_train_pool), initial_samples, replace=False)
    X_labeled = X_train_pool[initial_idx]
    y_labeled = y_train_pool[initial_idx]
    
    remaining_idx = np.setdiff1d(np.arange(len(X_train_pool)), initial_idx)
    X_unlabeled = X_train_pool[remaining_idx]
    y_unlabeled = y_train_pool[remaining_idx]
    
    # Scalers
    X_scaler = StandardScaler()
    y_scaler = StandardScaler()
    
    # Metrics
    rmse_history = []
    coverage_80_history = []
    coverage_90_history = []
    pi_width_history = []
    ece_history = []
    regret_history = []
    
    for round_idx in range(num_rounds):
        # Scale data
        X_labeled_scaled = X_scaler.fit_transform(X_labeled)
        y_labeled_scaled = y_scaler.fit_transform(y_labeled.reshape(-1, 1)).ravel()
        X_test_scaled = X_scaler.transform(X_test)
        X_calib_scaled = X_scaler.transform(X_calib)
        X_unlabeled_scaled = X_scaler.transform(X_unlabeled) if len(X_unlabeled) > 0 else None
        
        # Train DKL
        dkl = create_dkl_model(
            X_labeled_scaled, y_labeled_scaled,
            input_dim=X_labeled_scaled.shape[1],
            n_epochs=20,
            verbose=False
        )
        model = BoTorchDKL(dkl)
        
        # Evaluate on test set
        X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float64)
        with torch.no_grad():
            posterior_test = model.posterior(X_test_tensor)
            mu_test_scaled = posterior_test.mean.cpu().numpy().ravel()
            std_test_scaled = posterior_test.variance.clamp_min(1e-12).sqrt().cpu().numpy().ravel()
        
        mu_test = y_scaler.inverse_transform(mu_test_scaled.reshape(-1, 1)).ravel()
        rmse = np.sqrt(mean_squared_error(y_test, mu_test))
        rmse_history.append(rmse)
        
        # ECE
        ece = ece_regression(y_test, mu_test, n_bins=15)
        ece_history.append(ece)
        
        # Conformal calibration (locally adaptive)
        if method == "conformal_ei":
            X_calib_tensor = torch.tensor(X_calib_scaled, dtype=torch.float64)
            with torch.no_grad():
                posterior_calib = model.posterior(X_calib_tensor)
                mu_calib_scaled = posterior_calib.mean.cpu().numpy().ravel()
                std_calib_scaled = posterior_calib.variance.clamp_min(1e-12).sqrt().cpu().numpy().ravel()
            
            mu_calib = y_scaler.inverse_transform(mu_calib_scaled.reshape(-1, 1)).ravel()
            
            # Locally adaptive conformal (alpha=0.1 → 90% coverage)
            conformal = LocallyAdaptiveConformal(alpha=alpha, k=25, use_model_std=True)
            conformal.calibrate(X_calib_scaled, y_calib, mu_calib, std_calib_scaled)
            
            # Measure coverage@80 and coverage@90 on test set
            conformal_80 = LocallyAdaptiveConformal(alpha=0.2, k=25, use_model_std=True)
            conformal_80.calibrate(X_calib_scaled, y_calib, mu_calib, std_calib_scaled)
            
            coverage_80 = conformal_80.coverage(X_test_scaled, mu_test, y_test, std_test_scaled, nominal=0.80)
            coverage_90 = conformal.coverage(X_test_scaled, mu_test, y_test, std_test_scaled, nominal=0.90)
            
            _, _, half_width_test = conformal.intervals(X_test_scaled, mu_test, std_test_scaled)
            pi_width = half_width_test.mean() * 2  # Full width
            
            coverage_80_history.append(coverage_80)
            coverage_90_history.append(coverage_90)
            pi_width_history.append(pi_width)
        else:
            coverage_80_history.append(np.nan)
            coverage_90_history.append(np.nan)
            pi_width_history.append(np.nan)
        
        logger.info(f"   Round {round_idx:2d}: n_labeled={len(X_labeled):4d}, RMSE={rmse:.2f} K, "
                   f"Cov@80={coverage_80_history[-1]:.3f if not np.isnan(coverage_80_history[-1]) else 'N/A'}, "
                   f"Cov@90={coverage_90_history[-1]:.3f if not np.isnan(coverage_90_history[-1]) else 'N/A'}, "
                   f"ECE={ece:.3f}")
        
        # Acquisition
        if round_idx < num_rounds - 1 and len(X_unlabeled) > 0:
            best_f_scaled = y_labeled_scaled.max()
            X_unlabeled_tensor = torch.tensor(X_unlabeled_scaled, dtype=torch.float64)
            
            if X_unlabeled_tensor.dim() == 2:
                X_unlabeled_tensor = X_unlabeled_tensor.unsqueeze(1)
            
            if method == "conformal_ei":
                acq_values, cred, half_width = conformal_ei_acquisition(
                    model, X_unlabeled_tensor, best_f_scaled,
                    X_unlabeled_scaled, conformal, y_scaler, credibility_weight
                )
            else:  # vanilla_ei
                acq = ExpectedImprovement(model=model, best_f=best_f_scaled)
                acq_values = acq(X_unlabeled_tensor).squeeze()
            
            best_idx = acq_values.argmax().item()
            
            # Compute oracle regret
            best_available = y_unlabeled.max()
            selected_value = y_unlabeled[best_idx]
            regret = best_available - selected_value
            regret_history.append(regret)
            
            # Update sets
            X_labeled = np.vstack([X_labeled, X_unlabeled[best_idx:best_idx+1]])
            y_labeled = np.append(y_labeled, y_unlabeled[best_idx])
            X_unlabeled = np.delete(X_unlabeled, best_idx, axis=0)
            y_unlabeled = np.delete(y_unlabeled, best_idx)
    
    return {
        'rmse_history': rmse_history,
        'coverage_80_history': coverage_80_history,
        'coverage_90_history': coverage_90_history,
        'pi_width_history': pi_width_history,
        'ece_history': ece_history,
        'regret_history': regret_history,
        'final_rmse': rmse_history[-1],
        'mean_coverage_80': np.nanmean(coverage_80_history),
        'mean_coverage_90': np.nanmean(coverage_90_history),
        'mean_pi_width': np.nanmean(pi_width_history),
        'mean_ece': np.mean(ece_history),
        'mean_regret': np.mean(regret_history)
    }


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--seeds', type=int, default=20, help='Number of seeds (≥20 for proper stats)')
    parser.add_argument('--rounds', type=int, default=10, help='AL rounds per seed')
    parser.add_argument('--output', type=Path, default=Path("experiments/novelty"), help='Output dir')
    args = parser.parse_args()
    
    logger.info("="*70)
    logger.info("CONFORMAL-EI NOVELTY EXPERIMENT (LOCALLY ADAPTIVE)")
    logger.info("="*70)
    logger.info(f"Seeds: {args.seeds}, Rounds: {args.rounds}")
    
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
    
    # Run experiments
    seeds = list(range(42, 42 + args.seeds))
    
    results = {
        'conformal_ei': [],
        'vanilla_ei': []
    }
    
    for method in ['conformal_ei', 'vanilla_ei']:
        logger.info(f"\n{'='*70}")
        logger.info(f"METHOD: {method.upper()}")
        logger.info(f"{'='*70}")
        
        for seed in seeds:
            metrics = run_active_learning(
                X_pool.copy(), y_pool.copy(),
                X_test.copy(), y_test.copy(),
                method=method,
                initial_samples=100,
                num_rounds=args.rounds,
                alpha=0.1,  # 90% coverage
                credibility_weight=1.0,  # Full conformal weighting
                random_seed=seed
            )
            results[method].append(metrics)
    
    # Aggregate statistics
    logger.info(f"\n{'='*70}")
    logger.info("RESULTS")
    logger.info(f"{'='*70}")
    
    for method in ['conformal_ei', 'vanilla_ei']:
        final_rmses = [r['final_rmse'] for r in results[method]]
        mean_coverages_80 = [r['mean_coverage_80'] for r in results[method]]
        mean_coverages_90 = [r['mean_coverage_90'] for r in results[method]]
        mean_pi_widths = [r['mean_pi_width'] for r in results[method]]
        mean_eces = [r['mean_ece'] for r in results[method]]
        mean_regrets = [r['mean_regret'] for r in results[method]]
        
        logger.info(f"\n{method.upper()}:")
        logger.info(f"  RMSE: {np.mean(final_rmses):.2f} ± {np.std(final_rmses):.2f} K")
        logger.info(f"  Coverage@80: {np.nanmean(mean_coverages_80):.3f} ± {np.nanstd(mean_coverages_80):.3f}")
        logger.info(f"  Coverage@90: {np.nanmean(mean_coverages_90):.3f} ± {np.nanstd(mean_coverages_90):.3f}")
        logger.info(f"  PI Width: {np.nanmean(mean_pi_widths):.1f} ± {np.nanstd(mean_pi_widths):.1f} K")
        logger.info(f"  ECE: {np.mean(mean_eces):.3f} ± {np.std(mean_eces):.3f}")
        logger.info(f"  Oracle Regret: {np.mean(mean_regrets):.2f} ± {np.std(mean_regrets):.2f} K")
    
    # Paired statistics (95% CI)
    conformal_rmses = np.array([r['final_rmse'] for r in results['conformal_ei']])
    vanilla_rmses = np.array([r['final_rmse'] for r in results['vanilla_ei']])
    conformal_regrets = np.array([r['mean_regret'] for r in results['conformal_ei']])
    vanilla_regrets = np.array([r['mean_regret'] for r in results['vanilla_ei']])
    
    t_stat, p_value = stats.ttest_rel(conformal_rmses, vanilla_rmses)
    delta_rmse = conformal_rmses - vanilla_rmses
    ci_rmse = stats.t.interval(0.95, len(delta_rmse) - 1, 
                               loc=delta_rmse.mean(), 
                               scale=stats.sem(delta_rmse))
    
    t_stat_regret, p_value_regret = stats.ttest_rel(conformal_regrets, vanilla_regrets)
    delta_regret = vanilla_regrets - conformal_regrets  # Reduction (positive = better)
    ci_regret = stats.t.interval(0.95, len(delta_regret) - 1,
                                 loc=delta_regret.mean(),
                                 scale=stats.sem(delta_regret))
    
    logger.info(f"\n{'='*70}")
    logger.info("PAIRED COMPARISON (Conformal-EI vs Vanilla-EI)")
    logger.info(f"{'='*70}")
    logger.info(f"ΔRMSE: {delta_rmse.mean():.2f} K (95% CI: [{ci_rmse[0]:.2f}, {ci_rmse[1]:.2f}]), p={p_value:.4f}")
    logger.info(f"Regret reduction: {delta_regret.mean():.2f} K (95% CI: [{ci_regret[0]:.2f}, {ci_regret[1]:.2f}]), p={p_value_regret:.4f}")
    
    if p_value_regret < 0.05:
        pct = delta_regret.mean() / vanilla_regrets.mean() * 100
        logger.info(f"✅ SIGNIFICANT: {pct:.1f}% regret reduction (p<0.05)")
    else:
        logger.info(f"⚠️  NOT SIGNIFICANT (p≥0.05)")
    
    # Save results
    args.output.mkdir(parents=True, exist_ok=True)
    
    with open(args.output / "conformal_ei_results.json", 'w') as f:
        json.dump({
            'experiment': 'Conformal-EI Novelty (Locally Adaptive)',
            'n_seeds': len(seeds),
            'n_rounds': args.rounds,
            'results': {
                'conformal_ei': {
                    'mean_rmse': float(np.mean(conformal_rmses)),
                    'std_rmse': float(np.std(conformal_rmses)),
                    'mean_coverage_80': float(np.nanmean([r['mean_coverage_80'] for r in results['conformal_ei']])),
                    'mean_coverage_90': float(np.nanmean([r['mean_coverage_90'] for r in results['conformal_ei']])),
                    'mean_pi_width': float(np.nanmean([r['mean_pi_width'] for r in results['conformal_ei']])),
                    'mean_ece': float(np.mean([r['mean_ece'] for r in results['conformal_ei']])),
                    'mean_regret': float(np.mean(conformal_regrets))
                },
                'vanilla_ei': {
                    'mean_rmse': float(np.mean(vanilla_rmses)),
                    'std_rmse': float(np.std(vanilla_rmses)),
                    'mean_ece': float(np.mean([r['mean_ece'] for r in results['vanilla_ei']])),
                    'mean_regret': float(np.mean(vanilla_regrets))
                }
            },
            'comparison': {
                'delta_rmse_mean': float(delta_rmse.mean()),
                'delta_rmse_ci_95': [float(ci_rmse[0]), float(ci_rmse[1])],
                'p_value_rmse': float(p_value),
                'regret_reduction_mean': float(delta_regret.mean()),
                'regret_reduction_ci_95': [float(ci_regret[0]), float(ci_regret[1])],
                'p_value_regret': float(p_value_regret),
                'significant': bool(p_value_regret < 0.05)
            }
        }, f, indent=2)
    
    logger.info(f"\n✅ Results saved to: {args.output / 'conformal_ei_results.json'}")
    logger.info("="*70)
    logger.info("✅ CONFORMAL-EI EXPERIMENT COMPLETE")
    logger.info("="*70)


if __name__ == '__main__':
    main()
