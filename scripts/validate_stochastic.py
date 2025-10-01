#!/usr/bin/env python3
"""
Stochastic validation suite - Oct 2025 best practices.

Tests RL vs BO in noisy environments to better simulate real experiments.
Based on: https://towardsdatascience.com/best-practices-for-reinforcement-learning-1cf8c2d77b66/
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import torch
import logging
from typing import Dict, List
from scipy import stats
import matplotlib.pyplot as plt
import json
from datetime import datetime

from scripts.validate_rl_system import (
    RandomSearch, BayesianOptimization, PPOBaseline, 
    PPOWithCuriosity, branin
)
from src.reasoning.rl_env import ExperimentOptimizationEnv

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def noisy_branin(params: np.ndarray, noise_std: float = 0.1) -> float:
    """
    Branin function with Gaussian noise to simulate real experiments.
    
    Real experiments have measurement noise, instrument drift, and 
    environmental variability. This tests robustness.
    """
    clean_value = branin(params)
    noise = np.random.normal(0, noise_std)
    return clean_value + noise


class NoisyObjectiveWrapper:
    """Wrapper to add noise to any objective function."""
    
    def __init__(self, objective_fn, noise_std: float):
        self.objective_fn = objective_fn
        self.noise_std = noise_std
    
    def __call__(self, params: np.ndarray) -> float:
        clean_value = self.objective_fn(params)
        noise = np.random.normal(0, self.noise_std)
        return clean_value + noise


def run_stochastic_validation(
    noise_levels: List[float] = [0.0, 0.1, 0.5, 1.0, 2.0],
    n_trials: int = 10,  # Increased from 5 per Oct 2025 standards
    n_experiments: int = 100,
    device: str = "cpu"
):
    """
    Run comprehensive stochastic validation.
    
    Oct 2025 Best Practice: Test under realistic noise conditions.
    Hypothesis: RL may be more robust to noise than BO.
    """
    
    bounds = {"x": (-5, 10), "y": (0, 15)}
    results = {
        "config": {
            "noise_levels": noise_levels,
            "n_trials": n_trials,
            "n_experiments": n_experiments,
            "date": datetime.now().isoformat()
        },
        "results_by_noise": {}
    }
    
    for noise_std in noise_levels:
        logger.info(f"\n{'='*60}")
        logger.info(f"NOISE LEVEL: {noise_std}")
        logger.info(f"{'='*60}")
        
        noise_results = {
            "random": [],
            "bayesian": [],
            "ppo_baseline": [],
            "ppo_curiosity": []
        }
        
        for trial in range(n_trials):
            logger.info(f"\nTrial {trial + 1}/{n_trials} (noise={noise_std})")
            
            # Create noisy objective
            noisy_obj = NoisyObjectiveWrapper(branin, noise_std)
            
            # Create environment with noisy objective
            env = ExperimentOptimizationEnv(
                parameter_bounds=bounds,
                objective_function=noisy_obj
            )
            
            # Test each method
            methods = {
                "random": RandomSearch(),
                "bayesian": BayesianOptimization(bounds),
                "ppo_baseline": PPOBaseline(env, device),
                "ppo_curiosity": PPOWithCuriosity(env, device)
            }
            
            for method_name, method_obj in methods.items():
                logger.info(f"  Running {method_name}...")
                history = method_obj.optimize(env, n_experiments)
                
                # Get final best value
                final_best = history[-1] if history else float('inf')
                noise_results[method_name].append(final_best)
                
                logger.info(f"    Final best: {final_best:.6f}")
                
                # Reset environment for next method
                env.close()
                env = ExperimentOptimizationEnv(
                    parameter_bounds=bounds,
                    objective_function=noisy_obj
                )
        
        # Compute statistics for this noise level (convert all numpy types to Python floats)
        results["results_by_noise"][noise_std] = {
            "mean": {name: float(np.mean(vals)) for name, vals in noise_results.items()},
            "std": {name: float(np.std(vals)) for name, vals in noise_results.items()},
            "median": {name: float(np.median(vals)) for name, vals in noise_results.items()},
            "raw": {name: [float(v) for v in vals] for name, vals in noise_results.items()}
        }
        
        # Statistical tests
        logger.info(f"\nStatistical Analysis (noise={noise_std}):")
        ppo_icm = np.array(noise_results["ppo_curiosity"])
        bayesian = np.array(noise_results["bayesian"])
        
        t_stat, p_value = stats.ttest_ind(ppo_icm, bayesian)
        logger.info(f"  PPO+ICM vs Bayesian: t={t_stat:.3f}, p={p_value:.4f}")
        
        if p_value < 0.05:
            winner = "PPO+ICM" if np.mean(ppo_icm) < np.mean(bayesian) else "Bayesian"
            logger.info(f"  ✓ SIGNIFICANT DIFFERENCE: {winner} wins")
        else:
            logger.info(f"  ✗ No significant difference")
    
    # Save results
    results_path = Path(".") / f"validation_stochastic_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    logger.info(f"\nResults saved to {results_path}")
    
    # Plot results
    plot_stochastic_results(results)
    
    return results


def plot_stochastic_results(results: Dict):
    """
    Plot how performance degrades with noise.
    
    Key question: Does RL degrade more gracefully than BO under noise?
    """
    noise_levels = list(results["results_by_noise"].keys())
    methods = ["random", "bayesian", "ppo_baseline", "ppo_curiosity"]
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Performance Under Noise (Oct 2025 Validation)", fontsize=16)
    
    # Plot 1: Mean performance vs noise
    ax = axes[0, 0]
    for method in methods:
        means = [results["results_by_noise"][noise]["mean"][method] 
                 for noise in noise_levels]
        ax.plot(noise_levels, means, marker='o', label=method, linewidth=2)
    ax.set_xlabel("Noise Level (std)")
    ax.set_ylabel("Final Best Value (lower is better)")
    ax.set_title("Mean Performance vs Noise")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Std dev vs noise (robustness)
    ax = axes[0, 1]
    for method in methods:
        stds = [results["results_by_noise"][noise]["std"][method] 
                for noise in noise_levels]
        ax.plot(noise_levels, stds, marker='o', label=method, linewidth=2)
    ax.set_xlabel("Noise Level (std)")
    ax.set_ylabel("Std Dev of Final Value")
    ax.set_title("Robustness vs Noise (lower is better)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 3: Relative performance to Bayesian
    ax = axes[1, 0]
    for method in ["random", "ppo_baseline", "ppo_curiosity"]:
        relative = []
        for noise in noise_levels:
            method_mean = results["results_by_noise"][noise]["mean"][method]
            bayesian_mean = results["results_by_noise"][noise]["mean"]["bayesian"]
            relative.append(method_mean / bayesian_mean)  # <1 = better, >1 = worse
        ax.plot(noise_levels, relative, marker='o', label=method, linewidth=2)
    ax.axhline(y=1.0, color='k', linestyle='--', label='Bayesian baseline')
    ax.set_xlabel("Noise Level (std)")
    ax.set_ylabel("Relative Performance (Bayesian = 1.0)")
    ax.set_title("Performance Relative to Bayesian")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 4: Winner at each noise level
    ax = axes[1, 1]
    winners = []
    for noise in noise_levels:
        means = {m: results["results_by_noise"][noise]["mean"][m] for m in methods}
        winner = min(means, key=means.get)
        winners.append(winner)
    
    # Count wins
    from collections import Counter
    win_counts = Counter(winners)
    ax.bar(win_counts.keys(), win_counts.values())
    ax.set_xlabel("Method")
    ax.set_ylabel("Number of Wins")
    ax.set_title("Winner at Each Noise Level")
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plot_path = Path(".") / f"stochastic_validation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
    plt.savefig(plot_path, dpi=300)
    logger.info(f"Plot saved to {plot_path}")


def main():
    """Run stochastic validation per Oct 2025 best practices."""
    logger.info("="*60)
    logger.info("STOCHASTIC VALIDATION - OCT 2025 BEST PRACTICES")
    logger.info("="*60)
    logger.info("\nHypothesis: RL may be more robust to noise than BO")
    logger.info("Based on: https://towardsdatascience.com/best-practices-for-reinforcement-learning-1cf8c2d77b66/")
    logger.info("\n")
    
    results = run_stochastic_validation(
        noise_levels=[0.0, 0.1, 0.5, 1.0, 2.0],
        n_trials=10,  # Oct 2025 standard: minimum 10 trials
        n_experiments=100,
        device="cpu"
    )
    
    logger.info("\n" + "="*60)
    logger.info("VALIDATION COMPLETE")
    logger.info("="*60)
    logger.info("\nKey Findings:")
    
    # Summarize results
    for noise in results["results_by_noise"].keys():
        means = results["results_by_noise"][noise]["mean"]
        winner = min(means, key=means.get)
        logger.info(f"\nNoise={noise}:")
        logger.info(f"  Winner: {winner}")
        logger.info(f"  Bayesian: {means['bayesian']:.4f}")
        logger.info(f"  PPO+ICM: {means['ppo_curiosity']:.4f}")
        logger.info(f"  Ratio: {means['ppo_curiosity']/means['bayesian']:.2f}×")


if __name__ == "__main__":
    main()

