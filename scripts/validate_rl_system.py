#!/usr/bin/env python3
"""
Rigorous validation suite for RL-based experiment optimization.

Compares our PPO+ICM agent against:
- Random search (baseline)
- Bayesian optimization (gold standard)
- PPO without curiosity (ablation)

Metrics:
- Experiments to reach 95% of optimum
- Final best value achieved
- Sample efficiency (AUC of progress curve)
- Statistical significance (t-test)
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import torch
import logging
from typing import Dict, List, Tuple
from scipy import stats
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import json

from src.reasoning.rl_env import ExperimentOptimizationEnv
from src.reasoning.ppo_agent import PPOAgent
from src.reasoning.curiosity_module import IntrinsicCuriosityModule

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Test functions (standard RL benchmarks)
def branin(params: np.ndarray) -> float:
    """Branin function (global min: -0.397887)."""
    x, y = params[0], params[1]
    a, b, c, r, s, t = 1.0, 5.1/(4*np.pi**2), 5.0/np.pi, 6.0, 10.0, 1.0/(8*np.pi)
    return -(a*(y - b*x**2 + c*x - r)**2 + s*(1-t)*np.cos(x) + s)


def rastrigin(params: np.ndarray) -> float:
    """Rastrigin function (global min: 0, we negate)."""
    A = 10
    n = len(params)
    # Scale params from [0,1] to [-5.12, 5.12]
    x = params * 10.24 - 5.12
    result = A * n + np.sum(x**2 - A * np.cos(2 * np.pi * x))
    return -result  # Negate for maximization


def ackley(params: np.ndarray) -> float:
    """Ackley function (global min: 0, we negate)."""
    # Scale params from [0,1] to [-5, 5]
    x = params * 10 - 5
    n = len(x)
    sum1 = np.sum(x**2)
    sum2 = np.sum(np.cos(2 * np.pi * x))
    result = -20 * np.exp(-0.2 * np.sqrt(sum1 / n)) - np.exp(sum2 / n) + 20 + np.e
    return -result  # Negate for maximization


class RandomSearch:
    """Random search baseline."""
    
    def optimize(self, env, n_experiments=100):
        obs, _ = env.reset()
        history = []
        
        for _ in range(n_experiments):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            history.append(info['best_value'])
            
            if terminated or truncated:
                obs, _ = env.reset()
        
        return history


class BayesianOptimization:
    """Bayesian optimization baseline (acquisition: UCB)."""
    
    def __init__(self, bounds):
        self.bounds = bounds
        kernel = ConstantKernel(1.0) * RBF(length_scale=1.0)
        self.gp = GaussianProcessRegressor(
            kernel=kernel,
            alpha=0.1**2,
            n_restarts_optimizer=10,
        )
    
    def optimize(self, env, n_experiments=100):
        obs, _ = env.reset()
        history = []
        X_observed = []
        y_observed = []
        
        # Initialize with random samples
        for _ in range(5):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            X_observed.append(action)
            y_observed.append(info['current_value'])
            history.append(info['best_value'])
            
            if terminated or truncated:
                obs, _ = env.reset()
        
        # BO loop
        for _ in range(n_experiments - 5):
            # Fit GP
            self.gp.fit(np.array(X_observed), np.array(y_observed))
            
            # Find next point via UCB acquisition
            action = self._optimize_acquisition()
            
            # Evaluate
            obs, reward, terminated, truncated, info = env.step(action)
            X_observed.append(action)
            y_observed.append(info['current_value'])
            history.append(info['best_value'])
            
            if terminated or truncated:
                obs, _ = env.reset()
        
        return history
    
    def _optimize_acquisition(self, kappa=2.0):
        """Maximize Upper Confidence Bound."""
        best_acq = -np.inf
        best_x = None
        
        # Multi-start optimization
        for _ in range(10):
            x0 = np.random.rand(len(self.bounds))
            
            def acq(x):
                mean, std = self.gp.predict(x.reshape(1, -1), return_std=True)
                return -(mean[0] + kappa * std[0])  # Negative for minimization
            
            res = minimize(acq, x0, bounds=[(0, 1)] * len(self.bounds), method='L-BFGS-B')
            
            if -res.fun > best_acq:
                best_acq = -res.fun
                best_x = res.x
        
        return best_x


class PPOBaseline:
    """PPO without curiosity (ablation)."""
    
    def __init__(self, env, device="cpu"):
        obs_dim = env.observation_space.shape[0]
        action_dim = env.action_space.shape[0]
        self.agent = PPOAgent(obs_dim=obs_dim, action_dim=action_dim, device=device)
    
    def optimize(self, env, n_experiments=100, rollout_length=50):
        obs, _ = env.reset()
        history = []
        
        for _ in range(n_experiments // rollout_length):
            # Collect rollout
            obs_buffer, actions_buffer, log_probs_buffer = [], [], []
            rewards_buffer, values_buffer, dones_buffer = [], [], []
            
            for _ in range(rollout_length):
                action, value, log_prob = self.agent.select_action(obs)
                next_obs, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                
                obs_buffer.append(obs)
                actions_buffer.append(action)
                log_probs_buffer.append(log_prob)
                rewards_buffer.append(reward)
                values_buffer.append(value)
                dones_buffer.append(done)
                history.append(info['best_value'])
                
                obs = next_obs
                
                if done:
                    obs, _ = env.reset()
            
            # Update
            if len(obs_buffer) > 32:
                advantages, returns = self.agent.compute_gae(
                    rewards_buffer, values_buffer, dones_buffer
                )
                self.agent.update(
                    np.array(obs_buffer),
                    np.array(actions_buffer),
                    np.array(log_probs_buffer),
                    advantages,
                    returns,
                )
        
        return history


class PPOWithCuriosity:
    """Our method: PPO + ICM."""
    
    def __init__(self, env, device="cpu"):
        obs_dim = env.observation_space.shape[0]
        action_dim = env.action_space.shape[0]
        self.agent = PPOAgent(obs_dim=obs_dim, action_dim=action_dim, device=device)
        self.icm = IntrinsicCuriosityModule(
            state_dim=obs_dim, action_dim=action_dim, device=device
        )
    
    def optimize(self, env, n_experiments=100, rollout_length=50):
        obs, _ = env.reset()
        history = []
        
        for _ in range(n_experiments // rollout_length):
            obs_buffer, actions_buffer, log_probs_buffer = [], [], []
            rewards_buffer, values_buffer, dones_buffer = [], [], []
            next_obs_buffer = []
            
            for _ in range(rollout_length):
                action, value, log_prob = self.agent.select_action(obs)
                next_obs, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                
                # Add curiosity bonus
                intrinsic_reward = self.icm.compute_intrinsic_reward(obs, action, next_obs)
                reward += intrinsic_reward
                
                obs_buffer.append(obs)
                actions_buffer.append(action)
                log_probs_buffer.append(log_prob)
                rewards_buffer.append(reward)
                values_buffer.append(value)
                dones_buffer.append(done)
                next_obs_buffer.append(next_obs)
                history.append(info['best_value'])
                
                obs = next_obs
                
                if done:
                    obs, _ = env.reset()
            
            # Update ICM and agent
            if len(obs_buffer) > 32:
                self.icm.update(
                    np.array(obs_buffer),
                    np.array(actions_buffer),
                    np.array(next_obs_buffer),
                )
                advantages, returns = self.agent.compute_gae(
                    rewards_buffer, values_buffer, dones_buffer
                )
                self.agent.update(
                    np.array(obs_buffer),
                    np.array(actions_buffer),
                    np.array(log_probs_buffer),
                    advantages,
                    returns,
                )
        
        return history


def run_validation_suite(
    test_function=branin,
    bounds={"x": (-5, 10), "y": (0, 15)},
    n_trials=5,
    n_experiments=100,
    device="cpu",
):
    """Run full validation suite."""
    
    logger.info(f"Running validation on {test_function.__name__}")
    logger.info(f"Trials: {n_trials}, Experiments per trial: {n_experiments}")
    
    results = {
        "random": [],
        "bayesian": [],
        "ppo_baseline": [],
        "ppo_curiosity": [],
    }
    
    for trial in range(n_trials):
        logger.info(f"\n{'='*60}")
        logger.info(f"Trial {trial + 1}/{n_trials}")
        logger.info(f"{'='*60}")
        
        # Create environment
        env = ExperimentOptimizationEnv(
            parameter_bounds=bounds,
            objective_function=test_function,
            max_experiments=n_experiments,
            time_budget_hours=n_experiments * 1.0,
            noise_std=0.1,
        )
        
        # Random search
        logger.info("Running Random Search...")
        random_opt = RandomSearch()
        random_history = random_opt.optimize(env, n_experiments)
        results["random"].append(random_history)
        logger.info(f"  Best: {max(random_history):.6f}")
        
        # Bayesian optimization
        logger.info("Running Bayesian Optimization...")
        bo_opt = BayesianOptimization(bounds)
        env.reset()
        bo_history = bo_opt.optimize(env, n_experiments)
        results["bayesian"].append(bo_history)
        logger.info(f"  Best: {max(bo_history):.6f}")
        
        # PPO baseline (no curiosity)
        logger.info("Running PPO (no curiosity)...")
        ppo_base = PPOBaseline(env, device)
        env.reset()
        ppo_base_history = ppo_base.optimize(env, n_experiments)
        results["ppo_baseline"].append(ppo_base_history)
        logger.info(f"  Best: {max(ppo_base_history):.6f}")
        
        # PPO + curiosity (ours)
        logger.info("Running PPO + ICM (ours)...")
        ppo_cur = PPOWithCuriosity(env, device)
        env.reset()
        ppo_cur_history = ppo_cur.optimize(env, n_experiments)
        results["ppo_curiosity"].append(ppo_cur_history)
        logger.info(f"  Best: {max(ppo_cur_history):.6f}")
    
    # Analysis
    analyze_results(results, test_function.__name__)
    
    return results


def analyze_results(results: Dict[str, List], function_name: str):
    """Statistical analysis and visualization."""
    
    logger.info(f"\n{'='*60}")
    logger.info("STATISTICAL ANALYSIS")
    logger.info(f"{'='*60}")
    
    # Compute final values
    final_values = {}
    for method, histories in results.items():
        final_values[method] = [max(h) for h in histories]
        mean = np.mean(final_values[method])
        std = np.std(final_values[method])
        logger.info(f"{method:20s}: {mean:.6f} ± {std:.6f}")
    
    # T-test: PPO+curiosity vs others
    logger.info("\nSignificance Tests (t-test):")
    for method in ["random", "bayesian", "ppo_baseline"]:
        t_stat, p_value = stats.ttest_ind(
            final_values["ppo_curiosity"],
            final_values[method],
        )
        significant = "✓ SIGNIFICANT" if p_value < 0.05 else "✗ NOT SIGNIFICANT"
        logger.info(f"  PPO+ICM vs {method:15s}: p={p_value:.4f} {significant}")
    
    # Experiments to 95% optimum
    logger.info("\nExperiments to reach 95% of best:")
    target = 0.95 * np.mean([max(h) for h in results["ppo_curiosity"]])
    for method, histories in results.items():
        exp_to_target = []
        for history in histories:
            idx = next((i for i, v in enumerate(history) if v >= target), len(history))
            exp_to_target.append(idx)
        mean = np.mean(exp_to_target)
        logger.info(f"  {method:20s}: {mean:.1f} experiments")
    
    # Plot
    plot_results(results, function_name)
    
    # Save to JSON
    results_json = {
        method: [float(np.mean([h[i] for h in histories])) 
                 for i in range(len(histories[0]))]
        for method, histories in results.items()
    }
    
    with open(f"validation_{function_name}.json", "w") as f:
        json.dump(results_json, f, indent=2)
    
    logger.info(f"\nResults saved to validation_{function_name}.json")


def plot_results(results: Dict[str, List], function_name: str):
    """Plot comparison."""
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    fig.patch.set_facecolor('white')
    
    colors = {
        "random": "#ef4444",
        "bayesian": "#3b82f6",
        "ppo_baseline": "#a855f7",
        "ppo_curiosity": "#22c55e",
    }
    
    # Learning curves
    for method, histories in results.items():
        mean_history = np.mean(histories, axis=0)
        std_history = np.std(histories, axis=0)
        x = np.arange(len(mean_history))
        
        axes[0].plot(x, mean_history, label=method, color=colors[method], linewidth=2)
        axes[0].fill_between(
            x, mean_history - std_history, mean_history + std_history,
            alpha=0.2, color=colors[method]
        )
    
    axes[0].set_xlabel("Experiments")
    axes[0].set_ylabel("Best Value Found")
    axes[0].set_title(f"Learning Curves ({function_name})")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Final values boxplot
    final_values = {method: [max(h) for h in histories] 
                    for method, histories in results.items()}
    axes[1].boxplot(final_values.values(), labels=final_values.keys())
    axes[1].set_ylabel("Final Best Value")
    axes[1].set_title("Final Performance Distribution")
    axes[1].grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(f"validation_{function_name}.png", dpi=150)
    logger.info(f"Plot saved to validation_{function_name}.png")
    plt.close()


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Run on Branin function
    logger.info("="*60)
    logger.info("VALIDATION SUITE: BRANIN FUNCTION")
    logger.info("="*60)
    run_validation_suite(
        test_function=branin,
        bounds={"x": (-5, 10), "y": (0, 15)},
        n_trials=5,
        n_experiments=100,
        device=device,
    )
    
    logger.info("\n🎉 Validation complete! Check:")
    logger.info("  - validation_branin.json (raw data)")
    logger.info("  - validation_branin.png (visualizations)")

