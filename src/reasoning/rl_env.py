"""
Reinforcement Learning Environment for Autonomous Experiments

Gym-compatible environment that wraps the Experiment OS for training
RL agents to actively learn optimal experiment sequences.

The agent learns to maximize Expected Information Gain (EIG) per unit time
while respecting safety constraints and budget limits.
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
from typing import Dict, Any, Optional, Tuple, List
import logging
from dataclasses import dataclass
from datetime import datetime

logger = logging.getLogger(__name__)


@dataclass
class ExperimentState:
    """Current state of the R&D campaign."""
    # Belief state (Gaussian Process mean/variance at key points)
    gp_mean: np.ndarray  # Shape: (n_grid_points,)
    gp_variance: np.ndarray  # Shape: (n_grid_points,)
    
    # Budget remaining
    experiments_remaining: int
    time_budget_hours: float
    
    # Historical information
    experiments_done: int
    best_observed_value: float
    
    # Constraints
    safety_violations: int


class ExperimentGym(gym.Env):
    """
    OpenAI Gym environment for experiment planning.
    
    **Observation Space**:
    - GP mean/variance (surrogate model of objective function)
    - Experiments remaining
    - Time budget
    - Best value found so far
    
    **Action Space**:
    - Continuous: Next experiment parameters (e.g., temperature, pressure, composition)
    - Discrete: Experiment type (XRD, NMR, UV-Vis, DFT)
    
    **Reward**:
    - Primary: Information Gain per hour
    - Penalties: Safety violations, budget overruns
    - Bonus: Finding new optima
    
    Example usage:
        env = ExperimentGym(
            objective_function=branin_function,
            experiment_types=["xrd", "nmr"],
            max_experiments=20,
            time_budget_hours=24.0
        )
        
        obs = env.reset()
        for step in range(100):
            action = agent.select_action(obs)
            obs, reward, done, info = env.step(action)
            if done:
                break
    """
    
    metadata = {"render.modes": ["human"]}
    
    def __init__(
        self,
        objective_function: callable,
        experiment_types: List[str],
        max_experiments: int = 20,
        time_budget_hours: float = 24.0,
        param_bounds: Optional[Dict[str, Tuple[float, float]]] = None,
        safety_constraints: Optional[Dict[str, Any]] = None
    ):
        super().__init__()
        
        self.objective_function = objective_function
        self.experiment_types = experiment_types
        self.max_experiments = max_experiments
        self.time_budget_hours = time_budget_hours
        self.initial_time_budget = time_budget_hours  # Store initial for normalization
        
        # Default parameter bounds (2D for simplicity)
        self.param_bounds = param_bounds or {
            "param1": (0.0, 1.0),
            "param2": (0.0, 1.0)
        }
        self.n_params = len(self.param_bounds)
        
        self.safety_constraints = safety_constraints or {}
        
        # Define observation space
        # [gp_mean (50), gp_variance (50), experiments_remaining (1), 
        #  time_budget (1), best_value (1), safety_violations (1)]
        self.n_grid_points = 50
        obs_dim = self.n_grid_points * 2 + 4
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(obs_dim,),
            dtype=np.float32
        )
        
        # Define action space
        # [param1, param2, ..., experiment_type_idx]
        action_dim = self.n_params + 1
        self.action_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(action_dim,),
            dtype=np.float32
        )
        
        # Internal state
        self.state: Optional[ExperimentState] = None
        self.history: List[Dict[str, Any]] = []
        self.gp_model = None  # Will be initialized on reset
        
        logger.info(f"ExperimentGym initialized: {max_experiments} experiments, {time_budget_hours}h budget")
    
    def reset(self) -> np.ndarray:
        """
        Reset environment to initial state.
        
        Returns:
            Initial observation
        """
        from sklearn.gaussian_process import GaussianProcessRegressor
        from sklearn.gaussian_process.kernels import RBF, ConstantKernel
        
        # Initialize Gaussian Process surrogate model
        kernel = ConstantKernel(1.0) * RBF(length_scale=0.1)
        self.gp_model = GaussianProcessRegressor(
            kernel=kernel,
            alpha=0.01,  # Noise level
            n_restarts_optimizer=10
        )
        
        # Initial state with no experiments done
        self.state = ExperimentState(
            gp_mean=np.zeros(self.n_grid_points),
            gp_variance=np.ones(self.n_grid_points),
            experiments_remaining=self.max_experiments,
            time_budget_hours=self.time_budget_hours,
            experiments_done=0,
            best_observed_value=-np.inf,
            safety_violations=0
        )
        
        self.history = []
        
        logger.info("Environment reset")
        return self._get_observation()
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        """
        Execute one experiment based on action.
        
        Args:
            action: [param1, param2, ..., experiment_type_idx]
        
        Returns:
            observation: Next state observation
            reward: Reward for this step
            done: Whether episode is complete
            info: Additional information
        """
        if self.state is None:
            raise RuntimeError("Environment not initialized. Call reset() first.")
        
        # Parse action
        params = action[:-1]  # Experiment parameters
        exp_type_idx = int(action[-1] * len(self.experiment_types))
        exp_type_idx = min(exp_type_idx, len(self.experiment_types) - 1)
        exp_type = self.experiment_types[exp_type_idx]
        
        # Denormalize parameters to actual bounds
        params_denorm = self._denormalize_params(params)
        
        # Check safety constraints
        is_safe = self._check_safety(params_denorm)
        if not is_safe:
            self.state.safety_violations += 1
            reward = -10.0  # Large penalty for safety violation
            logger.warning(f"Safety violation at params: {params_denorm}")
            return self._get_observation(), reward, False, {"safety_violation": True}
        
        # Simulate experiment (in real world, this calls the hardware driver)
        result = self._run_experiment(params_denorm, exp_type)
        
        # Update GP model with new data
        self._update_gp_model(params_denorm, result["value"])
        
        # Calculate reward (Information Gain per hour)
        reward = self._calculate_reward(result)
        
        # Update state
        self.state.experiments_done += 1
        self.state.experiments_remaining -= 1
        self.state.time_budget_hours -= result["duration_hours"]
        self.state.best_observed_value = max(
            self.state.best_observed_value,
            result["value"]
        )
        
        # Record history BEFORE updating GP (needed for _update_gp_model)
        self.history.append({
            "step": self.state.experiments_done,
            "params": params_denorm,
            "params_array": np.array(list(params_denorm.values())),
            "experiment_type": exp_type,
            "value": result["value"],
            "reward": reward,
            "timestamp": datetime.now()
        })
        
        # Update GP predictions on grid
        self._update_grid_predictions()
        
        # Check if done
        done = (
            self.state.experiments_remaining <= 0 or
            self.state.time_budget_hours <= 0 or
            self.state.safety_violations >= 3
        )
        
        info = {
            "experiment_type": exp_type,
            "value": result["value"],
            "best_value": self.state.best_observed_value,
            "eig": result.get("eig", 0.0),
            "duration_hours": result["duration_hours"]
        }
        
        logger.debug(f"Step {self.state.experiments_done}: reward={reward:.3f}, value={result['value']:.3f}")
        
        return self._get_observation(), reward, done, info
    
    def render(self, mode: str = "human"):
        """Render the environment state."""
        if mode == "human":
            print(f"\n{'='*50}")
            print(f"Experiment Campaign Status")
            print(f"{'='*50}")
            print(f"Experiments done: {self.state.experiments_done}/{self.max_experiments}")
            print(f"Time remaining: {self.state.time_budget_hours:.1f}h")
            print(f"Best value found: {self.state.best_observed_value:.4f}")
            print(f"Safety violations: {self.state.safety_violations}")
            print(f"{'='*50}\n")
    
    def close(self):
        """Clean up resources."""
        pass
    
    # Private helper methods
    
    def _get_observation(self) -> np.ndarray:
        """Construct observation vector from current state."""
        # Normalize best_observed_value to avoid inf/-inf
        best_value_norm = self.state.best_observed_value
        if np.isinf(best_value_norm):
            best_value_norm = -10.0  # Reasonable default for uninitialized
        
        obs = np.concatenate([
            self.state.gp_mean,
            self.state.gp_variance,
            np.array([
                self.state.experiments_remaining / self.max_experiments,
                self.state.time_budget_hours / self.initial_time_budget,  # Normalize by initial budget
                best_value_norm,
                self.state.safety_violations / 3.0  # Normalize (max is 3)
            ])
        ])
        return obs.astype(np.float32)
    
    def _denormalize_params(self, params_norm: np.ndarray) -> Dict[str, float]:
        """Convert normalized [0, 1] params to actual bounds."""
        params_denorm = {}
        for i, (param_name, (low, high)) in enumerate(self.param_bounds.items()):
            params_denorm[param_name] = low + params_norm[i] * (high - low)
        return params_denorm
    
    def _check_safety(self, params: Dict[str, float]) -> bool:
        """Check if parameters satisfy safety constraints."""
        # TODO: Implement actual safety checks
        # For now, simple bounds check
        for param_name, value in params.items():
            low, high = self.param_bounds[param_name]
            if value < low or value > high:
                return False
        return True
    
    def _run_experiment(self, params: Dict[str, float], exp_type: str) -> Dict[str, Any]:
        """
        Simulate running an experiment.
        
        In production, this would call the actual hardware driver.
        """
        # Evaluate objective function (ground truth)
        param_array = np.array(list(params.values()))
        value = self.objective_function(param_array)
        
        # Add noise to simulate measurement uncertainty
        noise = np.random.normal(0, 0.05)
        measured_value = value + noise
        
        # Simulate experiment duration based on type
        duration_map = {
            "xrd": 0.5,  # 30 minutes
            "nmr": 1.0,  # 1 hour
            "uvvis": 0.25,  # 15 minutes
            "dft": 2.0  # 2 hours (simulation)
        }
        duration_hours = duration_map.get(exp_type, 1.0)
        
        return {
            "value": measured_value,
            "duration_hours": duration_hours,
            "eig": self._calculate_eig(param_array)
        }
    
    def _update_gp_model(self, params: Dict[str, float], value: float):
        """Update Gaussian Process model with new observation."""
        param_array = np.array(list(params.values())).reshape(1, -1)
        value_array = np.array([value])
        
        # Get existing data from history
        if len(self.history) == 0:
            X = param_array
            y = value_array
        else:
            X_prev = np.array([h["params_array"] for h in self.history[:-1] if "params_array" in h])
            y_prev = np.array([h["value"] for h in self.history[:-1]])
            if len(X_prev) > 0:
                X = np.vstack([X_prev, param_array])
                y = np.concatenate([y_prev, value_array])
            else:
                X = param_array
                y = value_array
        
        # Fit GP
        self.gp_model.fit(X, y)
    
    def _update_grid_predictions(self):
        """Update GP mean and variance on a grid for observation."""
        # Create grid of test points
        grid_1d = np.linspace(0, 1, int(np.sqrt(self.n_grid_points)))
        if self.n_params == 2:
            X1, X2 = np.meshgrid(grid_1d, grid_1d)
            X_grid = np.column_stack([X1.ravel(), X2.ravel()])
        else:
            X_grid = grid_1d.reshape(-1, 1)
        
        # Predict on grid
        if len(self.history) > 0:
            mean, std = self.gp_model.predict(X_grid[:self.n_grid_points], return_std=True)
            self.state.gp_mean = mean
            self.state.gp_variance = std ** 2
        else:
            self.state.gp_mean = np.zeros(self.n_grid_points)
            self.state.gp_variance = np.ones(self.n_grid_points)
    
    def _calculate_eig(self, params: np.ndarray) -> float:
        """
        Calculate Expected Information Gain at given parameters.
        
        Simplified: Use GP variance as proxy for information gain.
        """
        if len(self.history) == 0:
            return 1.0  # High EIG when no data
        
        _, std = self.gp_model.predict(params.reshape(1, -1), return_std=True)
        return float(std[0])
    
    def _calculate_reward(self, result: Dict[str, Any]) -> float:
        """
        Calculate reward for the step.
        
        Reward = EIG / duration_hours + bonuses/penalties
        """
        # Base reward: Information Gain per hour
        eig_per_hour = result["eig"] / max(result["duration_hours"], 0.01)
        eig_per_hour = np.clip(eig_per_hour, -100, 100)  # Clip to reasonable range
        
        # Bonus for finding new best
        improvement_bonus = 0.0
        if result["value"] > self.state.best_observed_value:
            improvement = result["value"] - self.state.best_observed_value
            improvement_bonus = np.clip(improvement * 10.0, -50, 50)
        
        # Penalty for running out of budget
        budget_penalty = 0.0
        if self.state.time_budget_hours < 0:
            budget_penalty = -5.0
        
        total_reward = eig_per_hour + improvement_bonus + budget_penalty
        total_reward = np.clip(total_reward, -100, 100)  # Final clipping
        return float(total_reward)


# Benchmark objective functions for testing

def branin_function(x: np.ndarray) -> float:
    """Branin function (common RL benchmark)."""
    x1, x2 = x[0], x[1]
    a = 1.0
    b = 5.1 / (4 * np.pi ** 2)
    c = 5.0 / np.pi
    r = 6.0
    s = 10.0
    t = 1.0 / (8 * np.pi)
    
    term1 = a * (x2 - b * x1 ** 2 + c * x1 - r) ** 2
    term2 = s * (1 - t) * np.cos(x1)
    term3 = s
    
    return -(term1 + term2 + term3)  # Negative for maximization


def rastrigin_function(x: np.ndarray) -> float:
    """Rastrigin function (challenging multi-modal function)."""
    A = 10
    n = len(x)
    return -(A * n + np.sum(x ** 2 - A * np.cos(2 * np.pi * x)))


def hartmann6d_function(x: np.ndarray) -> float:
    """6D Hartmann function (higher dimensional benchmark)."""
    alpha = np.array([1.0, 1.2, 3.0, 3.2])
    A = np.array([
        [10, 3, 17, 3.5, 1.7, 8],
        [0.05, 10, 17, 0.1, 8, 14],
        [3, 3.5, 1.7, 10, 17, 8],
        [17, 8, 0.05, 10, 0.1, 14]
    ])
    P = 1e-4 * np.array([
        [1312, 1696, 5569, 124, 8283, 5886],
        [2329, 4135, 8307, 3736, 1004, 9991],
        [2348, 1451, 3522, 2883, 3047, 6650],
        [4047, 8828, 8732, 5743, 1091, 381]
    ])
    
    outer = 0
    for i in range(4):
        inner = 0
        for j in range(6):
            inner += A[i, j] * (x[j] - P[i, j]) ** 2
        outer += alpha[i] * np.exp(-inner)
    
    return outer  # Already for maximization

