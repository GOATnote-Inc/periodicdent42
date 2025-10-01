"""
Reinforcement Learning Environment for Experiment Optimization.

Production environment using Gymnasium (formerly OpenAI Gym).
Trains agents to design optimal experiment sequences.
"""

import gymnasium as gym
import numpy as np
from typing import Dict, Tuple, Any, Optional
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel
import logging

logger = logging.getLogger(__name__)


class ExperimentOptimizationEnv(gym.Env):
    """
    Gymnasium environment for autonomous experiment design.
    
    Agent learns to select next experiments that maximize information gain
    per unit cost (time/resources).
    
    Action Space: Continuous [0,1]^n where n = number of parameters
    Observation Space: GP model state (mean, std on grid) + history info
    Reward: Expected Information Gain / Cost
    """
    
    metadata = {"render_modes": ["human", "rgb_array"]}
    
    def __init__(
        self,
        parameter_bounds: Dict[str, Tuple[float, float]],
        objective_function: callable,
        max_experiments: int = 100,
        time_budget_hours: float = 240.0,
        experiment_cost_fn: Optional[callable] = None,
        noise_std: float = 0.1,
        seed: Optional[int] = None,
    ):
        """
        Initialize experiment optimization environment.
        
        Args:
            parameter_bounds: Dict mapping parameter names to (min, max) bounds
            objective_function: Ground truth function to optimize
            max_experiments: Maximum experiments per episode
            time_budget_hours: Total time budget in hours
            experiment_cost_fn: Function that returns cost in hours for given params
            noise_std: Observation noise standard deviation
            seed: Random seed for reproducibility
        """
        super().__init__()
        
        self.param_bounds = parameter_bounds
        self.objective_fn = objective_function
        self.max_experiments = max_experiments
        self.initial_time_budget = time_budget_hours
        self.time_budget = time_budget_hours
        self.cost_fn = experiment_cost_fn or self._default_cost
        self.noise_std = noise_std
        
        if seed is not None:
            np.random.seed(seed)
        
        # Action space: continuous parameters in [0, 1]
        self.n_params = len(parameter_bounds)
        self.action_space = gym.spaces.Box(
            low=0.0, high=1.0, shape=(self.n_params,), dtype=np.float32
        )
        
        # Observation space: GP predictions on grid + metadata
        # Grid of mean/std predictions + best value + budget remaining + experiments done
        self.grid_size = 20
        obs_dim = (self.grid_size * 2 +  # mean + std for each grid point
                   1 +  # best value so far
                   1 +  # budget remaining (normalized)
                   1)   # experiments done (normalized)
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )
        
        # State
        self.gp = None
        self.history = []
        self.best_value = -np.inf
        self.best_params = None
        self.time_used = 0.0
        self.current_step = 0
        
        logger.info(
            f"Initialized ExperimentOptimizationEnv: "
            f"{self.n_params} params, {max_experiments} max experiments, "
            f"{time_budget_hours}h budget"
        )
    
    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None):
        """Reset environment to initial state."""
        super().reset(seed=seed)
        
        if seed is not None:
            np.random.seed(seed)
        
        # Initialize GP with RBF kernel
        kernel = ConstantKernel(1.0) * RBF(length_scale=1.0)
        self.gp = GaussianProcessRegressor(
            kernel=kernel,
            alpha=self.noise_std**2,
            n_restarts_optimizer=10,
            normalize_y=True,
        )
        
        self.history = []
        self.best_value = -np.inf
        self.best_params = None
        self.time_used = 0.0
        self.current_step = 0
        
        return self._get_observation(), {}
    
    def step(
        self, action: np.ndarray
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """
        Execute one experiment.
        
        Args:
            action: Normalized parameters in [0, 1]
        
        Returns:
            observation, reward, terminated, truncated, info
        """
        # Scale action from [0,1] to actual parameter ranges
        params = self._scale_action(action)
        
        # Execute experiment (with noise)
        true_value = self.objective_fn(params)
        observed_value = true_value + np.random.normal(0, self.noise_std)
        
        # Calculate cost
        cost_hours = self.cost_fn(params)
        
        # Calculate EIG before adding this point
        eig = self._calculate_eig(params)
        
        # Update history
        self.history.append({
            "params": params,
            "value": observed_value,
            "true_value": true_value,
            "cost_hours": cost_hours,
            "eig": eig,
        })
        
        # Update GP model
        self._update_gp()
        
        # Calculate reward (EIG per hour)
        reward = eig / max(cost_hours, 0.1)  # Avoid division by zero
        
        # Update state
        self.time_used += cost_hours
        self.current_step += 1
        
        if observed_value > self.best_value:
            self.best_value = observed_value
            self.best_params = params
        
        # Check termination conditions
        terminated = self.current_step >= self.max_experiments
        truncated = self.time_used >= self.time_budget
        
        # Get new observation
        obs = self._get_observation()
        
        # Info dictionary
        info = {
            "eig": eig,
            "cost_hours": cost_hours,
            "best_value": self.best_value,
            "best_params": self.best_params,
            "experiments_done": self.current_step,
            "time_used": self.time_used,
            "current_value": observed_value,
        }
        
        return obs, reward, terminated, truncated, info
    
    def _get_observation(self) -> np.ndarray:
        """
        Construct observation from GP state.
        
        Returns state of GP model (uncertainty landscape) plus metadata.
        """
        if len(self.history) < 2:
            # Not enough data for GP yet - return zeros
            return np.zeros(self.observation_space.shape[0], dtype=np.float32)
        
        # Create grid of test points in [0, 1]
        if self.n_params == 1:
            grid = np.linspace(0, 1, self.grid_size).reshape(-1, 1)
        else:
            # For multi-dimensional, sample grid
            grid = np.random.rand(self.grid_size, self.n_params)
        
        # Scale grid to parameter bounds
        scaled_grid = np.array([
            self._scale_action(g) for g in grid
        ])
        
        # GP predictions
        try:
            mean, std = self.gp.predict(scaled_grid, return_std=True)
        except Exception as e:
            logger.warning(f"GP prediction failed: {e}")
            return np.zeros(self.observation_space.shape[0], dtype=np.float32)
        
        # Normalize predictions
        mean_norm = (mean - mean.min()) / (mean.max() - mean.min() + 1e-8)
        std_norm = std / (std.max() + 1e-8)
        
        # Handle inf values for best_observed_value
        best_observed_value = self.best_value
        if np.isinf(best_observed_value):
            best_observed_value = 0.0
        
        # Construct observation vector
        obs = np.concatenate([
            mean_norm,
            std_norm,
            [best_observed_value],
            [self.time_used / self.initial_time_budget],
            [self.current_step / self.max_experiments],
        ])
        
        return obs.astype(np.float32)
    
    def _update_gp(self):
        """Update GP model with latest data."""
        if len(self.history) < 2:
            return
        
        X = np.array([h["params"] for h in self.history])
        y = np.array([h["value"] for h in self.history])
        
        try:
            self.gp.fit(X, y)
        except Exception as e:
            logger.warning(f"GP fit failed: {e}")
    
    def _calculate_eig(self, params: np.ndarray) -> float:
        """
        Calculate Expected Information Gain for this experiment.
        
        Uses GP uncertainty (standard deviation) as proxy for EIG.
        Higher uncertainty = more information gain.
        """
        if len(self.history) < 2:
            return 1.0  # High initial uncertainty
        
        try:
            _, std = self.gp.predict(params.reshape(1, -1), return_std=True)
            # Clip to prevent inf rewards
            eig = float(np.clip(std[0], 0.0, 10.0))
            return eig
        except Exception as e:
            logger.warning(f"EIG calculation failed: {e}")
            return 0.1  # Small default value
    
    def _scale_action(self, action: np.ndarray) -> np.ndarray:
        """Scale normalized action [0,1] to actual parameter bounds."""
        scaled = []
        for i, (name, (low, high)) in enumerate(self.param_bounds.items()):
            scaled.append(low + action[i] * (high - low))
        return np.array(scaled)
    
    def _default_cost(self, params: np.ndarray) -> float:
        """Default cost function: each experiment takes 1 hour."""
        return 1.0
    
    def render(self):
        """Render current state (for debugging)."""
        if self.current_step == 0:
            print("No experiments executed yet")
            return
        
        print(f"\n{'='*60}")
        print(f"Experiment Optimization Environment")
        print(f"{'='*60}")
        print(f"Experiments done: {self.current_step}/{self.max_experiments}")
        print(f"Time used: {self.time_used:.1f}/{self.time_budget:.1f} hours")
        print(f"Best value found: {self.best_value:.6f}")
        print(f"Best parameters: {self.best_params}")
        if self.history:
            last = self.history[-1]
            print(f"Last experiment:")
            print(f"  Params: {last['params']}")
            print(f"  Value: {last['value']:.6f}")
            print(f"  EIG: {last['eig']:.6f}")
            print(f"  Cost: {last['cost_hours']:.2f}h")
        print(f"{'='*60}\n")
    
    def close(self):
        """Cleanup resources."""
        pass
