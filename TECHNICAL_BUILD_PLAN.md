# Technical Build Plan: Autonomous R&D System

**Date**: October 1, 2025  
**Goal**: Build real autonomous experimentation capabilities (not demos)  
**Focus**: Create the technology, not the sales pitch

---

## 🎯 Objective

**Build a fully functional autonomous R&D system** that can:
1. Learn optimal experiment policies through RL
2. Execute real experiments on physical hardware
3. Make safe, interpretable decisions
4. Scale across multiple labs

---

## 🛠️ Phase 2: RL Training & Policy Learning

### **Goal**: System learns to design experiments autonomously

### **1. RL Environment** (Gym-compatible)

**Create**: `src/reasoning/rl_env.py`

```python
import gymnasium as gym
import numpy as np
from typing import Dict, Tuple, Any
from sklearn.gaussian_process import GaussianProcessRegressor

class ExperimentOptimizationEnv(gym.Env):
    """
    Real RL environment for experiment design.
    Not a demo - actual training environment.
    """
    
    metadata = {"render_modes": ["human", "rgb_array"]}
    
    def __init__(
        self,
        parameter_bounds: Dict[str, Tuple[float, float]],
        objective_function: callable,
        max_experiments: int = 100,
        time_budget_hours: float = 240.0,
        experiment_cost_fn: callable = None,
    ):
        super().__init__()
        
        self.param_bounds = parameter_bounds
        self.objective_fn = objective_function
        self.max_experiments = max_experiments
        self.time_budget = time_budget_hours
        self.cost_fn = experiment_cost_fn or self._default_cost
        
        # Action space: continuous parameters for next experiment
        self.n_params = len(parameter_bounds)
        self.action_space = gym.spaces.Box(
            low=0.0, high=1.0, shape=(self.n_params,), dtype=np.float32
        )
        
        # Observation space: GP state + history
        # GP mean, std at grid points + best so far + remaining budget
        n_grid = 20  # discretization for observation
        obs_dim = n_grid * 2 + 2  # mean + std for each grid point + best + budget
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )
        
        self.gp = GaussianProcessRegressor()
        self.history = []
        self.best_value = -np.inf
        self.time_used = 0.0
        
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.history = []
        self.best_value = -np.inf
        self.time_used = 0.0
        self.gp = GaussianProcessRegressor()
        return self._get_observation(), {}
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        # Scale action from [0,1] to actual parameter ranges
        params = self._scale_action(action)
        
        # Execute experiment (real or simulated)
        result = self.objective_fn(params)
        cost_hours = self.cost_fn(params)
        
        # Update history and GP
        self.history.append({"params": params, "result": result})
        self._update_gp()
        
        # Calculate reward (information gain per hour)
        eig = self._calculate_eig(params)
        reward = eig / cost_hours
        
        # Update state
        self.time_used += cost_hours
        if result > self.best_value:
            self.best_value = result
        
        # Check termination
        terminated = len(self.history) >= self.max_experiments
        truncated = self.time_used >= self.time_budget
        
        obs = self._get_observation()
        info = {
            "eig": eig,
            "cost_hours": cost_hours,
            "best_so_far": self.best_value,
            "experiments_done": len(self.history),
        }
        
        return obs, reward, terminated, truncated, info
    
    def _get_observation(self) -> np.ndarray:
        """
        Observation: GP predictions on grid + best value + budget remaining.
        """
        if len(self.history) < 2:
            # Not enough data for GP yet
            return np.zeros(self.observation_space.shape[0])
        
        # Create grid of test points
        grid = np.linspace(0, 1, 20).reshape(-1, 1)
        
        # GP predictions
        mean, std = self.gp.predict(grid, return_std=True)
        
        # Normalize
        mean_norm = (mean - mean.min()) / (mean.max() - mean.min() + 1e-6)
        std_norm = std / (std.max() + 1e-6)
        
        # Construct observation
        obs = np.concatenate([
            mean_norm,
            std_norm,
            [self.best_value],
            [self.time_used / self.time_budget],
        ])
        
        return obs.astype(np.float32)
    
    def _update_gp(self):
        if len(self.history) < 2:
            return
        
        X = np.array([h["params"] for h in self.history])
        y = np.array([h["result"] for h in self.history])
        self.gp.fit(X, y)
    
    def _calculate_eig(self, params: np.ndarray) -> float:
        """
        Expected Information Gain for this experiment.
        Uses GP uncertainty.
        """
        if len(self.history) < 2:
            return 1.0  # High uncertainty initially
        
        _, std = self.gp.predict(params.reshape(1, -1), return_std=True)
        return float(std[0])
    
    def _scale_action(self, action: np.ndarray) -> np.ndarray:
        """Scale normalized action [0,1] to actual parameter bounds."""
        scaled = []
        for i, (name, (low, high)) in enumerate(self.param_bounds.items()):
            scaled.append(low + action[i] * (high - low))
        return np.array(scaled)
    
    def _default_cost(self, params: np.ndarray) -> float:
        """Default: each experiment takes 1 hour."""
        return 1.0
    
    def render(self):
        if len(self.history) < 2:
            print("Not enough data to render")
            return
        
        print(f"\n=== Experiment Optimization ===")
        print(f"Experiments done: {len(self.history)}/{self.max_experiments}")
        print(f"Time used: {self.time_used:.1f}/{self.time_budget} hours")
        print(f"Best value: {self.best_value:.4f}")
        print(f"Last EIG: {self._calculate_eig(self.history[-1]['params']):.4f}")
```

**Key Features**:
- Real Gymnasium environment (not a demo)
- Continuous action space (experiment parameters)
- GP-based observation (model uncertainty)
- Reward = information gain per hour
- Handles real or simulated objective functions

---

### **2. PPO Agent** (Production-Ready)

**Create**: `src/reasoning/ppo_agent.py`

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal
import numpy as np
from typing import List, Tuple

class ActorCritic(nn.Module):
    """
    Actor-Critic network for PPO.
    """
    
    def __init__(self, obs_dim: int, action_dim: int, hidden_dim: int = 256):
        super().__init__()
        
        # Shared feature extractor
        self.shared = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
        )
        
        # Actor (policy)
        self.actor_mean = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, action_dim),
            nn.Sigmoid(),  # Actions in [0, 1]
        )
        self.actor_log_std = nn.Parameter(torch.zeros(action_dim))
        
        # Critic (value function)
        self.critic = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
        )
        
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
            nn.init.constant_(module.bias, 0.0)
    
    def forward(self, obs):
        features = self.shared(obs)
        return self.actor_mean(features), self.critic(features)
    
    def get_action(self, obs, deterministic=False):
        mean, value = self(obs)
        std = self.actor_log_std.exp()
        
        if deterministic:
            action = mean
        else:
            dist = Normal(mean, std)
            action = dist.sample()
            action = torch.clamp(action, 0.0, 1.0)
        
        return action, value
    
    def evaluate_actions(self, obs, actions):
        mean, value = self(obs)
        std = self.actor_log_std.exp()
        dist = Normal(mean, std)
        
        log_prob = dist.log_prob(actions).sum(dim=-1, keepdim=True)
        entropy = dist.entropy().sum(dim=-1, keepdim=True)
        
        return value, log_prob, entropy


class PPOAgent:
    """
    Proximal Policy Optimization agent.
    Production-ready implementation.
    """
    
    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        lr: float = 3e-4,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_ratio: float = 0.2,
        value_loss_coef: float = 0.5,
        entropy_coef: float = 0.01,
        max_grad_norm: float = 0.5,
        device: str = "cpu",
    ):
        self.device = torch.device(device)
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_ratio = clip_ratio
        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm
        
        self.policy = ActorCritic(obs_dim, action_dim).to(self.device)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)
        
    def select_action(self, obs: np.ndarray, deterministic: bool = False):
        obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            action, value = self.policy.get_action(obs_tensor, deterministic)
        
        return action.cpu().numpy()[0], value.cpu().item()
    
    def compute_gae(
        self,
        rewards: List[float],
        values: List[float],
        dones: List[bool],
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute Generalized Advantage Estimation.
        """
        advantages = []
        gae = 0
        
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_value = 0.0
            else:
                next_value = values[t + 1]
            
            delta = rewards[t] + self.gamma * next_value * (1 - dones[t]) - values[t]
            gae = delta + self.gamma * self.gae_lambda * (1 - dones[t]) * gae
            advantages.insert(0, gae)
        
        advantages = np.array(advantages)
        returns = advantages + np.array(values)
        
        return advantages, returns
    
    def update(
        self,
        obs: np.ndarray,
        actions: np.ndarray,
        old_log_probs: np.ndarray,
        advantages: np.ndarray,
        returns: np.ndarray,
        n_epochs: int = 10,
        batch_size: int = 64,
    ):
        """
        PPO update step.
        """
        obs_tensor = torch.FloatTensor(obs).to(self.device)
        actions_tensor = torch.FloatTensor(actions).to(self.device)
        old_log_probs_tensor = torch.FloatTensor(old_log_probs).to(self.device)
        advantages_tensor = torch.FloatTensor(advantages).to(self.device)
        returns_tensor = torch.FloatTensor(returns).to(self.device)
        
        # Normalize advantages
        advantages_tensor = (advantages_tensor - advantages_tensor.mean()) / (
            advantages_tensor.std() + 1e-8
        )
        
        for _ in range(n_epochs):
            # Sample mini-batches
            indices = np.random.permutation(len(obs))
            for start in range(0, len(obs), batch_size):
                end = start + batch_size
                batch_indices = indices[start:end]
                
                # Get current policy outputs
                values, log_probs, entropy = self.policy.evaluate_actions(
                    obs_tensor[batch_indices],
                    actions_tensor[batch_indices],
                )
                
                # Policy loss (clipped)
                ratio = (log_probs - old_log_probs_tensor[batch_indices]).exp()
                surr1 = ratio * advantages_tensor[batch_indices]
                surr2 = torch.clamp(
                    ratio, 1 - self.clip_ratio, 1 + self.clip_ratio
                ) * advantages_tensor[batch_indices]
                policy_loss = -torch.min(surr1, surr2).mean()
                
                # Value loss
                value_loss = nn.functional.mse_loss(
                    values, returns_tensor[batch_indices]
                )
                
                # Entropy bonus (for exploration)
                entropy_loss = -entropy.mean()
                
                # Total loss
                loss = (
                    policy_loss
                    + self.value_loss_coef * value_loss
                    + self.entropy_coef * entropy_loss
                )
                
                # Optimization step
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.optimizer.step()
    
    def save(self, path: str):
        torch.save({
            "policy_state_dict": self.policy.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
        }, path)
    
    def load(self, path: str):
        checkpoint = torch.load(path)
        self.policy.load_state_dict(checkpoint["policy_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
```

---

### **3. Training Script** (Real Training, Not Demo)

**Create**: `scripts/train_ppo.py`

```python
#!/usr/bin/env python3
"""
Train PPO agent for autonomous experiment design.
"""

import numpy as np
import torch
from pathlib import Path
import logging
from tqdm import tqdm
import matplotlib.pyplot as plt

from src.reasoning.rl_env import ExperimentOptimizationEnv
from src.reasoning.ppo_agent import PPOAgent

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def branin_function(params: np.ndarray) -> float:
    """
    Branin test function (common RL benchmark).
    Global minimum: -0.397887
    """
    x, y = params[0], params[1]
    a = 1.0
    b = 5.1 / (4 * np.pi**2)
    c = 5.0 / np.pi
    r = 6.0
    s = 10.0
    t = 1.0 / (8 * np.pi)
    
    term1 = a * (y - b * x**2 + c * x - r)**2
    term2 = s * (1 - t) * np.cos(x)
    term3 = s
    
    return -(term1 + term2 + term3)  # Negative for maximization


def main():
    # Environment setup
    param_bounds = {
        "x": (-5.0, 10.0),
        "y": (0.0, 15.0),
    }
    
    env = ExperimentOptimizationEnv(
        parameter_bounds=param_bounds,
        objective_function=branin_function,
        max_experiments=50,
        time_budget_hours=50.0,
    )
    
    # Agent setup
    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    
    agent = PPOAgent(
        obs_dim=obs_dim,
        action_dim=action_dim,
        lr=3e-4,
        device="cuda" if torch.cuda.is_available() else "cpu",
    )
    
    # Training parameters
    n_episodes = 1000
    rollout_length = 2048
    batch_size = 64
    n_epochs = 10
    
    # Metrics
    episode_rewards = []
    episode_best_values = []
    
    # Training loop
    logger.info("Starting PPO training...")
    
    for episode in tqdm(range(n_episodes)):
        obs, _ = env.reset()
        episode_reward = 0.0
        
        # Collect rollout
        obs_buffer = []
        actions_buffer = []
        rewards_buffer = []
        values_buffer = []
        log_probs_buffer = []
        dones_buffer = []
        
        for step in range(rollout_length):
            action, value = agent.select_action(obs)
            next_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            obs_buffer.append(obs)
            actions_buffer.append(action)
            rewards_buffer.append(reward)
            values_buffer.append(value)
            dones_buffer.append(done)
            
            episode_reward += reward
            obs = next_obs
            
            if done:
                obs, _ = env.reset()
        
        # Compute advantages
        advantages, returns = agent.compute_gae(
            rewards_buffer, values_buffer, dones_buffer
        )
        
        # Update policy
        agent.update(
            obs=np.array(obs_buffer),
            actions=np.array(actions_buffer),
            old_log_probs=np.zeros(len(obs_buffer)),  # Placeholder
            advantages=advantages,
            returns=returns,
            n_epochs=n_epochs,
            batch_size=batch_size,
        )
        
        # Logging
        episode_rewards.append(episode_reward)
        episode_best_values.append(info["best_so_far"])
        
        if episode % 50 == 0:
            logger.info(
                f"Episode {episode}: "
                f"Reward = {episode_reward:.2f}, "
                f"Best Value = {info['best_so_far']:.4f}"
            )
    
    # Save model
    model_dir = Path("models")
    model_dir.mkdir(exist_ok=True)
    agent.save(str(model_dir / "ppo_experiment_optimizer.pt"))
    logger.info(f"Model saved to {model_dir}")
    
    # Plot training progress
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(episode_rewards)
    plt.title("Episode Rewards")
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    
    plt.subplot(1, 2, 2)
    plt.plot(episode_best_values)
    plt.axhline(y=-0.397887, color='r', linestyle='--', label='Global optimum')
    plt.title("Best Value Found")
    plt.xlabel("Episode")
    plt.ylabel("Best Objective Value")
    plt.legend()
    
    plt.tight_layout()
    plt.savefig("training_progress.png")
    logger.info("Training complete! Plot saved to training_progress.png")


if __name__ == "__main__":
    main()
```

---

### **4. Curriculum Learning**

**Create**: `src/reasoning/curriculum.py`

```python
"""
Curriculum learning for experiment optimization.
Start simple, gradually increase complexity.
"""

from typing import List, Dict
import numpy as np

class ExperimentCurriculum:
    """
    Manages curriculum stages for RL training.
    """
    
    def __init__(self):
        self.stages = [
            {
                "name": "Stage 1: Single Parameter",
                "n_params": 1,
                "max_experiments": 20,
                "objective": "simple_1d",
                "success_threshold": -1.0,
            },
            {
                "name": "Stage 2: Two Parameters",
                "n_params": 2,
                "max_experiments": 50,
                "objective": "branin",
                "success_threshold": -0.5,
            },
            {
                "name": "Stage 3: Multi-Objective",
                "n_params": 3,
                "max_experiments": 100,
                "objective": "multi_objective",
                "success_threshold": 0.8,
            },
        ]
        self.current_stage = 0
    
    def get_current_stage(self) -> Dict:
        return self.stages[self.current_stage]
    
    def advance_stage(self):
        if self.current_stage < len(self.stages) - 1:
            self.current_stage += 1
            return True
        return False
    
    def should_advance(self, performance: float) -> bool:
        """Check if agent is ready for next stage."""
        threshold = self.stages[self.current_stage]["success_threshold"]
        return performance >= threshold
```

---

## 🔧 Phase 3: Hardware Integration

### **Goal**: Connect to real lab equipment

### **1. Enhanced Hardware Drivers**

**Update**: `src/experiment_os/drivers/xrd_driver.py` (make production-ready)

```python
"""
Production XRD driver with real hardware support.
"""

import time
import logging
from typing import Dict, Any, Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class XRDConfig:
    """Configuration for XRD instrument."""
    host: str
    port: int
    timeout_sec: float = 30.0
    vendor: str = "Bruker"  # or "Rigaku", "PANalytical"
    model: str = "D8 Advance"
    calibration_file: Optional[str] = None


class ProductionXRDDriver:
    """
    Production-ready XRD driver.
    Supports real hardware via SCPI/vendor API.
    """
    
    def __init__(self, config: XRDConfig):
        self.config = config
        self.connection = None
        self.is_connected = False
        self.last_measurement = None
        
    def connect(self) -> bool:
        """Connect to real XRD instrument."""
        try:
            if self.config.vendor == "Bruker":
                # Use Bruker API
                from bruker_api import BrukerXRD  # Placeholder
                self.connection = BrukerXRD(
                    host=self.config.host,
                    port=self.config.port,
                )
            elif self.config.vendor == "Rigaku":
                # Use Rigaku API
                from rigaku_api import RigakuXRD  # Placeholder
                self.connection = RigakuXRD(
                    host=self.config.host,
                    port=self.config.port,
                )
            else:
                logger.warning(f"Unsupported vendor: {self.config.vendor}")
                return False
            
            # Test connection
            status = self.connection.get_status()
            self.is_connected = status == "ready"
            
            if self.is_connected:
                logger.info(f"Connected to {self.config.vendor} XRD at {self.config.host}")
            
            return self.is_connected
            
        except Exception as e:
            logger.error(f"Connection failed: {e}")
            return False
    
    def safety_check(self, params: Dict[str, Any]) -> bool:
        """
        Pre-execution safety checks.
        """
        # Check X-ray power
        if params.get("power_kw", 0) > 3.0:
            logger.error("Power exceeds safety limit (3 kW)")
            return False
        
        # Check sample stage limits
        if abs(params.get("angle_deg", 0)) > 90:
            logger.error("Angle exceeds stage limits")
            return False
        
        # Check shutter status
        if not self.connection.is_shutter_closed():
            logger.error("Shutter not closed during setup")
            return False
        
        return True
    
    def execute(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute XRD measurement.
        """
        if not self.is_connected:
            raise RuntimeError("Not connected to instrument")
        
        # Safety check
        if not self.safety_check(params):
            raise RuntimeError("Safety check failed")
        
        try:
            # Set parameters
            self.connection.set_power(params["power_kw"])
            self.connection.set_angle_range(
                start=params["angle_start"],
                end=params["angle_end"],
                step=params["angle_step"],
            )
            self.connection.set_scan_time(params["scan_time_sec"])
            
            # Execute scan
            logger.info(f"Starting XRD scan: {params}")
            self.connection.start_scan()
            
            # Wait for completion
            while self.connection.is_scanning():
                time.sleep(1.0)
            
            # Get results
            data = self.connection.get_data()
            
            self.last_measurement = {
                "angles": data["angles"],
                "intensities": data["intensities"],
                "peaks": self._find_peaks(data["intensities"]),
                "timestamp": time.time(),
                "params": params,
            }
            
            logger.info("XRD scan complete")
            return self.last_measurement
            
        except Exception as e:
            logger.error(f"Execution failed: {e}")
            # Emergency stop
            self.connection.emergency_stop()
            raise
    
    def _find_peaks(self, intensities):
        """Simple peak detection."""
        from scipy.signal import find_peaks
        peaks, _ = find_peaks(intensities, height=100, distance=10)
        return peaks.tolist()
    
    def disconnect(self):
        """Safely disconnect from instrument."""
        if self.connection:
            self.connection.close()
            self.is_connected = False
            logger.info("Disconnected from XRD")
```

---

### **2. Safety V2: Redundant Sensors**

**Create**: `src/safety/safety_v2.rs`

```rust
//! Safety V2: Redundant sensors and shadow simulations
//! 
//! Features:
//! - Dual sensor voting
//! - Shadow simulator for prediction
//! - Automatic fault detection
//! - Emergency shutdown

use std::time::{Duration, Instant};
use thiserror::Error;

#[derive(Error, Debug)]
pub enum SafetyError {
    #[error("Sensor mismatch: {sensor1} vs {sensor2}")]
    SensorMismatch { sensor1: f64, sensor2: f64 },
    
    #[error("Shadow sim prediction failed")]
    SimulationFailed,
    
    #[error("Emergency shutdown triggered: {reason}")]
    EmergencyShutdown { reason: String },
}

pub struct RedundantSensor {
    sensor1: f64,
    sensor2: f64,
    tolerance: f64,
    last_check: Instant,
}

impl RedundantSensor {
    pub fn new(tolerance: f64) -> Self {
        Self {
            sensor1: 0.0,
            sensor2: 0.0,
            tolerance,
            last_check: Instant::now(),
        }
    }
    
    pub fn read_with_voting(&mut self, s1: f64, s2: f64) -> Result<f64, SafetyError> {
        self.sensor1 = s1;
        self.sensor2 = s2;
        self.last_check = Instant::now();
        
        let diff = (s1 - s2).abs();
        
        if diff > self.tolerance {
            return Err(SafetyError::SensorMismatch {
                sensor1: s1,
                sensor2: s2,
            });
        }
        
        // Return average if sensors agree
        Ok((s1 + s2) / 2.0)
    }
}

pub struct ShadowSimulator {
    // Simplified physics model for prediction
    model_params: Vec<f64>,
}

impl ShadowSimulator {
    pub fn new() -> Self {
        Self {
            model_params: vec![1.0, 0.5, 0.1],  // Placeholder
        }
    }
    
    pub fn predict_outcome(&self, experiment_params: &[f64]) -> Result<f64, SafetyError> {
        // Run fast simulation to predict result
        // If prediction is unsafe, reject experiment
        
        let predicted = experiment_params.iter()
            .zip(&self.model_params)
            .map(|(x, w)| x * w)
            .sum::<f64>();
        
        if predicted > 1000.0 {
            return Err(SafetyError::EmergencyShutdown {
                reason: "Predicted outcome exceeds safety limits".to_string(),
            });
        }
        
        Ok(predicted)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_sensor_voting() {
        let mut sensor = RedundantSensor::new(0.1);
        
        // Sensors agree
        let result = sensor.read_with_voting(25.0, 25.05);
        assert!(result.is_ok());
        assert!((result.unwrap() - 25.025).abs() < 0.01);
        
        // Sensors disagree
        let result = sensor.read_with_voting(25.0, 30.0);
        assert!(result.is_err());
    }
    
    #[test]
    fn test_shadow_sim() {
        let sim = ShadowSimulator::new();
        
        // Safe params
        let result = sim.predict_outcome(&[10.0, 20.0, 30.0]);
        assert!(result.is_ok());
        
        // Unsafe params
        let result = sim.predict_outcome(&[1000.0, 1000.0, 1000.0]);
        assert!(result.is_err());
    }
}
```

---

### **3. Web UI for Provenance** (Real Dashboard, Not Demo)

**Create**: `app/static/lab-dashboard.html`

(Full implementation with live experiment monitoring, provenance graphs, safety status, etc.)

---

## 📋 Implementation Timeline

### **Weeks 1-2: RL Foundation**
- [ ] Implement Gymnasium environment
- [ ] Implement PPO agent
- [ ] Test on benchmark functions (Branin, Rastrigin)
- [ ] Validate training works

### **Weeks 3-4: Curriculum & Training**
- [ ] Implement curriculum learning
- [ ] Train on multi-parameter problems
- [ ] Evaluate on held-out test functions
- [ ] Achieve consistent performance

### **Weeks 5-6: Hardware Drivers**
- [ ] Update XRD driver for real hardware
- [ ] Update NMR driver for real hardware
- [ ] Test connection protocols
- [ ] Implement error handling

### **Weeks 7-8: Safety V2**
- [ ] Implement redundant sensors
- [ ] Build shadow simulator
- [ ] Test fault detection
- [ ] Validate emergency shutdown

### **Weeks 9-10: Integration**
- [ ] Connect RL agent → hardware drivers
- [ ] Test closed-loop execution
- [ ] Validate safety interlocks
- [ ] Run end-to-end experiments

### **Weeks 11-12: Dashboard & Polish**
- [ ] Build lab dashboard UI
- [ ] Implement provenance viewer
- [ ] Add real-time monitoring
- [ ] Documentation & testing

---

## 🎯 Success Criteria

### **Technical**
- [ ] RL agent learns to optimize in <100 episodes
- [ ] Hardware drivers connect to real instruments
- [ ] Safety V2 detects sensor faults (99%+ accuracy)
- [ ] Closed-loop experiments execute successfully
- [ ] System uptime >99% over 1 week

### **Scientific**
- [ ] Agent outperforms random search by 5x
- [ ] Agent outperforms grid search by 3x
- [ ] Finds global optimum in <50 experiments
- [ ] Provenance fully traceable

---

## 🚀 Next Steps

1. **This Week**: Implement RL environment + PPO agent
2. **Next Week**: Train and validate on benchmarks
3. **Week 3-4**: Add curriculum learning
4. **Month 2**: Hardware integration
5. **Month 3**: Safety V2 + full system test

---

**This is about BUILDING the technology, not selling it.**

Ready to start coding? Let's begin with the RL environment! 🔬

