"""
Reinforcement Learning Agent for Experiment Planning

Implements PPO (Proximal Policy Optimization) agent for learning
optimal experiment sequences that maximize information gain.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import List, Tuple, Dict, Any
import logging
from dataclasses import dataclass
from collections import deque

logger = logging.getLogger(__name__)


@dataclass
class RLConfig:
    """Configuration for RL training."""
    learning_rate: float = 3e-4
    gamma: float = 0.99  # Discount factor
    gae_lambda: float = 0.95  # GAE parameter
    clip_epsilon: float = 0.2  # PPO clip range
    value_loss_coef: float = 0.5
    entropy_coef: float = 0.01
    max_grad_norm: float = 0.5
    batch_size: int = 64
    n_epochs: int = 10
    buffer_size: int = 2048


class ActorCriticNetwork(nn.Module):
    """
    Actor-Critic network for PPO.
    
    Actor: Outputs action distribution (mean and std for continuous actions)
    Critic: Outputs state value estimate
    """
    
    def __init__(self, obs_dim: int, action_dim: int, hidden_dim: int = 256):
        super().__init__()
        
        # Shared feature extractor
        self.shared = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        # Actor head (policy)
        self.actor_mean = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, action_dim),
            nn.Tanh()  # Actions in [0, 1]
        )
        
        self.actor_logstd = nn.Parameter(torch.zeros(action_dim))
        
        # Critic head (value function)
        self.critic = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )
    
    def forward(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass.
        
        Returns:
            action_mean: Mean of action distribution
            action_logstd: Log std of action distribution
            value: State value estimate
        """
        features = self.shared(obs)
        action_mean = self.actor_mean(features)
        action_logstd = self.actor_logstd.expand_as(action_mean)
        value = self.critic(features)
        
        return action_mean, action_logstd, value
    
    def get_action(self, obs: torch.Tensor, deterministic: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Sample action from policy.
        
        Args:
            obs: Observation tensor
            deterministic: If True, return mean action (no sampling)
        
        Returns:
            action: Sampled action
            log_prob: Log probability of action
        """
        action_mean, action_logstd, _ = self.forward(obs)
        
        if deterministic:
            return action_mean, None
        
        action_std = torch.exp(action_logstd)
        dist = torch.distributions.Normal(action_mean, action_std)
        action = dist.sample()
        log_prob = dist.log_prob(action).sum(dim=-1)
        
        # Clip action to [0, 1]
        action = torch.clamp(action, 0.0, 1.0)
        
        return action, log_prob
    
    def evaluate_actions(
        self,
        obs: torch.Tensor,
        actions: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Evaluate actions under current policy.
        
        Returns:
            log_probs: Log probabilities of actions
            values: State value estimates
            entropy: Policy entropy
        """
        action_mean, action_logstd, values = self.forward(obs)
        
        action_std = torch.exp(action_logstd)
        dist = torch.distributions.Normal(action_mean, action_std)
        
        log_probs = dist.log_prob(actions).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1).mean()
        
        return log_probs, values, entropy


class PPOAgent:
    """
    Proximal Policy Optimization agent.
    
    Example usage:
        agent = PPOAgent(obs_dim=104, action_dim=3, config=RLConfig())
        
        # Training loop
        for episode in range(1000):
            obs = env.reset()
            done = False
            
            while not done:
                action = agent.select_action(obs)
                next_obs, reward, done, info = env.step(action)
                agent.store_transition(obs, action, reward, done, next_obs)
                obs = next_obs
            
            # Update policy
            if agent.ready_to_update():
                agent.update()
    """
    
    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        config: RLConfig = None,
        device: str = "cpu"
    ):
        self.config = config or RLConfig()
        self.device = torch.device(device)
        
        # Initialize network
        self.network = ActorCriticNetwork(obs_dim, action_dim).to(self.device)
        self.optimizer = optim.Adam(self.network.parameters(), lr=self.config.learning_rate)
        
        # Experience buffer
        self.obs_buffer = deque(maxlen=self.config.buffer_size)
        self.action_buffer = deque(maxlen=self.config.buffer_size)
        self.reward_buffer = deque(maxlen=self.config.buffer_size)
        self.done_buffer = deque(maxlen=self.config.buffer_size)
        self.value_buffer = deque(maxlen=self.config.buffer_size)
        self.log_prob_buffer = deque(maxlen=self.config.buffer_size)
        
        # Training metrics
        self.episode_rewards = []
        self.update_count = 0
        
        logger.info(f"PPO Agent initialized on {self.device}")
    
    def select_action(self, obs: np.ndarray, deterministic: bool = False) -> np.ndarray:
        """
        Select action given observation.
        
        Args:
            obs: Observation array
            deterministic: If True, return mean action (for evaluation)
        
        Returns:
            Action array
        """
        obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            action, log_prob = self.network.get_action(obs_tensor, deterministic)
            _, _, value = self.network.forward(obs_tensor)
        
        # Store for training
        if not deterministic:
            self.value_buffer.append(value.cpu().item())
            if log_prob is not None:
                self.log_prob_buffer.append(log_prob.cpu().item())
        
        return action.cpu().numpy()[0]
    
    def store_transition(
        self,
        obs: np.ndarray,
        action: np.ndarray,
        reward: float,
        done: bool,
        next_obs: np.ndarray
    ):
        """Store transition in buffer."""
        self.obs_buffer.append(obs)
        self.action_buffer.append(action)
        self.reward_buffer.append(reward)
        self.done_buffer.append(done)
    
    def ready_to_update(self) -> bool:
        """Check if buffer has enough samples for update."""
        return len(self.obs_buffer) >= self.config.batch_size
    
    def update(self) -> Dict[str, float]:
        """
        Update policy using PPO.
        
        Returns:
            Dictionary of training metrics
        """
        if not self.ready_to_update():
            return {}
        
        # Convert buffers to tensors
        obs = torch.FloatTensor(np.array(self.obs_buffer)).to(self.device)
        actions = torch.FloatTensor(np.array(self.action_buffer)).to(self.device)
        rewards = torch.FloatTensor(np.array(self.reward_buffer)).to(self.device)
        dones = torch.FloatTensor(np.array(self.done_buffer)).to(self.device)
        old_log_probs = torch.FloatTensor(np.array(self.log_prob_buffer)).to(self.device)
        
        # Compute advantages using GAE
        advantages, returns = self._compute_gae(rewards, dones)
        
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # PPO update for multiple epochs
        metrics = {
            "policy_loss": 0.0,
            "value_loss": 0.0,
            "entropy": 0.0,
            "approx_kl": 0.0
        }
        
        for epoch in range(self.config.n_epochs):
            # Mini-batch updates
            indices = torch.randperm(len(obs))
            
            for start in range(0, len(obs), self.config.batch_size):
                end = start + self.config.batch_size
                batch_indices = indices[start:end]
                
                batch_obs = obs[batch_indices]
                batch_actions = actions[batch_indices]
                batch_old_log_probs = old_log_probs[batch_indices]
                batch_advantages = advantages[batch_indices]
                batch_returns = returns[batch_indices]
                
                # Evaluate actions under current policy
                log_probs, values, entropy = self.network.evaluate_actions(
                    batch_obs,
                    batch_actions
                )
                
                # PPO policy loss with clipping
                ratio = torch.exp(log_probs - batch_old_log_probs)
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(
                    ratio,
                    1.0 - self.config.clip_epsilon,
                    1.0 + self.config.clip_epsilon
                ) * batch_advantages
                policy_loss = -torch.min(surr1, surr2).mean()
                
                # Value loss
                value_loss = nn.functional.mse_loss(values.squeeze(), batch_returns)
                
                # Total loss
                loss = (
                    policy_loss +
                    self.config.value_loss_coef * value_loss -
                    self.config.entropy_coef * entropy
                )
                
                # Optimize
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.network.parameters(), self.config.max_grad_norm)
                self.optimizer.step()
                
                # Track metrics
                metrics["policy_loss"] += policy_loss.item()
                metrics["value_loss"] += value_loss.item()
                metrics["entropy"] += entropy.item()
                
                with torch.no_grad():
                    approx_kl = (batch_old_log_probs - log_probs).mean().item()
                    metrics["approx_kl"] += approx_kl
        
        # Average metrics
        n_updates = self.config.n_epochs * (len(obs) // self.config.batch_size)
        for key in metrics:
            metrics[key] /= n_updates
        
        self.update_count += 1
        
        # Clear buffers
        self.obs_buffer.clear()
        self.action_buffer.clear()
        self.reward_buffer.clear()
        self.done_buffer.clear()
        self.value_buffer.clear()
        self.log_prob_buffer.clear()
        
        logger.info(f"Update {self.update_count}: policy_loss={metrics['policy_loss']:.4f}, value_loss={metrics['value_loss']:.4f}")
        
        return metrics
    
    def _compute_gae(
        self,
        rewards: torch.Tensor,
        dones: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute Generalized Advantage Estimation (GAE).
        
        Returns:
            advantages: GAE advantages
            returns: Value targets
        """
        advantages = torch.zeros_like(rewards)
        returns = torch.zeros_like(rewards)
        
        # Get values from buffer
        values = torch.FloatTensor(self.value_buffer).to(self.device)
        
        gae = 0
        next_value = 0  # Assume terminal state has value 0
        
        # Compute GAE backwards
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_non_terminal = 1.0 - dones[t]
                next_value = 0
            else:
                next_non_terminal = 1.0 - dones[t]
                next_value = values[t + 1]
            
            delta = rewards[t] + self.config.gamma * next_value * next_non_terminal - values[t]
            gae = delta + self.config.gamma * self.config.gae_lambda * next_non_terminal * gae
            
            advantages[t] = gae
            returns[t] = gae + values[t]
        
        return advantages, returns
    
    def save(self, path: str):
        """Save agent to file."""
        torch.save({
            "network_state_dict": self.network.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "update_count": self.update_count,
            "config": self.config
        }, path)
        logger.info(f"Agent saved to {path}")
    
    def load(self, path: str):
        """Load agent from file."""
        checkpoint = torch.load(path, map_location=self.device)
        self.network.load_state_dict(checkpoint["network_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.update_count = checkpoint["update_count"]
        logger.info(f"Agent loaded from {path}")

