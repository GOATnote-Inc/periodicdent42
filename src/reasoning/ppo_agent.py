"""
Proximal Policy Optimization (PPO) Agent for Experiment Design.

Production-ready implementation with Actor-Critic architecture.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal
import numpy as np
from typing import List, Tuple, Dict, Optional
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class ActorCritic(nn.Module):
    """
    Actor-Critic network for PPO.
    
    Shared feature extractor with separate actor (policy) and critic (value) heads.
    """
    
    def __init__(self, obs_dim: int, action_dim: int, hidden_dim: int = 256):
        super().__init__()
        
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        
        # Shared feature extractor
        self.shared = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
        )
        
        # Actor (policy) network
        self.actor_mean = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, action_dim),
            nn.Sigmoid(),  # Output in [0, 1]
        )
        
        # Learnable log standard deviation
        self.actor_log_std = nn.Parameter(torch.zeros(action_dim))
        
        # Critic (value) network
        self.critic = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
        )
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        """Orthogonal initialization for better training stability."""
        if isinstance(module, nn.Linear):
            nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
            nn.init.constant_(module.bias, 0.0)
    
    def forward(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.
        
        Returns:
            mean: Action mean
            value: State value estimate
        """
        features = self.shared(obs)
        mean = self.actor_mean(features)
        value = self.critic(features)
        return mean, value
    
    def get_action(
        self, obs: torch.Tensor, deterministic: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Sample action from policy.
        
        Args:
            obs: Observation tensor
            deterministic: If True, return mean action (no sampling)
        
        Returns:
            action: Sampled action
            value: Value estimate for this state
        """
        mean, value = self(obs)
        std = self.actor_log_std.exp().clamp(min=1e-3, max=1.0)
        
        if deterministic:
            action = mean
        else:
            dist = Normal(mean, std)
            action = dist.sample()
            action = torch.clamp(action, 0.0, 1.0)
        
        return action, value
    
    def evaluate_actions(
        self, obs: torch.Tensor, actions: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Evaluate actions for PPO update.
        
        Returns:
            value: Value estimates
            log_prob: Log probabilities of actions
            entropy: Entropy of action distribution
        """
        mean, value = self(obs)
        std = self.actor_log_std.exp().clamp(min=1e-3, max=1.0)
        dist = Normal(mean, std)
        
        log_prob = dist.log_prob(actions).sum(dim=-1, keepdim=True)
        entropy = dist.entropy().sum(dim=-1, keepdim=True)
        
        return value, log_prob, entropy


class PPOAgent:
    """
    Proximal Policy Optimization agent.
    
    Production implementation with:
    - Generalized Advantage Estimation (GAE)
    - Clipped surrogate objective
    - Value function loss
    - Entropy regularization
    - Gradient clipping
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
        """
        Initialize PPO agent.
        
        Args:
            obs_dim: Observation space dimension
            action_dim: Action space dimension
            lr: Learning rate
            gamma: Discount factor
            gae_lambda: GAE lambda parameter
            clip_ratio: PPO clipping parameter
            value_loss_coef: Value loss coefficient
            entropy_coef: Entropy bonus coefficient
            max_grad_norm: Maximum gradient norm
            device: Device to run on ('cpu' or 'cuda')
        """
        self.device = torch.device(device)
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_ratio = clip_ratio
        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm
        
        self.policy = ActorCritic(obs_dim, action_dim).to(self.device)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)
        
        logger.info(f"Initialized PPO agent on {self.device}")
        logger.info(f"  obs_dim={obs_dim}, action_dim={action_dim}")
        logger.info(f"  lr={lr}, gamma={gamma}, clip_ratio={clip_ratio}")
    
    def select_action(
        self, obs: np.ndarray, deterministic: bool = False
    ) -> Tuple[np.ndarray, float, float]:
        """
        Select action given observation.
        
        Args:
            obs: Observation array
            deterministic: If True, use mean action (no exploration)
        
        Returns:
            action: Selected action
            value: Value estimate
            log_prob: Log probability of action
        """
        obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            action, value = self.policy.get_action(obs_tensor, deterministic)
            
            # Calculate log prob
            mean, _ = self.policy(obs_tensor)
            std = self.policy.actor_log_std.exp().clamp(min=1e-3, max=1.0)
            dist = Normal(mean, std)
            log_prob = dist.log_prob(action).sum(dim=-1)
        
        return (
            action.cpu().numpy()[0],
            value.cpu().item(),
            log_prob.cpu().item()
        )
    
    def compute_gae(
        self,
        rewards: List[float],
        values: List[float],
        dones: List[bool],
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute Generalized Advantage Estimation.
        
        Args:
            rewards: List of rewards
            values: List of value estimates
            dones: List of done flags
        
        Returns:
            advantages: Advantage estimates
            returns: Discounted returns
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
    ) -> Dict[str, float]:
        """
        Update policy using PPO.
        
        Args:
            obs: Observations
            actions: Actions taken
            old_log_probs: Log probs of actions under old policy
            advantages: Advantage estimates
            returns: Discounted returns
            n_epochs: Number of epochs to train
            batch_size: Mini-batch size
        
        Returns:
            Dictionary of training metrics
        """
        # Convert to tensors
        obs_tensor = torch.FloatTensor(obs).to(self.device)
        actions_tensor = torch.FloatTensor(actions).to(self.device)
        old_log_probs_tensor = torch.FloatTensor(old_log_probs).to(self.device)
        advantages_tensor = torch.FloatTensor(advantages).to(self.device)
        returns_tensor = torch.FloatTensor(returns).to(self.device)
        
        # Normalize advantages
        advantages_tensor = (advantages_tensor - advantages_tensor.mean()) / (
            advantages_tensor.std() + 1e-8
        )
        
        # Training metrics
        total_policy_loss = 0.0
        total_value_loss = 0.0
        total_entropy = 0.0
        n_updates = 0
        
        for epoch in range(n_epochs):
            # Sample mini-batches
            indices = np.random.permutation(len(obs))
            
            for start in range(0, len(obs), batch_size):
                end = min(start + batch_size, len(obs))
                batch_indices = indices[start:end]
                
                # Get current policy outputs
                values, log_probs, entropy = self.policy.evaluate_actions(
                    obs_tensor[batch_indices],
                    actions_tensor[batch_indices],
                )
                
                # Policy loss (clipped surrogate objective)
                ratio = (log_probs - old_log_probs_tensor[batch_indices]).exp()
                surr1 = ratio * advantages_tensor[batch_indices]
                surr2 = torch.clamp(
                    ratio, 1 - self.clip_ratio, 1 + self.clip_ratio
                ) * advantages_tensor[batch_indices]
                policy_loss = -torch.min(surr1, surr2).mean()
                
                # Value loss (ensure shapes match)
                value_loss = nn.functional.mse_loss(
                    values.squeeze(-1), returns_tensor[batch_indices]
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
                nn.utils.clip_grad_norm_(
                    self.policy.parameters(), self.max_grad_norm
                )
                self.optimizer.step()
                
                # Track metrics
                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                total_entropy += entropy.mean().item()
                n_updates += 1
        
        return {
            "policy_loss": total_policy_loss / n_updates,
            "value_loss": total_value_loss / n_updates,
            "entropy": total_entropy / n_updates,
        }
    
    def save(self, path: str):
        """Save model checkpoint."""
        save_path = Path(path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        torch.save({
            "policy_state_dict": self.policy.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
        }, save_path)
        
        logger.info(f"Model saved to {save_path}")
    
    def load(self, path: str):
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        self.policy.load_state_dict(checkpoint["policy_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        
        logger.info(f"Model loaded from {path}")

