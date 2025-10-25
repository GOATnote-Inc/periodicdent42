"""
Intrinsic Curiosity Module (ICM) for improved exploration.

Based on latest research (2025): Curiosity-driven exploration significantly
improves RL performance in sparse reward environments like experiment optimization.

Reference: "Curiosity-driven Exploration by Self-supervised Prediction"
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import Tuple
import logging

logger = logging.getLogger(__name__)


class ForwardModel(nn.Module):
    """
    Forward model predicts next state given current state and action.
    Prediction error = intrinsic reward (curiosity).
    """
    
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 128):
        super().__init__()
        
        self.network = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, state_dim),
        )
        
    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """Predict next state given current state and action."""
        x = torch.cat([state, action], dim=-1)
        return self.network(x)


class IntrinsicCuriosityModule:
    """
    ICM: Generates intrinsic rewards based on prediction error.
    
    Helps agent explore novel states, crucial for experiment optimization
    where we want to explore high-uncertainty regions.
    """
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        lr: float = 1e-3,
        curiosity_weight: float = 0.1,
        device: str = "cpu",
    ):
        self.device = torch.device(device)
        self.curiosity_weight = curiosity_weight
        
        # Forward model
        self.forward_model = ForwardModel(state_dim, action_dim).to(self.device)
        self.optimizer = optim.Adam(self.forward_model.parameters(), lr=lr)
        
        # Track curiosity over time
        self.curiosity_history = []
        
        logger.info(f"Initialized ICM with curiosity_weight={curiosity_weight}")
    
    def compute_intrinsic_reward(
        self,
        state: np.ndarray,
        action: np.ndarray,
        next_state: np.ndarray,
    ) -> float:
        """
        Compute intrinsic reward based on prediction error.
        
        Higher prediction error = more novel state = higher curiosity reward.
        """
        state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        action_t = torch.FloatTensor(action).unsqueeze(0).to(self.device)
        next_state_t = torch.FloatTensor(next_state).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            predicted_next_state = self.forward_model(state_t, action_t)
            prediction_error = nn.functional.mse_loss(
                predicted_next_state, next_state_t, reduction='none'
            ).mean(dim=-1)
        
        intrinsic_reward = self.curiosity_weight * prediction_error.item()
        self.curiosity_history.append(intrinsic_reward)
        
        return intrinsic_reward
    
    def update(
        self,
        states: np.ndarray,
        actions: np.ndarray,
        next_states: np.ndarray,
    ):
        """Update forward model using collected transitions."""
        states_t = torch.FloatTensor(states).to(self.device)
        actions_t = torch.FloatTensor(actions).to(self.device)
        next_states_t = torch.FloatTensor(next_states).to(self.device)
        
        # Predict next states
        predicted_next_states = self.forward_model(states_t, actions_t)
        
        # Forward model loss
        loss = nn.functional.mse_loss(predicted_next_states, next_states_t)
        
        # Update
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        return loss.item()
    
    def get_avg_curiosity(self, last_n: int = 100) -> float:
        """Get average curiosity over last N steps."""
        if not self.curiosity_history:
            return 0.0
        recent = self.curiosity_history[-last_n:]
        return np.mean(recent)

