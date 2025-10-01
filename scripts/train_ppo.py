#!/usr/bin/env python3
"""
Train PPO agent for autonomous experiment design.

Week 1 implementation - trains on Branin benchmark function.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import torch
import logging
from tqdm import tqdm
import matplotlib.pyplot as plt
from collections import deque

from src.reasoning.rl_env import ExperimentOptimizationEnv
from src.reasoning.ppo_agent import PPOAgent

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def branin_function(params: np.ndarray) -> float:
    """
    Branin test function (common RL benchmark).
    
    Global minimum: f(x) = 0.397887 at:
    - (-π, 12.275)
    - (π, 2.275)
    - (9.42478, 2.475)
    
    We negate for maximization problem.
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


def train_ppo(
    n_episodes: int = 500,
    max_steps_per_episode: int = 50,
    rollout_length: int = 1024,
    batch_size: int = 64,
    n_epochs: int = 10,
    device: str = "auto",
):
    """
    Train PPO agent.
    
    Args:
        n_episodes: Number of training episodes
        max_steps_per_episode: Max steps per episode
        rollout_length: Steps to collect before update
        batch_size: Mini-batch size for updates
        n_epochs: Epochs per PPO update
        device: Device to use ('cpu', 'cuda', or 'auto')
    """
    # Device selection
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")
    
    # Environment setup
    param_bounds = {
        "x": (-5.0, 10.0),
        "y": (0.0, 15.0),
    }
    
    env = ExperimentOptimizationEnv(
        parameter_bounds=param_bounds,
        objective_function=branin_function,
        max_experiments=max_steps_per_episode,
        time_budget_hours=max_steps_per_episode * 1.0,  # 1 hour per experiment
        noise_std=0.1,
    )
    
    # Agent setup
    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    
    agent = PPOAgent(
        obs_dim=obs_dim,
        action_dim=action_dim,
        lr=3e-4,
        gamma=0.99,
        gae_lambda=0.95,
        clip_ratio=0.2,
        device=device,
    )
    
    # Training metrics
    episode_rewards = []
    episode_best_values = []
    episode_lengths = []
    recent_rewards = deque(maxlen=100)
    
    # Training loop
    logger.info("Starting PPO training...")
    logger.info(f"Episodes: {n_episodes}, Rollout: {rollout_length}")
    logger.info(f"Environment: {obs_dim}D obs, {action_dim}D action")
    
    global_step = 0
    
    for episode in tqdm(range(n_episodes), desc="Training"):
        obs, _ = env.reset()
        episode_reward = 0.0
        episode_length = 0
        
        # Collect rollout
        obs_buffer = []
        actions_buffer = []
        log_probs_buffer = []
        rewards_buffer = []
        values_buffer = []
        dones_buffer = []
        
        for step in range(rollout_length):
            # Select action
            action, value, log_prob = agent.select_action(obs)
            
            # Execute in environment
            next_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            # Store transition
            obs_buffer.append(obs)
            actions_buffer.append(action)
            log_probs_buffer.append(log_prob)
            rewards_buffer.append(reward)
            values_buffer.append(value)
            dones_buffer.append(done)
            
            episode_reward += reward
            episode_length += 1
            global_step += 1
            
            obs = next_obs
            
            if done:
                # Episode complete
                break
        
        # Compute advantages with GAE
        advantages, returns = agent.compute_gae(
            rewards_buffer, values_buffer, dones_buffer
        )
        
        # PPO update
        if len(obs_buffer) > batch_size:
            metrics = agent.update(
                obs=np.array(obs_buffer),
                actions=np.array(actions_buffer),
                old_log_probs=np.array(log_probs_buffer),
                advantages=advantages,
                returns=returns,
                n_epochs=n_epochs,
                batch_size=batch_size,
            )
        else:
            metrics = {}
        
        # Logging
        episode_rewards.append(episode_reward)
        episode_best_values.append(info.get("best_value", -np.inf))
        episode_lengths.append(episode_length)
        recent_rewards.append(episode_reward)
        
        if episode % 50 == 0:
            avg_reward = np.mean(recent_rewards) if recent_rewards else 0.0
            logger.info(
                f"Episode {episode}/{n_episodes} | "
                f"Reward: {episode_reward:.2f} | "
                f"Avg (100): {avg_reward:.2f} | "
                f"Best: {info.get('best_value', 0):.4f} | "
                f"Steps: {episode_length} | "
                f"Policy Loss: {metrics.get('policy_loss', 0):.4f}"
            )
    
    # Save model
    model_dir = Path("models")
    model_dir.mkdir(exist_ok=True)
    model_path = model_dir / "ppo_experiment_optimizer.pt"
    agent.save(str(model_path))
    logger.info(f"Model saved to {model_path}")
    
    # Plot training progress
    plot_training_progress(
        episode_rewards,
        episode_best_values,
        episode_lengths,
    )
    
    logger.info("Training complete!")
    
    return agent, env


def plot_training_progress(
    episode_rewards: list,
    episode_best_values: list,
    episode_lengths: list,
):
    """Plot and save training metrics."""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Episode rewards
    axes[0, 0].plot(episode_rewards, alpha=0.3, label='Raw')
    if len(episode_rewards) > 10:
        window = min(50, len(episode_rewards) // 10)
        smooth = np.convolve(
            episode_rewards, np.ones(window)/window, mode='valid'
        )
        axes[0, 0].plot(smooth, label=f'Smoothed ({window})')
    axes[0, 0].set_title("Episode Rewards")
    axes[0, 0].set_xlabel("Episode")
    axes[0, 0].set_ylabel("Total Reward")
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Best values found
    axes[0, 1].plot(episode_best_values, alpha=0.6)
    axes[0, 1].axhline(
        y=-0.397887, color='r', linestyle='--',
        label='Global optimum (-0.398)'
    )
    axes[0, 1].set_title("Best Value Found per Episode")
    axes[0, 1].set_xlabel("Episode")
    axes[0, 1].set_ylabel("Best Objective Value")
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Episode lengths
    axes[1, 0].plot(episode_lengths, alpha=0.6)
    axes[1, 0].set_title("Episode Lengths")
    axes[1, 0].set_xlabel("Episode")
    axes[1, 0].set_ylabel("Steps")
    axes[1, 0].grid(True, alpha=0.3)
    
    # Learning progress (cumulative best)
    cumulative_best = []
    best_so_far = -np.inf
    for val in episode_best_values:
        if val > best_so_far:
            best_so_far = val
        cumulative_best.append(best_so_far)
    
    axes[1, 1].plot(cumulative_best, linewidth=2)
    axes[1, 1].axhline(
        y=-0.397887, color='r', linestyle='--',
        label='Global optimum'
    )
    axes[1, 1].set_title("Cumulative Best Value")
    axes[1, 1].set_xlabel("Episode")
    axes[1, 1].set_ylabel("Best Value Achieved")
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plot_path = Path("training_progress.png")
    plt.savefig(plot_path, dpi=150)
    logger.info(f"Training plot saved to {plot_path}")
    plt.close()


if __name__ == "__main__":
    # Run training
    agent, env = train_ppo(
        n_episodes=500,
        max_steps_per_episode=50,
        rollout_length=1024,
        batch_size=64,
        n_epochs=10,
    )
    
    # Test trained agent
    logger.info("\nTesting trained agent...")
    obs, _ = env.reset()
    total_reward = 0.0
    
    for step in range(50):
        action, _, _ = agent.select_action(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        
        if terminated or truncated:
            break
    
    logger.info(f"Test episode:")
    logger.info(f"  Total reward: {total_reward:.2f}")
    logger.info(f"  Best value: {info['best_value']:.6f}")
    logger.info(f"  Best params: {info['best_params']}")
    logger.info(f"  Global optimum: -0.397887")

