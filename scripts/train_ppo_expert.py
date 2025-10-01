#!/usr/bin/env python3
"""
Expert PPO Training with Intrinsic Curiosity Module (ICM).

Improvements based on latest research (2025):
- Curiosity-driven exploration for sparse rewards
- Adaptive learning rate scheduling
- Improved normalization techniques
- Better logging and visualization
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import torch
import logging
from tqdm import tqdm
import matplotlib.pyplot as plt
from collections import deque

from src.reasoning.rl_env import ExperimentOptimizationEnv
from src.reasoning.ppo_agent import PPOAgent
from src.reasoning.curiosity_module import IntrinsicCuriosityModule

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def branin_function(params: np.ndarray) -> float:
    """Branin test function (negated for maximization)."""
    x, y = params[0], params[1]
    a, b, c, r, s, t = 1.0, 5.1/(4*np.pi**2), 5.0/np.pi, 6.0, 10.0, 1.0/(8*np.pi)
    return -(a*(y - b*x**2 + c*x - r)**2 + s*(1-t)*np.cos(x) + s)


def train_ppo_expert(
    n_episodes: int = 500,
    max_steps_per_episode: int = 50,
    use_curiosity: bool = True,
    curiosity_weight: float = 0.1,
    device: str = "auto",
):
    """
    Expert-level PPO training with ICM.
    """
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"🚀 Expert Training Mode: {'ENABLED' if use_curiosity else 'DISABLED'}")
    logger.info(f"Using device: {device}")
    
    # Environment
    param_bounds = {"x": (-5.0, 10.0), "y": (0.0, 15.0)}
    env = ExperimentOptimizationEnv(
        parameter_bounds=param_bounds,
        objective_function=branin_function,
        max_experiments=max_steps_per_episode,
        time_budget_hours=max_steps_per_episode * 1.0,
        noise_std=0.1,
    )
    
    # Agent
    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    agent = PPOAgent(
        obs_dim=obs_dim,
        action_dim=action_dim,
        lr=3e-4,
        gamma=0.99,
        device=device,
    )
    
    # Curiosity module (NEW!)
    icm = None
    if use_curiosity:
        icm = IntrinsicCuriosityModule(
            state_dim=obs_dim,
            action_dim=action_dim,
            curiosity_weight=curiosity_weight,
            device=device,
        )
        logger.info(f"✅ ICM enabled with curiosity_weight={curiosity_weight}")
    
    # Training metrics
    episode_rewards = []
    episode_best_values = []
    recent_rewards = deque(maxlen=100)
    curiosity_values = []
    
    logger.info("Starting expert training...")
    logger.info(f"Episodes: {n_episodes}, Environment: Branin function")
    
    for episode in tqdm(range(n_episodes), desc="Training"):
        obs, _ = env.reset()
        episode_reward = 0.0
        
        # Buffers
        obs_buffer, actions_buffer, log_probs_buffer = [], [], []
        rewards_buffer, values_buffer, dones_buffer = [], [], []
        next_obs_buffer = []
        
        for step in range(max_steps_per_episode):
            action, value, log_prob = agent.select_action(obs)
            next_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            # Add intrinsic reward (curiosity)
            intrinsic_reward = 0.0
            if icm:
                intrinsic_reward = icm.compute_intrinsic_reward(obs, action, next_obs)
                reward += intrinsic_reward  # Augment extrinsic reward
            
            # Store transition
            obs_buffer.append(obs)
            actions_buffer.append(action)
            log_probs_buffer.append(log_prob)
            rewards_buffer.append(reward)
            values_buffer.append(value)
            dones_buffer.append(done)
            next_obs_buffer.append(next_obs)
            
            episode_reward += reward
            obs = next_obs
            
            if done:
                break
        
        # Update ICM
        if icm and len(obs_buffer) > 1:
            icm_loss = icm.update(
                np.array(obs_buffer),
                np.array(actions_buffer),
                np.array(next_obs_buffer),
            )
        
        # PPO update
        if len(obs_buffer) > 32:
            advantages, returns = agent.compute_gae(
                rewards_buffer, values_buffer, dones_buffer
            )
            metrics = agent.update(
                obs=np.array(obs_buffer),
                actions=np.array(actions_buffer),
                old_log_probs=np.array(log_probs_buffer),
                advantages=advantages,
                returns=returns,
                n_epochs=10,
                batch_size=64,
            )
        else:
            metrics = {}
        
        # Logging
        episode_rewards.append(episode_reward)
        episode_best_values.append(info.get("best_value", -np.inf))
        recent_rewards.append(episode_reward)
        
        if icm:
            curiosity_values.append(icm.get_avg_curiosity())
        
        if episode % 50 == 0:
            avg_reward = np.mean(recent_rewards) if recent_rewards else 0.0
            curiosity_str = f"| Curiosity: {curiosity_values[-1]:.4f}" if curiosity_values else ""
            logger.info(
                f"Ep {episode}/{n_episodes} | "
                f"Reward: {episode_reward:.2f} | "
                f"Avg: {avg_reward:.2f} | "
                f"Best: {info['best_value']:.4f} "
                f"{curiosity_str}"
            )
    
    # Save model
    model_dir = Path("models")
    model_dir.mkdir(exist_ok=True)
    model_path = model_dir / "ppo_expert_curiosity.pt"
    agent.save(str(model_path))
    logger.info(f"✅ Model saved to {model_path}")
    
    # Plot
    plot_expert_training(
        episode_rewards, episode_best_values, curiosity_values
    )
    
    logger.info("🎉 Expert training complete!")
    return agent, env


def plot_expert_training(rewards, best_values, curiosity_values):
    """Enhanced plotting with curiosity metrics."""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.patch.set_facecolor('#0f172a')
    
    for ax in axes.flat:
        ax.set_facecolor('#1e293b')
        ax.spines['bottom'].set_color('white')
        ax.spines['top'].set_color('white')
        ax.spines['right'].set_color('white')
        ax.spines['left'].set_color('white')
        ax.tick_params(colors='white')
        ax.xaxis.label.set_color('white')
        ax.yaxis.label.set_color('white')
        ax.title.set_color('white')
    
    # Rewards
    axes[0, 0].plot(rewards, alpha=0.3, color='#22c55e', label='Raw')
    if len(rewards) > 10:
        window = 50
        smooth = np.convolve(rewards, np.ones(window)/window, mode='valid')
        axes[0, 0].plot(smooth, color='#22c55e', linewidth=2, label='Smoothed')
    axes[0, 0].set_title("Episode Rewards")
    axes[0, 0].set_xlabel("Episode")
    axes[0, 0].set_ylabel("Reward")
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.2, color='white')
    
    # Best values
    axes[0, 1].plot(best_values, alpha=0.6, color='#a855f7')
    axes[0, 1].axhline(y=-0.397887, color='#f87171', linestyle='--', label='Global opt')
    axes[0, 1].set_title("Best Value per Episode")
    axes[0, 1].set_xlabel("Episode")
    axes[0, 1].set_ylabel("Best Value")
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.2, color='white')
    
    # Cumulative best
    cumulative_best = []
    best_so_far = -np.inf
    for val in best_values:
        if val > best_so_far:
            best_so_far = val
        cumulative_best.append(best_so_far)
    axes[1, 0].plot(cumulative_best, linewidth=2, color='#3b82f6')
    axes[1, 0].axhline(y=-0.397887, color='#f87171', linestyle='--', label='Global opt')
    axes[1, 0].set_title("Cumulative Best")
    axes[1, 0].set_xlabel("Episode")
    axes[1, 0].set_ylabel("Best Achieved")
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.2, color='white')
    
    # Curiosity
    if curiosity_values:
        axes[1, 1].plot(curiosity_values, color='#eab308', linewidth=2)
        axes[1, 1].set_title("Intrinsic Curiosity")
        axes[1, 1].set_xlabel("Episode")
        axes[1, 1].set_ylabel("Curiosity Reward")
        axes[1, 1].grid(True, alpha=0.2, color='white')
    else:
        axes[1, 1].text(0.5, 0.5, 'No Curiosity Data', 
                       ha='center', va='center', color='white')
        axes[1, 1].set_xticks([])
        axes[1, 1].set_yticks([])
    
    plt.tight_layout()
    plt.savefig("expert_training_progress.png", dpi=150, facecolor='#0f172a')
    logger.info("📊 Plot saved to expert_training_progress.png")
    plt.close()


if __name__ == "__main__":
    agent, env = train_ppo_expert(
        n_episodes=500,
        use_curiosity=True,
        curiosity_weight=0.1,
    )
    
    # Test
    logger.info("\n🧪 Testing trained agent...")
    obs, _ = env.reset()
    total_reward = 0.0
    
    for step in range(50):
        action, _, _ = agent.select_action(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        if terminated or truncated:
            break
    
    logger.info(f"Test results:")
    logger.info(f"  Total reward: {total_reward:.2f}")
    logger.info(f"  Best value: {info['best_value']:.6f}")
    logger.info(f"  Global optimum: -0.397887")
    logger.info(f"  Gap: {abs(info['best_value'] - (-0.397887)):.6f}")

