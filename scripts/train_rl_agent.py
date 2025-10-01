"""
Training script for RL agent on experiment planning.

Trains a PPO agent to learn optimal experiment sequences
that maximize Expected Information Gain per unit time.
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import logging
import argparse
from datetime import datetime
from pathlib import Path
import matplotlib.pyplot as plt
import json

from src.reasoning.rl_env import ExperimentGym, branin_function, rastrigin_function
from src.reasoning.rl_agent import PPOAgent, RLConfig

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def train_agent(
    env: ExperimentGym,
    agent: PPOAgent,
    n_episodes: int = 1000,
    eval_interval: int = 50,
    save_dir: str = "checkpoints"
):
    """
    Train RL agent on experiment planning task.
    
    Args:
        env: Experiment gym environment
        agent: PPO agent
        n_episodes: Number of training episodes
        eval_interval: Episodes between evaluations
        save_dir: Directory to save checkpoints
    """
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    
    # Training metrics
    episode_rewards = []
    episode_lengths = []
    eval_rewards = []
    best_eval_reward = -np.inf
    
    logger.info(f"Starting training for {n_episodes} episodes...")
    
    for episode in range(n_episodes):
        obs = env.reset()
        episode_reward = 0
        episode_length = 0
        done = False
        
        # Collect trajectory
        while not done:
            # Select action
            action = agent.select_action(obs, deterministic=False)
            
            # Step environment
            next_obs, reward, done, info = env.step(action)
            
            # Store transition
            agent.store_transition(obs, action, reward, done, next_obs)
            
            episode_reward += reward
            episode_length += 1
            obs = next_obs
        
        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)
        
        # Update policy
        if agent.ready_to_update():
            metrics = agent.update()
            
            if episode % 10 == 0:
                logger.info(
                    f"Episode {episode}: "
                    f"reward={episode_reward:.2f}, "
                    f"length={episode_length}, "
                    f"policy_loss={metrics.get('policy_loss', 0):.4f}"
                )
        
        # Evaluation
        if episode > 0 and episode % eval_interval == 0:
            eval_reward = evaluate_agent(env, agent, n_eval_episodes=5)
            eval_rewards.append(eval_reward)
            
            logger.info(f"Evaluation at episode {episode}: avg_reward={eval_reward:.2f}")
            
            # Save best model
            if eval_reward > best_eval_reward:
                best_eval_reward = eval_reward
                agent.save(os.path.join(save_dir, "best_agent.pth"))
                logger.info(f"New best model saved! Eval reward: {eval_reward:.2f}")
        
        # Periodic checkpoint
        if episode > 0 and episode % 100 == 0:
            agent.save(os.path.join(save_dir, f"agent_ep{episode}.pth"))
    
    # Final save
    agent.save(os.path.join(save_dir, "final_agent.pth"))
    
    # Save training metrics
    metrics_data = {
        "episode_rewards": episode_rewards,
        "episode_lengths": episode_lengths,
        "eval_rewards": eval_rewards,
        "best_eval_reward": best_eval_reward
    }
    
    with open(os.path.join(save_dir, "training_metrics.json"), "w") as f:
        json.dump(metrics_data, f, indent=2)
    
    # Plot results
    plot_training_results(episode_rewards, eval_rewards, save_dir)
    
    logger.info(f"Training complete! Best eval reward: {best_eval_reward:.2f}")
    return episode_rewards, eval_rewards


def evaluate_agent(
    env: ExperimentGym,
    agent: PPOAgent,
    n_eval_episodes: int = 10
) -> float:
    """
    Evaluate agent on multiple episodes.
    
    Returns:
        Average episode reward
    """
    eval_rewards = []
    
    for _ in range(n_eval_episodes):
        obs = env.reset()
        episode_reward = 0
        done = False
        
        while not done:
            action = agent.select_action(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
            episode_reward += reward
        
        eval_rewards.append(episode_reward)
    
    return np.mean(eval_rewards)


def plot_training_results(
    episode_rewards: list,
    eval_rewards: list,
    save_dir: str
):
    """Plot training progress."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Episode rewards
    ax1.plot(episode_rewards, alpha=0.3, label="Episode Reward")
    
    # Moving average
    window = 50
    if len(episode_rewards) >= window:
        moving_avg = np.convolve(
            episode_rewards,
            np.ones(window) / window,
            mode='valid'
        )
        ax1.plot(range(window-1, len(episode_rewards)), moving_avg, label=f"Moving Avg ({window})")
    
    ax1.set_xlabel("Episode")
    ax1.set_ylabel("Reward")
    ax1.set_title("Training Progress")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Evaluation rewards
    if len(eval_rewards) > 0:
        eval_episodes = np.arange(0, len(episode_rewards), len(episode_rewards) // len(eval_rewards))[:len(eval_rewards)]
        ax2.plot(eval_episodes, eval_rewards, marker='o', label="Eval Reward")
        ax2.set_xlabel("Episode")
        ax2.set_ylabel("Average Eval Reward")
        ax2.set_title("Evaluation Progress")
        ax2.legend()
        ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "training_progress.png"), dpi=150)
    logger.info(f"Training plot saved to {save_dir}/training_progress.png")


def main():
    parser = argparse.ArgumentParser(description="Train RL agent for experiment planning")
    parser.add_argument("--objective", type=str, default="branin", choices=["branin", "rastrigin"],
                        help="Objective function to optimize")
    parser.add_argument("--episodes", type=int, default=1000, help="Number of training episodes")
    parser.add_argument("--max-experiments", type=int, default=20, help="Max experiments per episode")
    parser.add_argument("--time-budget", type=float, default=24.0, help="Time budget in hours")
    parser.add_argument("--learning-rate", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--save-dir", type=str, default="checkpoints", help="Directory to save checkpoints")
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"], help="Device to use")
    
    args = parser.parse_args()
    
    # Create save directory with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = os.path.join(args.save_dir, f"{args.objective}_{timestamp}")
    
    # Select objective function
    if args.objective == "branin":
        objective_fn = branin_function
        param_bounds = {"param1": (0.0, 1.0), "param2": (0.0, 1.0)}
    elif args.objective == "rastrigin":
        objective_fn = rastrigin_function
        param_bounds = {"param1": (-5.12, 5.12), "param2": (-5.12, 5.12)}
    else:
        raise ValueError(f"Unknown objective: {args.objective}")
    
    # Create environment
    env = ExperimentGym(
        objective_function=objective_fn,
        experiment_types=["xrd", "nmr", "uvvis"],
        max_experiments=args.max_experiments,
        time_budget_hours=args.time_budget,
        param_bounds=param_bounds
    )
    
    # Create agent
    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    
    config = RLConfig(learning_rate=args.learning_rate)
    agent = PPOAgent(obs_dim, action_dim, config, device=args.device)
    
    # Train
    logger.info(f"Environment: {args.objective}")
    logger.info(f"Observation dim: {obs_dim}, Action dim: {action_dim}")
    logger.info(f"Device: {args.device}")
    logger.info(f"Checkpoints will be saved to: {save_dir}")
    
    episode_rewards, eval_rewards = train_agent(
        env,
        agent,
        n_episodes=args.episodes,
        eval_interval=50,
        save_dir=save_dir
    )
    
    # Test final agent
    logger.info("\nTesting final agent...")
    final_reward = evaluate_agent(env, agent, n_eval_episodes=10)
    logger.info(f"Final agent average reward: {final_reward:.2f}")
    
    # Visualize final episode
    logger.info("\nRunning final visualization episode...")
    obs = env.reset()
    done = False
    step = 0
    
    while not done:
        action = agent.select_action(obs, deterministic=True)
        obs, reward, done, info = env.step(action)
        logger.info(f"Step {step}: reward={reward:.3f}, value={info.get('value', 0):.3f}, "
                   f"best={info.get('best_value', 0):.3f}")
        step += 1
    
    env.render()


if __name__ == "__main__":
    main()

