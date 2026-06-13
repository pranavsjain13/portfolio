"""Training Script for PPO Agent with Curriculum Learning

Students will use this to train their RL agent across curriculum levels.
"""

import os
from typing import Optional
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback
from stable_baselines3.common.vec_env import VecMonitor

from .gym_env import ExplorationEnv
from .custom_cnn import create_policy_kwargs


class CoverageCallback(BaseCallback):
    """
    Custom callback to log coverage and reward components during training.
    """
    
    def __init__(self, verbose: int = 0):
        super().__init__(verbose)
        self.episode_rewards = []
        self.episode_coverages = []
        self.episode_lengths = []
        self.episode_collisions = []
    
    def _on_step(self) -> bool:
        # Check if episode is done
        for i, done in enumerate(self.locals['dones']):
            if done:
                # Get info from the environment
                info = self.locals['infos'][i]
                
                # Get custom episode stats (stored separately from 'episode' key to avoid VecMonitor overwrite)
                if 'custom_episode_stats' in info:
                    stats = info['custom_episode_stats']
                    self.episode_rewards.append(stats.get('total_reward', 0))
                    self.episode_collisions.append(stats.get('collisions', 0))
                
                # VecMonitor provides episode length in its 'episode' dict
                if 'episode' in info:
                    self.episode_lengths.append(info['episode'].get('l', 0))
                
                # Coverage is in the regular info dict
                if 'coverage' in info:
                    self.episode_coverages.append(info['coverage'])
        
        return True
    
    def get_statistics(self):
        """Return training statistics."""
        return {
            'rewards': self.episode_rewards,
            'coverages': self.episode_coverages,
            'lengths': self.episode_lengths,
            'collisions': self.episode_collisions,
        }


def train_curriculum_level(
    level: int,
    timesteps: int,
    model_path: Optional[str] = None,
    save_path: str = "./models",
    env_size: int = 20,
    n_envs: int = 4,
    learning_rate: float = 3e-4,
    verbose: int = 1,
) -> tuple[PPO, CoverageCallback]:
    """
    Train PPO agent on a specific curriculum level.
    
    Args:
        level: Curriculum level (0, 1, or 2)
        timesteps: Number of training timesteps
        model_path: Path to load pretrained model (for continuing training)
        save_path: Directory to save trained model
        env_size: Grid size
        n_envs: Number of parallel environments
        learning_rate: Learning rate for PPO optimizer (default: 3e-4)
        verbose: Verbosity level
        
    Returns:
        model: Trained PPO model
        callback: Callback with training statistics
    """
    print(f"\n{'='*60}")
    print(f"Training on Level {level}")
    print(f"{'='*60}")
    
    # Create vectorized environments
    def make_env():
        return ExplorationEnv(
            size=env_size,
            sensor_radius=5,
            history_decay=0.05,
            alpha=1.0,
            beta=5.0,
            lambda_=0.5,
            default_level=level,  # Set curriculum level at creation
        )
    
    # Create parallel environments
    env = make_vec_env(make_env, n_envs=n_envs)
    
    # Wrap with monitor for episode statistics
    env = VecMonitor(env)
    
    # Create or load model
    if model_path and os.path.exists(model_path):
        print(f"Loading model from {model_path}")
        model = PPO.load(model_path, env=env)
        # Update learning rate if loading existing model
        # Need to update optimizer param groups directly
        model.learning_rate = learning_rate
        for param_group in model.policy.optimizer.param_groups:
            param_group['lr'] = learning_rate
        print(f"Updated learning rate to {learning_rate}")
    else:
        print("Creating new PPO model")
        # Use custom CNN designed for small (11x11) observations
        policy_kwargs = create_policy_kwargs(features_dim=128)
        
        model = PPO(
            "CnnPolicy",  # CNN policy for image-like observations
            env,
            learning_rate=learning_rate,
            n_steps=2048,
            batch_size=64,
            n_epochs=10,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=0.01,  # Encourage exploration
            vf_coef=0.5,
            max_grad_norm=0.5,
            verbose=verbose,
            tensorboard_log=f"./tensorboard/level_{level}/",
            policy_kwargs=policy_kwargs,
        )
    
    # Create callback
    callback = CoverageCallback()
    
    # Train the model
    print(f"Training for {timesteps:,} timesteps...")
    model.learn(
        total_timesteps=timesteps,
        callback=callback,
        progress_bar=True,
    )
    
    # Save the model
    os.makedirs(save_path, exist_ok=True)
    model_save_path = os.path.join(save_path, f"ppo_level_{level}.zip")
    model.save(model_save_path)
    print(f"Model saved to {model_save_path}")
    
    return model, callback


def continue_training(
    model_path: str,
    level: int,
    additional_timesteps: int,
    save_path: Optional[str] = None,
    env_size: int = 20,
    n_envs: int = 4,
    learning_rate: float = 3e-4,
    verbose: int = 1,
) -> tuple[PPO, CoverageCallback]:
    """
    Continue training an existing model on a specific level.
    
    This is a convenience wrapper around train_curriculum_level that makes it
    easier to continue training existing models.
    
    Args:
        model_path: Path to the model to continue training (e.g., "./models/ppo_level_0.zip")
        level: Curriculum level to train on (0, 1, or 2)
        additional_timesteps: Number of additional training timesteps
        save_path: Where to save the updated model (if None, overwrites original)
        env_size: Grid size
        n_envs: Number of parallel environments
        learning_rate: Learning rate for PPO optimizer (default: 3e-4)
        verbose: Verbosity level
        
    Returns:
        model: Updated PPO model
        callback: Callback with training statistics
        
    Example:
        # Continue training level 1 model for 50k more steps
        model, callback = continue_training(
            model_path="./models_full/ppo_level_1.zip",
            level=1,
            additional_timesteps=50_000,
            learning_rate=1e-4,  # Optional: Use lower learning rate for fine-tuning
        )
    """
    # If no save path specified, use the directory of the original model
    if save_path is None:
        save_path = os.path.dirname(model_path)
        if not save_path:
            save_path = "."
    
    print(f"\n{'='*60}")
    print(f"Continuing training from: {model_path}")
    print(f"Level: {level}, Additional steps: {additional_timesteps:,}")
    print(f"Will save to: {save_path}")
    print(f"{'='*60}\n")
    
    # Use train_curriculum_level with model_path
    return train_curriculum_level(
        level=level,
        timesteps=additional_timesteps,
        model_path=model_path,
        save_path=save_path,
        env_size=env_size,
        n_envs=n_envs,
        learning_rate=learning_rate,
        verbose=verbose,
    )


def train_from_scratch(
    level: int,
    timesteps: int,
    save_dir: str = "./models",
    env_size: int = 20,
    n_envs: int = 4,
    learning_rate: float = 3e-4,
    verbose: int = 1,
) -> tuple[PPO, CoverageCallback]:
    """
    Train a new model from scratch on a specific level.
    
    This is a convenience wrapper that clearly indicates training from scratch.
    
    Args:
        level: Curriculum level to train on (0, 1, or 2)
        timesteps: Number of training timesteps
        save_dir: Directory to save the trained model
        env_size: Grid size
        n_envs: Number of parallel environments
        learning_rate: Learning rate for PPO optimizer (default: 3e-4)
        verbose: Verbosity level
        
    Returns:
        model: Trained PPO model
        callback: Callback with training statistics
        
    Example:
        # Train a new level 0 model for 100k steps
        model, callback = train_from_scratch(
            level=0,
            timesteps=100_000,
            save_dir="./my_models",
            learning_rate=5e-4,  # Optional: Use higher learning rate
        )
    """
    print(f"\n{'='*60}")
    print(f"Training NEW model from scratch")
    print(f"Level: {level}, Timesteps: {timesteps:,}")
    print(f"{'='*60}\n")
    
    return train_curriculum_level(
        level=level,
        timesteps=timesteps,
        model_path=None,  # No pretrained model
        save_path=save_dir,
        env_size=env_size,
        n_envs=n_envs,
        learning_rate=learning_rate,
        verbose=verbose,
    )


def plot_training_curves(callbacks: dict, save_path: Optional[str] = None):
    """
    Plot training curves across curriculum levels.
    
    Args:
        callbacks: Dictionary mapping level -> CoverageCallback
        save_path: Optional path to save plot
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    colors = ['blue', 'orange', 'green']
    
    for level, callback in callbacks.items():
        stats = callback.get_statistics()
        color = colors[level]
        label = f"Level {level}"
        
        # Plot rewards
        if stats['rewards']:
            axes[0, 0].plot(stats['rewards'], color=color, alpha=0.6, label=label)
            # Smooth with moving average
            if len(stats['rewards']) > 10:
                window = min(50, len(stats['rewards']) // 10)
                smoothed = np.convolve(stats['rewards'], np.ones(window)/window, mode='valid')
                axes[0, 0].plot(smoothed, color=color, linewidth=2)
        
        # Plot coverage
        if stats['coverages']:
            axes[0, 1].plot(stats['coverages'], color=color, alpha=0.6, label=label)
            if len(stats['coverages']) > 10:
                window = min(50, len(stats['coverages']) // 10)
                smoothed = np.convolve(stats['coverages'], np.ones(window)/window, mode='valid')
                axes[0, 1].plot(smoothed, color=color, linewidth=2)
        
        # Plot episode lengths
        if stats['lengths']:
            axes[1, 0].plot(stats['lengths'], color=color, alpha=0.6, label=label)
            if len(stats['lengths']) > 10:
                window = min(50, len(stats['lengths']) // 10)
                smoothed = np.convolve(stats['lengths'], np.ones(window)/window, mode='valid')
                axes[1, 0].plot(smoothed, color=color, linewidth=2)
        
        # Plot collisions
        if stats['collisions']:
            axes[1, 1].plot(stats['collisions'], color=color, alpha=0.6, label=label)
            if len(stats['collisions']) > 10:
                window = min(50, len(stats['collisions']) // 10)
                smoothed = np.convolve(stats['collisions'], np.ones(window)/window, mode='valid')
                axes[1, 1].plot(smoothed, color=color, linewidth=2)
    
    axes[0, 0].set_title('Episode Rewards', fontsize=14, fontweight='bold')
    axes[0, 0].set_xlabel('Episode')
    axes[0, 0].set_ylabel('Total Reward')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    axes[0, 1].set_title('Episode Coverage', fontsize=14, fontweight='bold')
    axes[0, 1].set_xlabel('Episode')
    axes[0, 1].set_ylabel('Coverage %')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    axes[1, 0].set_title('Episode Lengths', fontsize=14, fontweight='bold')
    axes[1, 0].set_xlabel('Episode')
    axes[1, 0].set_ylabel('Steps')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    axes[1, 1].set_title('Episode Collisions', fontsize=14, fontweight='bold')
    axes[1, 1].set_xlabel('Episode')
    axes[1, 1].set_ylabel('Collisions')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"Training curves saved to {save_path}")
    
    plt.show()


def full_curriculum_training(
    level_0_steps: int = 50_000,
    level_1_steps: int = 100_000,
    env_size: int = 20,
    save_dir: str = "./models",
    learning_rate: float = 3e-4,
):
    """
    Execute full curriculum training pipeline.
    
    Args:
        level_0_steps: Training steps for level 0
        level_1_steps: Training steps for level 1
        env_size: Grid size
        save_dir: Directory to save models
        learning_rate: Learning rate for PPO optimizer (default: 3e-4)
    """
    callbacks = {}
    
    # Level 0: Vacuum
    print("\n🎯 CURRICULUM STAGE 1: The Vacuum")
    model_0, callback_0 = train_curriculum_level(
        level=0,
        timesteps=level_0_steps,
        model_path=None,
        save_path=save_dir,
        env_size=env_size,
        learning_rate=learning_rate,
    )
    callbacks[0] = callback_0
    
    # Level 1: Sparse Obstacles
    print("\n🎯 CURRICULUM STAGE 2: Sparse Obstacles")
    model_1, callback_1 = train_curriculum_level(
        level=1,
        timesteps=level_1_steps,
        model_path=os.path.join(save_dir, "ppo_level_0.zip"),
        save_path=save_dir,
        env_size=env_size,
        learning_rate=learning_rate,
    )
    callbacks[1] = callback_1
    
    # Plot training curves
    print("\n📊 Generating training curves...")
    plot_training_curves(callbacks, save_path=os.path.join(save_dir, "training_curves.png"))
    
    print("\n✅ Full curriculum training complete!")
    print(f"Models saved in: {save_dir}")
    
    return model_1, callbacks


def evaluate_model(
    model_path: str,
    level: int,
    n_episodes: int = 10,
    env_size: int = 20,
    render: bool = False,
):
    """
    Evaluate a trained model.
    
    Args:
        model_path: Path to saved model
        level: Curriculum level to evaluate on
        n_episodes: Number of evaluation episodes
        env_size: Grid size
        render: Whether to render episodes
    """
    print(f"\n{'='*60}")
    print(f"Evaluating model on Level {level}")
    print(f"{'='*60}")
    
    # Load model
    model = PPO.load(model_path)
    
    # Create environment
    env = ExplorationEnv(
        size=env_size,
        sensor_radius=5,
        history_decay=0.05,
        alpha=1.0,
        beta=5.0,
        lambda_=0.5,
        default_level=level,  # Set curriculum level at creation
    )
    
    # Evaluate
    episode_rewards = []
    episode_coverages = []
    episode_lengths = []
    episode_collisions = []
    
    for ep in range(n_episodes):
        obs, info = env.reset()  # Level already set in constructor
        done = False
        truncated = False
        total_reward = 0
        step_count = 0
        
        while not (done or truncated):
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, done, truncated, info = env.step(action)
            total_reward += reward
            step_count += 1
            
            if render:
                env.render('ascii')
        
        episode_rewards.append(total_reward)
        episode_coverages.append(info['coverage'])
        episode_lengths.append(step_count)
        if 'episode' in info:
            episode_collisions.append(info['episode'].get('collisions', 0))
        
        print(f"Episode {ep+1}/{n_episodes}: "
              f"Reward={total_reward:.2f}, "
              f"Coverage={info['coverage']:.2%}, "
              f"Steps={step_count}")
    
    # Print statistics
    print(f"\n📊 Evaluation Statistics:")
    print(f"  Average Reward: {np.mean(episode_rewards):.2f} ± {np.std(episode_rewards):.2f}")
    print(f"  Average Coverage: {np.mean(episode_coverages):.2%} ± {np.std(episode_coverages):.2%}")
    print(f"  Average Length: {np.mean(episode_lengths):.1f} ± {np.std(episode_lengths):.1f}")
    if episode_collisions:
        print(f"  Average Collisions: {np.mean(episode_collisions):.1f} ± {np.std(episode_collisions):.1f}")
    
    return {
        'rewards': episode_rewards,
        'coverages': episode_coverages,
        'lengths': episode_lengths,
        'collisions': episode_collisions,
    }


if __name__ == "__main__":
    # Example: Run full curriculum training
    print("Starting full curriculum training...")
    
    final_model, callbacks = full_curriculum_training(
        level_0_steps=50_000,
        level_1_steps=100_000,
        level_2_steps=200_000,
        env_size=20,
        save_dir="./models",
    )
    
    # Evaluate final model on all levels
    for level in [0, 1, 2]:
        evaluate_model(
            model_path=f"./models/ppo_level_2.zip",
            level=level,
            n_episodes=10,
            env_size=20,
        )
