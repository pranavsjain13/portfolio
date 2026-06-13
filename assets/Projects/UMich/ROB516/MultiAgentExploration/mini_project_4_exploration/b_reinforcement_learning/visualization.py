"""Visualization utilities for RL exploration"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Rectangle
from typing import List, Tuple, Optional
import os

from simulation.types import CellState


def visualize_episode(
    snapshots: List[Tuple[np.ndarray, Tuple[int, int]]],
    title: str = "Episode Exploration",
    save_path: Optional[str] = None,
):
    """
    Visualize an episode as a sequence of snapshots.
    
    Args:
        snapshots: List of (fused_map, robot_pos) tuples
        title: Plot title
        save_path: Optional path to save figure
    """
    n_snapshots = len(snapshots)
    cols = min(5, n_snapshots)
    rows = (n_snapshots + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(3*cols, 3*rows))
    if rows == 1:
        axes = axes.reshape(1, -1) if cols > 1 else np.array([[axes]])
    
    for idx, (fused_map, robot_pos) in enumerate(snapshots):
        row = idx // cols
        col = idx % cols
        ax = axes[row, col] if rows > 1 or cols > 1 else axes[0, 0]
        
        # Create RGB visualization
        h, w = fused_map.shape
        img = np.zeros((h, w, 3), dtype=np.uint8)
        img[fused_map == int(CellState.UNKNOWN)] = [128, 128, 128]  # Gray
        img[fused_map == int(CellState.FREE)] = [255, 255, 255]  # White
        img[fused_map == int(CellState.OCCUPIED)] = [0, 0, 0]  # Black
        
        # Draw robot
        if robot_pos is not None:
            img[robot_pos[1], robot_pos[0]] = [255, 0, 0]  # Red
        
        ax.imshow(img, interpolation='nearest')
        ax.set_title(f'Step {idx}', fontsize=10)
        ax.axis('off')
    
    # Hide empty subplots
    for idx in range(n_snapshots, rows * cols):
        row = idx // cols
        col = idx % cols
        ax = axes[row, col] if rows > 1 or cols > 1 else axes[0, 0]
        ax.axis('off')
    
    plt.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Figure saved to {save_path}")
    
    plt.show()


def create_gif_from_episode(
    env,
    model,
    level: int = 0,
    max_steps: int = 200,
    output_path: str = "exploration.gif",
    fps: int = 10,
):
    """
    Create an animated GIF of an agent exploring.
    
    Args:
        env: ExplorationEnv instance
        model: Trained PPO model
        level: Curriculum level
        max_steps: Maximum steps to record
        output_path: Path to save GIF
        fps: Frames per second
    """
    try:
        import imageio
    except ImportError:
        print("imageio not installed. Run: pip install imageio")
        return None
    
    # Run episode and collect frames
    obs, info = env.reset(options={'level': level})
    frames = []
    
    for step in range(max_steps):
        # Render current state
        maps = env.get_maps()
        fused_map = maps['fused_map']
        robot_pos = env.robot_pos
        
        # Create RGB frame
        h, w = fused_map.shape
        frame = np.zeros((h, w, 3), dtype=np.uint8)
        frame[fused_map == int(CellState.UNKNOWN)] = [128, 128, 128]
        frame[fused_map == int(CellState.FREE)] = [255, 255, 255]
        frame[fused_map == int(CellState.OCCUPIED)] = [0, 0, 0]
        frame[robot_pos[1], robot_pos[0]] = [255, 0, 0]
        
        # Upscale for better visibility
        frame = np.repeat(np.repeat(frame, 4, axis=0), 4, axis=1)
        frames.append(frame)
        
        # Take action
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        
        if terminated or truncated:
            break
    
    # Save GIF
    imageio.mimsave(output_path, frames, fps=fps)
    print(f"GIF saved to {output_path}")
    
    return output_path


def plot_observation_channels(obs: np.ndarray, save_path: Optional[str] = None):
    """
    Visualize the 3 channels of an observation.
    
    Args:
        obs: Observation tensor (3, H, W)
        save_path: Optional path to save figure
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    channel_names = ['Obstacles', 'Exploration (Unknown)', 'History']
    cmaps = ['gray', 'Blues', 'hot']
    
    for i in range(3):
        ax = axes[i]
        im = ax.imshow(obs[i], cmap=cmaps[i], interpolation='nearest')
        ax.set_title(f'Channel {i}: {channel_names[i]}', fontsize=14, fontweight='bold')
        ax.axis('off')
        plt.colorbar(im, ax=ax, fraction=0.046)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Figure saved to {save_path}")
    
    plt.show()


def compare_maps(truth, fused_map, robot_pos, title="Map Comparison"):
    """
    Compare ground truth map with agent's knowledge.
    
    Args:
        truth: Ground truth map
        fused_map: Agent's knowledge map
        robot_pos: Robot position (x, y)
        title: Plot title
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Ground truth
    h, w = truth.shape
    truth_img = np.zeros((h, w, 3), dtype=np.uint8)
    truth_img[truth == int(CellState.FREE)] = [255, 255, 255]
    truth_img[truth == int(CellState.OCCUPIED)] = [0, 0, 0]
    truth_img[robot_pos[1], robot_pos[0]] = [255, 0, 0]
    
    axes[0].imshow(truth_img, interpolation='nearest')
    axes[0].set_title('Ground Truth', fontsize=14, fontweight='bold')
    axes[0].axis('off')
    
    # Agent's knowledge
    fused_img = np.zeros((h, w, 3), dtype=np.uint8)
    fused_img[fused_map == int(CellState.UNKNOWN)] = [128, 128, 128]
    fused_img[fused_map == int(CellState.FREE)] = [255, 255, 255]
    fused_img[fused_map == int(CellState.OCCUPIED)] = [0, 0, 0]
    fused_img[robot_pos[1], robot_pos[0]] = [255, 0, 0]
    
    axes[1].imshow(fused_img, interpolation='nearest')
    axes[1].set_title("Agent's Knowledge", fontsize=14, fontweight='bold')
    axes[1].axis('off')
    
    plt.suptitle(title, fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.show()
