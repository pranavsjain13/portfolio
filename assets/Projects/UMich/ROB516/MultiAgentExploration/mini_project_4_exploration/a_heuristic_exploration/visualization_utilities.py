import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from typing import List, Tuple
from PIL import Image
import io

# Import simulation components
from simulation.env import GridWorldEnv
from simulation.types import EnvConfig, CellState, Action
from simulation.render import render_ascii, render_matplotlib
from simulation.metrics import compute_coverage

# Import heuristic exploration components (students will implement these)
from a_heuristic_exploration.frontiers import extract_frontiers
from a_heuristic_exploration.information_gain import information_gain_count_unknown
from a_heuristic_exploration.planner_astar import astar_path
from a_heuristic_exploration.clustering import frontier_clusters_region_pca
from a_heuristic_exploration.assignment import greedy_assignment
from a_heuristic_exploration.policy import HeuristicFrontierPolicy
from a_heuristic_exploration.policy import HeuristicConfig

def visualize_map(fused_map, robot_positions=None, frontiers=None, title="Map Visualization", ax=None):
    """Visualize the exploration map.
    
    Args:
        fused_map: The map to visualize
        robot_positions: Optional list of robot positions
        frontiers: Optional list of frontier points
        title: Title for the plot
        ax: Optional matplotlib axes to plot on. If None, creates new figure.
    
    Returns:
        fig, ax: The figure and axes objects
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 10))
    else:
        fig = ax.figure
    
    # Create color-coded map: unknown=gray, free=white, occupied=black
    visual_map = np.zeros((*fused_map.shape, 3))
    visual_map[fused_map == int(CellState.UNKNOWN)] = [0.7, 0.7, 0.7]  # Gray
    visual_map[fused_map == int(CellState.FREE)] = [1.0, 1.0, 1.0]     # White
    visual_map[fused_map == int(CellState.OCCUPIED)] = [0.0, 0.0, 0.0] # Black
    
    ax.imshow(visual_map, origin='upper', interpolation='nearest')
    
    # Plot frontiers if provided
    if frontiers:
        fx, fy = zip(*frontiers) if frontiers else ([], [])
        ax.scatter(fx, fy, c='green', s=30, marker='s', alpha=0.6, label='Frontiers')
    
    # Plot robots if provided
    if robot_positions:
        rx, ry = zip(*robot_positions)
        ax.scatter(rx, ry, c='red', s=200, marker='o', edgecolors='black', 
                   linewidths=2, label='Robots', zorder=5)
    
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    
    if ax is None:
        plt.tight_layout()
    
    return fig, ax

def create_exploration_gif(
    config: EnvConfig,
    policy_config: HeuristicConfig,
    output_path: str = "exploration.gif",
    frame_interval: int = 5,
    duration: int = 100,
    show_frontiers: bool = True,
    coverage_threshold: float = 0.99,
    figsize: tuple = (10, 10)
):
    """
    Create an animated GIF of multi-agent exploration.
    
    Args:
        config: Environment configuration
        policy_config: Heuristic policy configuration
        output_path: Path to save the GIF file
        frame_interval: Capture a frame every N steps
        duration: Duration of each frame in milliseconds
        show_frontiers: Whether to show frontier points
        figsize: Figure size for each frame
    
    Returns:
        Path to the created GIF file
    """
    # Initialize environment and policy
    env = GridWorldEnv(config)
    obs = env.reset()
    policy = HeuristicFrontierPolicy(config=policy_config)
    policy.reset(obs)
    
    frames = []
    step = 0
    
    print(f"Generating exploration GIF with {config.num_robots} robots...")
    print(f"Capturing frames every {frame_interval} steps")
    
    # Capture initial frame
    frames.append(_capture_frame(obs, step, config, show_frontiers, figsize))
    
    # Run exploration and capture frames
    while step < config.max_steps:
        # Execute policy
        actions = policy.act(obs)
        result = env.step(actions)
        obs = env.observe()
        step += 1
        
        # Capture frame at intervals
        if step % frame_interval == 0:
            frames.append(_capture_frame(obs, step, config, show_frontiers, figsize))
            print(f"  Step {step:3d}: Coverage = {result.coverage:.2%}, Frames captured: {len(frames)}")
        
        # Stop if done or nearly complete
        if result.done or result.coverage >= coverage_threshold:
            # Capture final frame
            if step % frame_interval != 0:
                frames.append(_capture_frame(obs, step, config, show_frontiers, figsize))
            print(f"\nExploration completed at step {step}")
            print(f"Final coverage: {result.coverage:.2%}")
            break
    
    # Save as GIF
    print(f"\nSaving GIF with {len(frames)} frames to {output_path}...")
    frames[0].save(
        output_path,
        save_all=True,
        append_images=frames[1:],
        duration=duration,
        loop=0,
        optimize=False
    )
    
    print(f"✓ GIF saved successfully to {output_path}")
    print(f"  Total frames: {len(frames)}")
    print(f"  File size: {_get_file_size_mb(output_path):.2f} MB")
    
    
    return output_path

def _capture_frame(obs, step, config, show_frontiers, figsize):
    """Capture a single frame of the current exploration state."""
    fig, ax = plt.subplots(figsize=figsize)
    
    # Create color-coded map
    visual_map = np.zeros((*obs.fused_map.shape, 3))
    visual_map[obs.fused_map == int(CellState.UNKNOWN)] = [0.7, 0.7, 0.7]  # Gray
    visual_map[obs.fused_map == int(CellState.FREE)] = [1.0, 1.0, 1.0]     # White
    visual_map[obs.fused_map == int(CellState.OCCUPIED)] = [0.0, 0.0, 0.0] # Black
    
    ax.imshow(visual_map, origin='upper', interpolation='nearest')
    
    # Plot frontiers if requested
    if show_frontiers:
        frontiers = extract_frontiers(obs.fused_map)
        if frontiers:
            fx, fy = zip(*frontiers)
            ax.scatter(fx, fy, c='green', s=20, marker='s', alpha=0.5, label='Frontiers')
    
    # Plot robots
    if obs.robot_positions:
        rx, ry = zip(*obs.robot_positions)
        # Use different colors for each robot
        colors = plt.cm.Set1(np.linspace(0, 1, len(obs.robot_positions)))
        for i, (x, y) in enumerate(obs.robot_positions):
            ax.scatter(x, y, c=[colors[i]], s=200, marker='o', 
                      edgecolors='black', linewidths=2, zorder=5)
    
    # Add info text
    coverage = compute_coverage(obs.fused_map)
    ax.set_title(
        f"Step {step} | Coverage: {coverage:.1%} | Robots: {config.num_robots}",
        fontsize=14,
        fontweight='bold'
    )
    
    # Remove axes for cleaner look
    ax.set_xticks([])
    ax.set_yticks([])
    
    # Convert plot to PIL Image
    buf = io.BytesIO()
    plt.tight_layout()
    plt.savefig(buf, format='png', dpi=80, bbox_inches='tight')
    plt.close(fig)
    
    buf.seek(0)
    return Image.open(buf).copy()


def _get_file_size_mb(filepath):
    """Get file size in megabytes."""
    import os
    return os.path.getsize(filepath) / (1024 * 1024)