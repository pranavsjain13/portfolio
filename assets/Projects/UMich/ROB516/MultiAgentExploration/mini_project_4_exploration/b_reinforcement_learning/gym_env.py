"""Gymnasium-Compatible Single-Agent Exploration Environment

This wraps the simulation environment to provide:
1. Egocentric observations (3-channel local view)
2. Single-agent control
3. Reward shaping with history tracking
4. Curriculum learning support
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
from typing import Optional, Tuple, Dict, Any

from simulation.types import CellState, Action, ACTION_DELTAS
from .rewards import RewardCalculator
from .curriculum import CurriculumMapGenerator


class ExplorationEnv(gym.Env):
    """
    Gymnasium environment for single-agent exploration with egocentric observations.
    
    Observation Space: (2*radius+1, 2*radius+1, 5) tensor (channels-last)
        - Channel 0: Obstacles (1 = occupied, 0 = free/unknown)
        - Channel 1: Exploration (1 = unknown, 0 = known)
        - Channel 2: Trajectory history (0.0 to 1.0, recent visits have higher values)
        - Channel 3: Consecutive collisions (0.0 to 1.0, normalized by /10, capped at 1.0)
        - Channel 4: Frontier distances (0.0 = at frontier, 1.0 = far from frontier)
    
    Action Space: Discrete(4)
        - 0: UP, 1: DOWN, 2: LEFT, 3: RIGHT
    
    The environment maintains:
        - truth: Ground truth map (hidden from agent)
        - fused_map: Agent's knowledge map (starts all UNKNOWN)
        - history_map: Tracks where agent has been (for intrinsic reward)
        - consecutive_collisions: Counter for consecutive collision actions
    """
    
    metadata = {'render_modes': ['rgb_array', 'ascii']}
    
    def __init__(
        self, 
        size: int = 20, 
        sensor_radius: int = 5,
        history_decay: float = 0.05,
        alpha: float = 1.0,
        beta: float = 5.0,
        lambda_: float = 0.5,
        completion_bonus: float = 100.0,        default_level: int = 0,        seed: Optional[int] = None
    ):
        """
        Initialize the exploration environment.
        
        Args:
            size: Grid size (size x size)
            sensor_radius: Radius of egocentric observation window
            history_decay: Decay rate for history map per step
            alpha: Extrinsic reward weight
            beta: Safety penalty weight
            lambda_: Intrinsic reward weight
            completion_bonus: Large bonus for reaching coverage threshold
            seed: Random seed
        """
        super().__init__()
        
        self.size = size
        self.sensor_radius = sensor_radius
        self.history_decay = history_decay
        self.seed_value = seed
        self.rng = np.random.default_rng(seed)
        
        # Reward calculator
        self.reward_calc = RewardCalculator(alpha=alpha, beta=beta, lambda_=lambda_, completion_bonus=completion_bonus)
        
        # Map generator for curriculum
        self.map_generator = CurriculumMapGenerator(size=size, seed=seed)
        
        # Action space: 4 discrete actions (no STAY)
        self.action_space = spaces.Discrete(4)
        
        # Observation space: local window with 5 channels (channels-last for SB3 CnnPolicy)
        # Channel 0: Obstacles, Channel 1: Unknown, Channel 2: History, 
        # Channel 3: Collision indicator, Channel 4: Frontier distance
        obs_size = 2 * sensor_radius + 1
        self.observation_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(obs_size, obs_size, 5),  # (height, width, channels) - 5 channels
            dtype=np.float32
        )
        
        # State variables (initialized in reset)
        self.truth: Optional[np.ndarray] = None
        self.fused_map: Optional[np.ndarray] = None
        self.history_map: Optional[np.ndarray] = None
        self.robot_pos: Optional[Tuple[int, int]] = None
        self.step_count: int = 0
        self.default_level: int = default_level  # Store default curriculum level
        self.current_level: int = default_level
        self.episode_info: Dict[str, Any] = {}
        self.consecutive_collisions: int = 0  # Track consecutive collisions for observation
    
    def reset(
        self, 
        seed: Optional[int] = None, 
        options: Optional[Dict[str, Any]] = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Reset the environment for a new episode.
        
        Args:
            seed: Random seed
            options: Dict that may contain 'level' key for curriculum learning
            
        Returns:
            observation: Initial egocentric observation
            info: Additional information dictionary
        """
        super().reset(seed=seed)
        
        # IMPORTANT: Generate new random seed for each episode to get different maps!
        # This ensures the agent doesn't overfit to specific map layouts
        if seed is None:
            # Use a random seed for each reset (different maps each episode)
            seed = self.rng.integers(0, 2**31 - 1)
        
        self.rng = np.random.default_rng(seed)
        self.map_generator = CurriculumMapGenerator(size=self.size, seed=seed)
        
        # Get curriculum level from options (use default_level if not specified)
        self.current_level = self.default_level
        if options is not None and 'level' in options:
            self.current_level = options['level']
        
        # Generate map based on curriculum level
        self.truth, self.robot_pos = self.map_generator.generate(self.current_level)
        
        # Initialize fused map (all unknown)
        self.fused_map = np.full((self.size, self.size), int(CellState.UNKNOWN), dtype=np.int8)
        
        # Initialize history map (all zeros)
        self.history_map = np.zeros((self.size, self.size), dtype=np.float32)
        
        # Perform initial sensing
        self._sense_environment()
        
        # Mark current position in history
        self.history_map[self.robot_pos[1], self.robot_pos[0]] = 1.0
        
        self.step_count = 0
        self.consecutive_collisions = 0  # Reset collision counter
        self.episode_info = {
            'total_reward': 0.0,
            'extrinsic_reward': 0.0,
            'safety_penalty': 0.0,
            'intrinsic_reward': 0.0,
            'collisions': 0,
            'cells_explored': 0,
        }
        
        observation = self._get_observation()
        
        # Calculate initial coverage
        initial_coverage = self._calculate_coverage()
        
        info = {
            'level': self.current_level,
            'coverage': initial_coverage,
            'collision': False,
            'newly_revealed': 0,
            'step': 0,
        }
        
        return observation, info
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """
        Execute one step in the environment.
        
        Args:
            action: Integer action (0=UP, 1=DOWN, 2=LEFT, 3=RIGHT)
            
        Returns:
            observation: Next egocentric observation
            reward: Reward for this step
            terminated: Whether episode is complete (coverage >= 95%)
            truncated: Whether episode exceeded max steps
            info: Additional information
        """
        # Map action to Action enum (0-3 map directly to UP, DOWN, LEFT, RIGHT)
        action_enum = Action(action)
        dx, dy = ACTION_DELTAS[action_enum]
        
        # Calculate intended position
        intended_x = self.robot_pos[0] + dx
        intended_y = self.robot_pos[1] + dy
        
        # Sample history value at INTENDED position BEFORE collision detection
        # This ensures the agent is penalized for trying to revisit areas,
        # not for staying in its current position after a collision
        if self._in_bounds(intended_x, intended_y):
            history_value = self.history_map[intended_y, intended_x]
        else:
            history_value = 0.0  # Out of bounds has no history penalty
        
        # Check for collisions and update actual new position
        collision = False
        new_x, new_y = intended_x, intended_y
        if not self._in_bounds(new_x, new_y):
            new_x, new_y = self.robot_pos  # Stay in place
            collision = True
        elif self.truth[new_y, new_x] == int(CellState.OCCUPIED):
            # Collision with obstacle
            new_x, new_y = self.robot_pos  # Stay in place
            collision = True
        
        # Update position
        old_pos = self.robot_pos
        self.robot_pos = (new_x, new_y)
        
        # Sense environment and count newly revealed cells
        newly_revealed = self._sense_environment()
        
        # Decay history map
        self.history_map = np.maximum(0.0, self.history_map - self.history_decay)
        
        # Update history at new position ONLY if agent actually moved
        # Don't update history if agent collided and stayed in place
        if not collision:
            self.history_map[new_y, new_x] = 1.0
        
        # Check if coverage threshold reached
        coverage = self._calculate_coverage()
        reached_threshold = coverage >= 0.95
        
        # Calculate reward
        reward, reward_dict = self.reward_calc.calculate_total_reward(
            newly_revealed=newly_revealed,
            collision=collision,
            history_value=history_value,
            reached_threshold=reached_threshold
        )
        
        # Update episode info
        self.episode_info['total_reward'] += reward_dict['total']
        self.episode_info['extrinsic_reward'] += reward_dict['extrinsic']
        self.episode_info['safety_penalty'] += reward_dict['safety']
        self.episode_info['intrinsic_reward'] += reward_dict['intrinsic']
        if collision:
            self.episode_info['collisions'] += 1
        self.episode_info['cells_explored'] += newly_revealed
        
        self.step_count += 1
        
        # Update consecutive collision counter
        if collision:
            self.consecutive_collisions += 1
        else:
            self.consecutive_collisions = 0  # Reset on successful move
        
        # Get new observation (includes collision counter)
        observation = self._get_observation()
        
        # Check termination conditions (coverage already calculated above)
        terminated = reached_threshold
        truncated = self.step_count >= 500  # Max steps per episode
        
        # Info dictionary
        info = {
            'coverage': coverage,
            'collision': collision,
            'newly_revealed': newly_revealed,
            'reward_components': reward_dict,
            'step': self.step_count,
        }
        
        # Add episode info if done
        # Store custom metrics separately from 'episode' key since VecMonitor will overwrite it
        if terminated or truncated:
            info['terminal_observation'] = observation
            # Store episode stats with a different key that VecMonitor won't touch
            info['custom_episode_stats'] = {
                'total_reward': self.episode_info['total_reward'],
                'collisions': self.episode_info['collisions'],
                'cells_explored': self.episode_info['cells_explored'],
                'extrinsic_reward': self.episode_info['extrinsic_reward'],
                'safety_penalty': self.episode_info['safety_penalty'],
                'intrinsic_reward': self.episode_info['intrinsic_reward'],
            }
        
        return observation, float(reward), terminated, truncated, info
    
    def _in_bounds(self, x: int, y: int) -> bool:
        """Check if position is within grid bounds."""
        return 0 <= x < self.size and 0 <= y < self.size
    
    def _get_frontier_distances(self) -> np.ndarray:
        """
        Calculate distance from each cell to nearest frontier using BFS wavefront expansion.
        Frontier = known free cells adjacent to unknown cells.
        
        This is O(n²) instead of O(n² × num_frontiers) by using multi-source BFS.
        
        Returns:
            Array of normalized distances (0.0 = at frontier, 1.0 = far from frontier)
        """
        from collections import deque
        
        distances = np.full((self.size, self.size), -1, dtype=np.int32)
        queue = deque()
        
        # Find all frontier cells and initialize BFS queue
        for y in range(self.size):
            for x in range(self.size):
                if self.fused_map[y, x] == int(CellState.FREE):
                    # Check if adjacent to unknown
                    for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                        nx, ny = x + dx, y + dy
                        if self._in_bounds(nx, ny) and self.fused_map[ny, nx] == int(CellState.UNKNOWN):
                            distances[y, x] = 0
                            queue.append((x, y))
                            break
        
        if not queue:
            # No frontiers = return all ones (far from frontier)
            return np.ones((self.size, self.size), dtype=np.float32)
        
        # BFS wavefront expansion from all frontiers simultaneously
        # ONLY propagate through FREE cells (known navigable space)
        max_dist = 0
        while queue:
            x, y = queue.popleft()
            current_dist = distances[y, x]
            
            for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                nx, ny = x + dx, y + dy
                # Only expand through FREE cells, not UNKNOWN or OCCUPIED
                if (self._in_bounds(nx, ny) and 
                    distances[ny, nx] == -1 and 
                    self.fused_map[ny, nx] == int(CellState.FREE)):
                    distances[ny, nx] = current_dist + 1
                    max_dist = max(max_dist, current_dist + 1)
                    queue.append((nx, ny))
        
        # Convert to float and normalize
        distances_float = distances.astype(np.float32)
        if max_dist > 0:
            distances_float = np.where(distances_float == -1, 1.0, distances_float / max_dist)
        else:
            distances_float = np.where(distances_float == -1, 1.0, 0.0)
        
        return distances_float
    
    def _sense_environment(self) -> int:
        """
        Directly update self.fused_map based on robot's sensor within self.sensor_radius. Make sure to use manhattan distance.
        
        Returns:
            Number of cells newly revealed
        """

        newly_revealed = None
        
        # ===== STUDENT IMPLEMENTATION BEGIN =====
        raise NotImplementedError("Student implementation required for sensing environment.")

        # ===== STUDENT IMPLEMENTATION END =====

        return newly_revealed
    
    def _get_observation(self) -> np.ndarray:
        """
        Construct egocentric observation tensor.
        
        Returns:
            (2*radius+1, 2*radius+1, 5) float32 tensor (channels-last for SB3):
                - Channel 0: Obstacles (1 = occupied, 0 = free/unknown)
                - Channel 1: Exploration (1 = unknown, 0 = known)
                - Channel 2: History (0.0 to 1.0)
                - Channel 3: Consecutive collisions (0.0 to 1.0, normalized)
                - Channel 4: Frontier distances (0.0 to 1.0)
        """
        obstacles, exploration, history, collision_indicator, frontier_distances = None, None, None, None, None



        rx, ry = self.robot_pos
        obs_size = 2 * self.sensor_radius + 1
        
        # Initialize channels
        obstacles = np.zeros((obs_size, obs_size), dtype=np.float32)
        exploration = np.zeros((obs_size, obs_size), dtype=np.float32)
        history = np.zeros((obs_size, obs_size), dtype=np.float32)

        # ===== STUDENT IMPLEMENTATION BEGIN =====
        raise NotImplementedError("Student implementation required for completion bonus.")

        # ===== STUDENT IMPLEMENTATION END =====
        
        # Stack channels (channels-last for Stable-Baselines3 CnnPolicy)
        observation = np.stack([obstacles, exploration, history, collision_indicator, frontier_distances], 
                               axis=-1).astype(np.float32)

        
        return observation
    
    def _calculate_coverage(self) -> float:
        """Calculate exploration coverage percentage."""
        total_free = np.sum(self.truth == int(CellState.FREE))
        explored_free = np.sum(
            (self.fused_map == int(CellState.FREE)) & 
            (self.truth == int(CellState.FREE))
        )
        if total_free == 0:
            return 1.0
        return float(explored_free) / float(total_free)
    
    def render(self, mode: str = 'ascii') -> Optional[np.ndarray]:
        """Render the environment."""
        if mode == 'ascii':
            self._render_ascii()
            return None
        elif mode == 'rgb_array':
            return self._render_rgb()
        else:
            raise ValueError(f"Unsupported render mode: {mode}")
    
    def _render_ascii(self):
        """Print ASCII representation of the environment."""
        print(f"\nStep {self.step_count} | Coverage: {self._calculate_coverage():.1%}")
        print("=" * (self.size + 2))
        
        for y in range(self.size):
            row = "|"
            for x in range(self.size):
                if (x, y) == self.robot_pos:
                    row += "R"
                elif self.fused_map[y, x] == int(CellState.UNKNOWN):
                    row += "?"
                elif self.fused_map[y, x] == int(CellState.OCCUPIED):
                    row += "#"
                else:
                    row += "."
            row += "|"
            print(row)
        
        print("=" * (self.size + 2))
    
    def _render_rgb(self) -> np.ndarray:
        """Render RGB image of the environment."""
        # Create RGB image (placeholder implementation)
        img = np.zeros((self.size, self.size, 3), dtype=np.uint8)
        
        # Color mapping
        for y in range(self.size):
            for x in range(self.size):
                if self.fused_map[y, x] == int(CellState.UNKNOWN):
                    img[y, x] = [128, 128, 128]  # Gray for unknown
                elif self.fused_map[y, x] == int(CellState.OCCUPIED):
                    img[y, x] = [0, 0, 0]  # Black for obstacles
                else:
                    img[y, x] = [255, 255, 255]  # White for free
        
        # Draw robot
        rx, ry = self.robot_pos
        img[ry, rx] = [255, 0, 0]  # Red for robot
        
        return img
    
    def get_maps(self) -> Dict[str, np.ndarray]:
        """Return all internal maps for visualization."""
        return {
            'truth': self.truth,
            'fused_map': self.fused_map,
            'history_map': self.history_map,
        }
