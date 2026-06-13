"""Curriculum Learning: Map Generation at Different Difficulty Levels

Students will implement map generators for three difficulty levels:
- Level 0: Vacuum (no obstacles)
- Level 1: Sparse obstacles
- Level 2: Structured mazes with rooms and corridors
"""

from typing import Tuple, List
import numpy as np
from simulation.types import CellState


class CurriculumMapGenerator:
    """Generates maps at different difficulty levels for curriculum learning."""
    
    def __init__(self, size: int = 20, seed: int = 0):
        """
        Initialize the curriculum map generator.
        
        Args:
            size: Grid size (will create size x size grid)
            seed: Random seed for reproducibility
        """
        self.size = size
        self.rng = np.random.default_rng(seed)
    
    def generate_level_0_vacuum(self) -> Tuple[np.ndarray, Tuple[int, int]]:
        """
        Generate Level 0: The Vacuum
        
        An empty map with only border walls. The simplest case for sanity checking.
        The agent should learn to explore in straight lines.
        
        Returns:
            truth: (size, size) grid with int encoding
            robot_start: (x, y) starting position
        """
        # Create a grid that is all FREE except for borders which are OCCUPIED


        occ = None # (self.size, self.size) np.ndarray of CellState as int8
        robot_start = None # (x, y) starting position for the robot

        # ===== STUDENT IMPLEMENTATION BEGIN =====
        raise NotImplementedError("Student TODO: implement generate_level_0_vacuum")
        
        # ===== STUDENT IMPLEMENTATION END =====
        return occ, robot_start
    
    def generate_level_1_sparse(self, obstacle_density: float = 0.1) -> Tuple[np.ndarray, Tuple[int, int]]:
        """
        Generate Level 1: Sparse Obstacles
        
        Randomly place obstacles with given density. Agent learns collision avoidance.
        
        Args:
            obstacle_density: Probability of each cell being an obstacle (default 0.1)

        Hint: Make sure starting position is not an obstacle
            
        Returns:
            truth: (size, size) grid with int encoding
            robot_start: (x, y) starting position
        """
        occ = None # (self.size, self.size) np.ndarray of CellState as int8
        robot_start = None # (x, y) starting position for the robot

        # ===== STUDENT IMPLEMENTATION BEGIN =====
        raise NotImplementedError("Student TODO: implement generate_level_0_vacuum")

        # ===== STUDENT IMPLEMENTATION END =====
        return truth, robot_start
    
    def generate(self, level: int) -> Tuple[np.ndarray, Tuple[int, int]]:
        """
        Generate a map for the specified curriculum level.
        
        Args:
            level: 0 (vacuum), 1 (sparse), or 2 (maze)
            
        Returns:
            truth: (size, size) grid with int encoding
            robot_start: (x, y) starting position
        """
        if level == 0:
            return self.generate_level_0_vacuum()
        elif level == 1:
            return self.generate_level_1_sparse()
        else:
            raise ValueError(f"Unknown level: {level}. Must be 0 or 1.")
