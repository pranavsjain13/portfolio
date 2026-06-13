"""Reward Function Components for Exploration RL

Students will implement three reward components:
1. Extrinsic: Coverage/information gain reward
2. Safety: Collision penalty
3. Intrinsic: Boredom penalty (potential-based reward shaping)
"""

import numpy as np
from typing import Tuple


class RewardCalculator:
    """Calculates reward components for the exploration task."""
    
    def __init__(self, alpha: float = 1.0, beta: float = 15.0, lambda_: float = 0.5, completion_bonus: float = 100.0):
        """
        Initialize reward calculator with hyperparameters.
        
        Args:
            alpha: Weight for extrinsic reward (information gain)
            beta: Weight for safety penalty (collision)
            lambda_: Weight for intrinsic reward (boredom/history)
            completion_bonus: Large bonus reward for reaching coverage threshold (95%)
        """
        self.alpha = alpha
        self.beta = beta
        self.lambda_ = lambda_
        self.completion_bonus = completion_bonus
        self.step_penalty = -0.2  # Small negative reward per step to encourage efficiency
    
    def calculate_extrinsic_reward(self, newly_revealed: int) -> float:
        """
        Calculate extrinsic reward based on information gain.
        
        The agent gets rewarded for discovering new cells (changing them from
        UNKNOWN to FREE or OCCUPIED).
        
        Formula: R_extrinsic = alpha * (count of newly revealed cells)
        
        Args:
            newly_revealed: Number of cells revealed in this step
            
        Returns:
            Extrinsic reward value
        """
        extrinsic_reward = None

        # ===== STUDENT IMPLEMENTATION BEGIN =====
        raise NotImplementedError("Student implementation required for extrinsic reward.")
        # ===== STUDENT IMPLEMENTATION END =====

        return extrinsic_reward

    def calculate_safety_penalty(self, collision: bool) -> float:
        """
        Calculate safety penalty for collisions.
        
        The agent is heavily penalized for attempting to move into obstacles.
        This penalty should be larger than the max 1-step exploration reward
        to ensure safety is prioritized.
        
        Formula: R_safety = -beta if collision else 0
        
        Args:
            collision: Whether a collision occurred
            
        Returns:
            Safety penalty (negative value or 0)
        """
        safety_penalty = None

        # ===== STUDENT IMPLEMENTATION BEGIN =====
        raise NotImplementedError("Student implementation required for safety penalty.")
        # ===== STUDENT IMPLEMENTATION END =====

        return safety_penalty
    
    def calculate_intrinsic_reward(self, history_value: float) -> float:
        """
        Calculate intrinsic reward based on trajectory history.
        
        This implements potential-based reward shaping to discourage the agent
        from revisiting recently explored areas (creates a "boredom" signal).
        
        The history map tracks where the agent has been, with values decaying
        over time. Higher history values mean the agent was there more recently.
        
        Formula: R_intrinsic = -lambda * history_value
        
        Args:
            history_value: Value from history map at new position (0.0 to 1.0)
            
        Returns:
            Intrinsic reward (negative penalty for high history)
        """
        intrinsic_reward = None

        # ===== STUDENT IMPLEMENTATION BEGIN =====
        raise NotImplementedError("Student implementation required for intrinsic reward.")
        # ===== STUDENT IMPLEMENTATION END =====

        return intrinsic_reward
    
    def calculate_completion_bonus(self, reached_threshold: bool) -> float:
        """
        Calculate completion bonus for reaching coverage threshold.
        
        The agent receives a large positive reward when it successfully
        completes the exploration task (reaching 95% coverage).
        
        Args:
            reached_threshold: Whether coverage threshold (95%) was reached
            
        Returns:
            Completion bonus (large positive value or 0)
        """

        completion_bonus = None

        # ===== STUDENT IMPLEMENTATION BEGIN =====
        raise NotImplementedError("Student implementation required for completion bonus.")
        # ===== STUDENT IMPLEMENTATION END =====

        return completion_bonus
    
    def calculate_total_reward(
        self, 
        newly_revealed: int, 
        collision: bool, 
        history_value: float,
        reached_threshold: bool = False
    ) -> Tuple[float, dict]:
        """
        Calculate total reward as sum of all components.
        
        Formula: R_total = R_extrinsic + R_safety + R_intrinsic + R_step + R_completion
        
        Args:
            newly_revealed: Number of cells revealed
            collision: Whether a collision occurred
            history_value: History map value at new position
            reached_threshold: Whether coverage threshold (95%) was reached
            
        Returns:
            total_reward: Sum of all reward components
            reward_dict: Dictionary with individual reward components for logging
        """
        r_extrinsic = self.calculate_extrinsic_reward(newly_revealed)
        r_safety = self.calculate_safety_penalty(collision)
        r_intrinsic = self.calculate_intrinsic_reward(history_value)
        r_completion = self.calculate_completion_bonus(reached_threshold)
        r_step = self.step_penalty
        
        total = r_extrinsic + r_safety + r_intrinsic + r_completion + r_step
        
        reward_dict = {
            'total': total,
            'extrinsic': r_extrinsic,
            'safety': r_safety,
            'intrinsic': r_intrinsic,
            'completion': r_completion,
            'step': r_step,
        }
        
        return total, reward_dict
