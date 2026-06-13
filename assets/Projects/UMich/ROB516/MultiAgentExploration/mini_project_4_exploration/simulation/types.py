from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Tuple

import numpy as np


class CellState(int, Enum):
    UNKNOWN = -1
    FREE = 0
    OCCUPIED = 1


class Action(int, Enum):
    UP = 0
    DOWN = 1
    LEFT = 2
    RIGHT = 3
    STAY = 4


ACTION_DELTAS: Dict[Action, Tuple[int, int]] = {
    Action.UP: (0, -1),
    Action.DOWN: (0, 1),
    Action.LEFT: (-1, 0),
    Action.RIGHT: (1, 0),
    Action.STAY: (0, 0),
}


@dataclass(frozen=True)
class EnvConfig:
    width: int = 32
    height: int = 32
    num_robots: int = 3
    obstacle_prob: float = 0.18
    sensor_radius: int = 4
    max_steps: int = 400
    seed: int = 0

    # Collision rules
    allow_robot_overlap: bool = False


@dataclass
class RobotState:
    x: int
    y: int
    distance_traveled: int = 0
    collisions: int = 0


@dataclass
class StepResult:
    step_index: int
    actions: List[Action]
    robot_positions: List[Tuple[int, int]]
    newly_revealed: int
    collisions: int
    coverage: float
    done: bool


@dataclass(frozen=True)
class Observation:
    """Observation returned to a policy.

    For the heuristic method we expose global fused map + robot poses.
    RL will wrap this env with a local observation builder.

    When produced by ``GridWorldEnv.observe_robot(idx)`` or
    ``GridWorldEnv.observe_all()``, ``robot_map`` holds the individual
    robot's private knowledge map (only cells *it* has sensed), while
    ``fused_map`` is the union of all robots' maps.  ``robot_positions``
    is filtered to teammates visible within sensor radius.

    The legacy ``GridWorldEnv.observe()`` sets ``robot_map`` to ``None``
    and returns all robot positions.
    """

    # 2D grid [y, x] containing int-encoded cell states: -1 unknown, 0 free, 1 occupied
    fused_map: "np.ndarray"
    robot_positions: List[Tuple[int, int]]
    step_index: int
    max_steps: int
    # Per-robot private map; None when using the legacy observe() call.
    robot_map: Optional["np.ndarray"] = None
