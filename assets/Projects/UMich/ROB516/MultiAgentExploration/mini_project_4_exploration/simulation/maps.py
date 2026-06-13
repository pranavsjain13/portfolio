from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np

from .types import CellState


@dataclass(frozen=True)
class GeneratedMap:
    truth: np.ndarray  # int-encoded grid [y, x]
    robot_starts: List[Tuple[int, int]]

def generate_random_map(
    width: int,
    height: int,
    num_robots: int,
    obstacle_prob: float,
    rng: np.random.Generator,
) -> GeneratedMap:
    """Generate a random occupancy grid with guaranteed free start positions."""

    # Sample obstacles, then enforce boundaries as occupied to reduce edge weirdness.
    occ = rng.random((height, width)) < obstacle_prob
    occ[0, :] = True
    occ[-1, :] = True
    occ[:, 0] = True
    occ[:, -1] = True

    truth = np.where(occ, int(CellState.OCCUPIED), int(CellState.FREE)).astype(np.int8)

    free_cells = np.argwhere(truth == int(CellState.FREE))
    if free_cells.shape[0] < num_robots:
        # If too dense, retry with lower obstacle rate.
        return generate_random_map(width, height, num_robots, max(0.05, obstacle_prob * 0.7), rng)

    rng.shuffle(free_cells)
    starts = [(int(xy[1]), int(xy[0])) for xy in free_cells[:num_robots]]  # (x,y)

    return GeneratedMap(truth=truth, robot_starts=starts)
