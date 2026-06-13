from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np

from .types import CellState, RobotState


def compute_coverage(fused_map: np.ndarray) -> float:
    if fused_map.size == 0:
        return 0.0
    unknown = int(CellState.UNKNOWN)
    return float(np.mean(fused_map != unknown))


@dataclass
class EpisodeStats:
    steps: int
    final_coverage: float
    total_distance: int
    total_collisions: int


def summarize_episode(
    robots: List[RobotState],
    fused_map: np.ndarray,
    steps: int,
) -> EpisodeStats:
    total_distance = sum(r.distance_traveled for r in robots)
    total_collisions = sum(r.collisions for r in robots)
    return EpisodeStats(
        steps=steps,
        final_coverage=compute_coverage(fused_map),
        total_distance=int(total_distance),
        total_collisions=int(total_collisions),
    )
