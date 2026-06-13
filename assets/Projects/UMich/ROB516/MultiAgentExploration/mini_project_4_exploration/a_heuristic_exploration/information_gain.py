from __future__ import annotations

from typing import Tuple

import numpy as np

from simulation.types import CellState


def information_gain_count_unknown(
    fused_map: np.ndarray,
    target: Tuple[int, int],
    radius: int,
) -> int:
    """Count-based information gain: number of UNKNOWN cells in a local box.

    Inputs:
    - fused_map: 2D numpy array indexed as fused_map[y, x]
    - target: (x, y) centroid/goal
    - radius: reveal radius used for scoring (can match sensor radius)

    Student TODO:
    - Implement IG(t) = count of UNKNOWN cells within a radius-R neighborhood.
    - Use the same neighborhood consistently (Chebyshev or Manhattan is fine).
    """

    if fused_map.ndim != 2:
        raise ValueError(f"Expected 2D fused_map, got shape={fused_map.shape}")
    if radius < 0:
        return 0

    # ===== STUDENT IMPLEMENTATION BEGIN =====
    count = 0
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            x = target[0] + dx
            y = target[1] + dy
            if 0 <= x < fused_map.shape[1] and 0 <= y < fused_map.shape[0]:
                if fused_map[y, x] == CellState.UNKNOWN:
                    count += 1
    return count
    # raise NotImplementedError("Implement count-based information gain")
    # ===== STUDENT IMPLEMENTATION END =====
