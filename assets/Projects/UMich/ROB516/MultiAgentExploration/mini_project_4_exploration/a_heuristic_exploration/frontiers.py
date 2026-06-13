from __future__ import annotations

from typing import List, Tuple

import numpy as np

from simulation.types import CellState


def extract_frontiers(fused_map: np.ndarray) -> List[Tuple[int, int]]:
    """Return frontier cells as (x, y) coordinates.

    Frontier definition (per project spec):
    - A frontier is a FREE cell adjacent (4-neighborhood) to at least one UNKNOWN cell.

    Inputs (IMPORTANT: y,x indexing):
    - fused_map: 2D numpy array indexed as fused_map[y, x]

    Output:
    - List of (x, y) frontier coordinates

    Student TODO:
    - Implement this function.
    - Keep it deterministic (no randomness).
    """

    if fused_map.ndim != 2:
        raise ValueError(f"Expected 2D fused_map, got shape={fused_map.shape}")

    free = fused_map == int(CellState.FREE)
    unknown = fused_map == int(CellState.UNKNOWN)

    # ===== STUDENT IMPLEMENTATION BEGIN =====
    frontiers = []
    for y in range(fused_map.shape[0]):
        for x in range(fused_map.shape[1]):
            if free[y, x]:
                # Check 4-neighborhood for unknown cells
                neighbors = [
                    (x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1)
                ]
                if any(0 <= nx < fused_map.shape[1] and 0 <= ny < fused_map.shape[0] and unknown[ny, nx] for nx, ny in neighbors):
                    frontiers.append((x, y))
    return frontiers
    # raise NotImplementedError("Implement frontier extraction")

    # ===== STUDENT IMPLEMENTATION END =====
