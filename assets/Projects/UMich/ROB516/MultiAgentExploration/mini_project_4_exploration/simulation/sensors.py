from __future__ import annotations

from typing import List, Tuple

import numpy as np

from .types import CellState


def reveal_within_radius(
    truth: np.ndarray,
    fused_map: np.ndarray,
    robot_positions: List[Tuple[int, int]],
    radius: int,
) -> int:
    """Reveal cells within Chebyshev radius around each robot into fused_map.

    Returns: number of cells that changed from UNKNOWN to known.
    """

    height, width = int(truth.shape[0]), int(truth.shape[1])

    changed = 0
    unknown = int(CellState.UNKNOWN)
    for (rx, ry) in robot_positions:
        x0 = max(0, rx - radius)
        x1 = min(width - 1, rx + radius)
        y0 = max(0, ry - radius)
        y1 = min(height - 1, ry + radius)
        fused_view = fused_map[y0 : y1 + 1, x0 : x1 + 1]
        truth_view = truth[y0 : y1 + 1, x0 : x1 + 1]
        mask = fused_view == unknown
        if mask.any():
            changed += int(mask.sum())
            fused_view[mask] = truth_view[mask]
    return changed
