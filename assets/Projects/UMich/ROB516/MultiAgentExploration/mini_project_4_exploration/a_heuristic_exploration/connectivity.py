from __future__ import annotations

from collections import deque
from typing import Dict, List, Tuple

import numpy as np

from simulation.types import CellState


def label_free_components(fused_map: np.ndarray) -> np.ndarray:
    """Label connected components of FREE cells.

    Returns:
    - labels: int32 array shaped (H,W)
      - labels[y, x] == -1 if cell is not FREE
      - otherwise a non-negative component id

    Connectivity: 4-neighborhood.
    """

    if fused_map.ndim != 2:
        raise ValueError(f"Expected 2D fused_map, got shape={fused_map.shape}")

    h, w = int(fused_map.shape[0]), int(fused_map.shape[1])
    labels = np.full((h, w), -1, dtype=np.int32)

    free = fused_map == int(CellState.FREE)

    comp_id = 0
    q: deque[Tuple[int, int]] = deque()

    for y in range(h):
        for x in range(w):
            if not free[y, x] or labels[y, x] != -1:
                continue

            labels[y, x] = comp_id
            q.append((x, y))

            while q:
                cx, cy = q.popleft()
                for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nx, ny = cx + dx, cy + dy
                    if 0 <= nx < w and 0 <= ny < h and free[ny, nx] and labels[ny, nx] == -1:
                        labels[ny, nx] = comp_id
                        q.append((nx, ny))

            comp_id += 1

    return labels


def group_robots_by_component(
    labels: np.ndarray,
    robot_positions: List[Tuple[int, int]],
) -> Dict[int, List[int]]:
    """Return mapping component_id -> list of robot indices in that component.

    Robots not on a FREE cell are excluded.
    """

    h, w = int(labels.shape[0]), int(labels.shape[1])
    groups: Dict[int, List[int]] = {}
    for i, (x, y) in enumerate(robot_positions):
        if not (0 <= x < w and 0 <= y < h):
            continue
        cid = int(labels[y, x])
        if cid < 0:
            continue
        groups.setdefault(cid, []).append(i)
    return groups
