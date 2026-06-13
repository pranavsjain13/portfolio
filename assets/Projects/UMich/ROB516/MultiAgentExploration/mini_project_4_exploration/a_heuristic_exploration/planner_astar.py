from __future__ import annotations

import heapq
from typing import Dict, List, Optional, Tuple

import numpy as np

from simulation.types import CellState


def _heuristic(a: Tuple[int, int], b: Tuple[int, int]) -> int:
    # Manhattan distance
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def astar_path(
    fused_map: np.ndarray,
    start: Tuple[int, int],
    goal: Tuple[int, int],
) -> Optional[List[Tuple[int, int]]]:
    """A* shortest path on current fused_map.

    Traversable cells: FREE (and UNKNOWN is treated as blocked for safety).

    Returns:
    - List of (x, y) including start and goal, or None if no path.
    """

    height, width = int(fused_map.shape[0]), int(fused_map.shape[1])

    sx, sy = start
    gx, gy = goal

    if not (0 <= sx < width and 0 <= sy < height):
        return None
    if not (0 <= gx < width and 0 <= gy < height):
        return None

    def passable(x: int, y: int) -> bool:
        return int(fused_map[y, x]) == int(CellState.FREE)

    if not passable(sx, sy) or not passable(gx, gy):
        return None

    open_heap: List[Tuple[int, int, Tuple[int, int]]] = []
    heapq.heappush(open_heap, (0 + _heuristic(start, goal), 0, start))

    came_from: Dict[Tuple[int, int], Tuple[int, int]] = {}
    g_score: Dict[Tuple[int, int], int] = {start: 0}

    neighbors = [(1, 0), (-1, 0), (0, 1), (0, -1)]

    while open_heap:
        _, g, current = heapq.heappop(open_heap)
        if current == goal:
            # Reconstruct
            path = [current]
            while current in came_from:
                current = came_from[current]
                path.append(current)
            path.reverse()
            return path

        cx, cy = current
        for dx, dy in neighbors:
            nx, ny = cx + dx, cy + dy
            if not (0 <= nx < width and 0 <= ny < height):
                continue
            if not passable(nx, ny):
                continue
            tentative = g + 1
            npos = (nx, ny)
            if tentative < g_score.get(npos, 1_000_000_000):
                came_from[npos] = current
                g_score[npos] = tentative
                f = tentative + _heuristic(npos, goal)
                heapq.heappush(open_heap, (f, tentative, npos))

    return None
