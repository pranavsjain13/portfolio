from __future__ import annotations

from typing import Iterable, List, Optional, Tuple

import numpy as np

from .types import CellState

def _to_rgb_image(fused_map: np.ndarray) -> np.ndarray:
    """Convert int-encoded fused_map into an RGB image.

    Colors:
    - UNKNOWN: light gray
    - FREE: white
    - OCCUPIED: black
    """

    h, w = int(fused_map.shape[0]), int(fused_map.shape[1])
    img = np.empty((h, w, 3), dtype=np.float32)
    img[:] = (0.75, 0.75, 0.75)

    free = fused_map == int(CellState.FREE)
    occ = fused_map == int(CellState.OCCUPIED)
    img[free] = (1.0, 1.0, 1.0)
    img[occ] = (0.0, 0.0, 0.0)
    return img


class MatplotlibRenderer:
    """Matplotlib renderer that supports fast updates.

    Usage:
        r = MatplotlibRenderer()
        r.update(fused_map, robot_positions, title="Step 10")
    """

    def __init__(self) -> None:
        self._initialized = False
        self._fig = None
        self._ax = None
        self._im = None
        self._robots = None
        self._cluster_artists: list[object] = []
        self._target_artist: object | None = None

    def update(
        self,
        fused_map: np.ndarray,
        robot_positions: List[Tuple[int, int]],
        title: str | None = None,
        frontier_clusters: Optional[List[List[Tuple[int, int]]]] = None,
        targets: Optional[List[Tuple[int, int]]] = None,
    ) -> None:
        import matplotlib.pyplot as plt
        import matplotlib.cm as cm

        rgb = _to_rgb_image(fused_map)
        xs = [p[0] for p in robot_positions]
        ys = [p[1] for p in robot_positions]

        if not self._initialized:
            self._fig, self._ax = plt.subplots(figsize=(6, 6))
            self._im = self._ax.imshow(rgb, interpolation="nearest", origin="upper")
            (self._robots,) = self._ax.plot(xs, ys, "ro", markersize=4)
            self._ax.set_xticks([])
            self._ax.set_yticks([])
            self._initialized = True
        else:
            assert self._im is not None and self._robots is not None
            self._im.set_data(rgb)
            self._robots.set_data(xs, ys)

        if title and self._ax is not None:
            self._ax.set_title(title)

        # Optional overlays
        if self._ax is not None:
            # Clear previous overlay artists
            for a in self._cluster_artists:
                try:
                    a.remove()
                except Exception:
                    pass
            self._cluster_artists.clear()

            if self._target_artist is not None:
                try:
                    self._target_artist.remove()
                except Exception:
                    pass
                self._target_artist = None

            if frontier_clusters:
                cmap = cm.get_cmap("tab20", max(1, len(frontier_clusters)))
                for idx, cluster in enumerate(frontier_clusters):
                    if not cluster:
                        continue
                    cx = [p[0] for p in cluster]
                    cy = [p[1] for p in cluster]
                    artist = self._ax.scatter(
                        cx,
                        cy,
                        s=8,
                        marker="s",
                        c=[cmap(idx)],
                        alpha=0.85,
                        linewidths=0,
                    )
                    self._cluster_artists.append(artist)

            if targets:
                tx = [p[0] for p in targets]
                ty = [p[1] for p in targets]
                self._target_artist = self._ax.scatter(
                    tx,
                    ty,
                    s=30,
                    marker="x",
                    c="yellow",
                    linewidths=1.5,
                )

        # Draw without blocking; works in notebooks and interactive sessions.
        plt.pause(0.001)


def render_ascii(
    fused_map: np.ndarray,
    robot_positions: List[Tuple[int, int]],
) -> str:
    """Simple ASCII renderer for quick debugging.

    Legend: ? unknown, . free, # occupied, 0..9 robots
    """

    height, width = int(fused_map.shape[0]), int(fused_map.shape[1])

    grid = [["?" for _ in range(width)] for _ in range(height)]
    for y in range(height):
        for x in range(width):
            cell = int(fused_map[y, x])
            if cell == int(CellState.UNKNOWN):
                grid[y][x] = "?"
            elif cell == int(CellState.FREE):
                grid[y][x] = "."
            else:
                grid[y][x] = "#"

    for idx, (x, y) in enumerate(robot_positions):
        if 0 <= y < height and 0 <= x < width:
            grid[y][x] = str(idx % 10)

    return "\n".join("".join(row) for row in grid)


def render_matplotlib(
    fused_map: np.ndarray,
    robot_positions: List[Tuple[int, int]],
    title: str | None = None,
):
    """One-shot matplotlib render.

    Returns (fig, ax) for embedding in notebooks.
    """

    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(_to_rgb_image(fused_map), interpolation="nearest", origin="upper")
    ax.plot([p[0] for p in robot_positions], [p[1] for p in robot_positions], "ro", markersize=4)
    ax.set_xticks([])
    ax.set_yticks([])
    if title:
        ax.set_title(title)
    return fig, ax
