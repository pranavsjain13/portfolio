from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

from simulation.types import Action, CellState, Observation

from .assignment import greedy_assignment, hungarian_assignment
from .clustering import (
    cluster_representative_point,
    frontier_clusters_region_pca,
    kmeans_centroids,
)
from .connectivity import group_robots_by_component, label_free_components
from .frontiers import extract_frontiers
from .information_gain import information_gain_count_unknown
from .planner_astar import astar_path


def _next_action_from_path(path: List[Tuple[int, int]]) -> Action:
    if len(path) < 2:
        return Action.STAY
    (x0, y0), (x1, y1) = path[0], path[1]
    dx, dy = x1 - x0, y1 - y0
    if dx == 1 and dy == 0:
        return Action.RIGHT
    if dx == -1 and dy == 0:
        return Action.LEFT
    if dx == 0 and dy == 1:
        return Action.DOWN
    if dx == 0 and dy == -1:
        return Action.UP
    return Action.STAY


@dataclass
class HeuristicConfig:
    alpha: float = 1.0
    beta: float = 100.00
    ig_radius: int = 4
    replanning_period: int = 10
    assignment_method: str = 'hungarian'  # 'hungarian' or 'greedy'

    # Frontier clustering controls (region-growing + PCA splitting)
    min_frontier_cluster_size: int = 8
    frontier_pca_eigen_threshold: float = 25.0
    max_targets_per_component: int = 40


class HeuristicFrontierPolicy:
    """Frontier-based exploration policy scaffold.

    This class is mostly glue. Students implement the core algorithms in the
    designated TODO modules.
    """

    def __init__(self, config: HeuristicConfig | None = None):
        self.config = config or HeuristicConfig()
        self._paths: Dict[int, List[Tuple[int, int]]] = {}
        self._targets: Dict[int, Tuple[int, int]] = {}

        self.assign_frontiers = hungarian_assignment if self.config.assignment_method == 'hungarian' else greedy_assignment

        # Debug overlays (used by simulation.runner when enabled)
        self._debug_frontier_clusters: List[List[Tuple[int, int]]] = []
        self._debug_targets: List[Tuple[int, int]] = []

    def reset(self, obs: Observation) -> None:
        self._paths.clear()
        self._targets.clear()
        self._debug_frontier_clusters.clear()
        self._debug_targets.clear()

    def act(self, obs: Observation) -> List[Action]:
        fused_map = obs.fused_map
        robot_positions = obs.robot_positions

        # Periodic replanning, and also replan if we have no path.
        should_replan = (obs.step_index % self.config.replanning_period == 0)

        if should_replan or any(i not in self._paths or len(self._paths[i]) <= 1 for i in range(len(robot_positions))):
            self._replan(fused_map, robot_positions)

        actions: List[Action] = []
        for i, pos in enumerate(robot_positions):
            path = self._paths.get(i)
            if not path:
                actions.append(Action.STAY)
                continue

            # Path should start at current position; if not, drop it and replan next step.
            if path[0] != pos:
                self._paths.pop(i, None)
                actions.append(Action.STAY)
                continue

            actions.append(_next_action_from_path(path))

            # Advance cached path by one step optimistically.
            if len(path) >= 2:
                self._paths[i] = path[1:]

        return actions

    def _replan(self, fused_map: np.ndarray, robot_positions: List[Tuple[int, int]]) -> None:
        frontiers = extract_frontiers(fused_map)
        if not frontiers:
            self._paths.clear()
            self._targets.clear()
            self._debug_frontier_clusters.clear()
            self._debug_targets.clear()
            return

        # Keep a snapshot of previous targets so we can avoid oscillation.
        prev_targets = dict(self._targets)

        # Label connected components of currently-known FREE space.
        labels = label_free_components(fused_map)
        robot_groups = group_robots_by_component(labels, robot_positions)

        # Group frontier cells by component.
        frontiers_by_comp: Dict[int, List[Tuple[int, int]]] = {}
        for (fx, fy) in frontiers:
            cid = int(labels[fy, fx])
            if cid < 0:
                continue
            frontiers_by_comp.setdefault(cid, []).append((fx, fy))

        new_paths: Dict[int, List[Tuple[int, int]]] = {}
        new_targets: Dict[int, Tuple[int, int]] = {}

        debug_clusters_all: List[List[Tuple[int, int]]] = []
        debug_targets_all: List[Tuple[int, int]] = []

        # Plan independently within each component so robots aren't assigned unreachable targets.
        for cid, robot_ids in robot_groups.items():
            component_frontiers = frontiers_by_comp.get(cid, [])
            if not component_frontiers:
                continue

            # Cluster ALL frontier cells in this component for plotting/validation.
            clusters_all = frontier_clusters_region_pca(
                component_frontiers,
                min_cluster_size=self.config.min_frontier_cluster_size,
                eigenvalue_threshold=self.config.frontier_pca_eigen_threshold,
            )
            debug_clusters_all.extend([list(c) for c in clusters_all])

            frontier_set = set(component_frontiers)

            # First, try to preserve existing robot targets to prevent oscillation.
            preserved_targets: set[Tuple[int, int]] = set()
            robots_to_assign: List[int] = []
            for robot_i in robot_ids:
                pos = robot_positions[robot_i]
                prev_t = prev_targets.get(robot_i)

                # Preserve only if:
                # - target exists
                # - target is still a frontier in this component
                # - robot hasn't already reached it
                if prev_t is not None and prev_t in frontier_set and pos != prev_t:
                    path = astar_path(fused_map, start=pos, goal=prev_t)
                    if path is not None:
                        new_targets[robot_i] = prev_t
                        new_paths[robot_i] = path
                        preserved_targets.add(prev_t)
                        continue

                robots_to_assign.append(robot_i)

            remaining_frontiers = [f for f in component_frontiers if f not in preserved_targets]
            if not robots_to_assign or not remaining_frontiers:
                continue

            # Region-growing frontier clusters, then recursively split large/elongated clusters using PCA.
            clusters = frontier_clusters_region_pca(
                remaining_frontiers,
                min_cluster_size=self.config.min_frontier_cluster_size,
                eigenvalue_threshold=self.config.frontier_pca_eigen_threshold,
            )

            # Fall back to k-means if clustering produced nothing (e.g., too few frontiers).
            # if clusters:
            targets = [cluster_representative_point(c) for c in clusters]
            # else:
            #     k = min(len(robots_to_assign), len(remaining_frontiers))
            #     targets = kmeans_centroids(remaining_frontiers, k=k, seed=0)

            debug_targets_all.extend(list(targets))

            if not targets:
                continue

            # Cap targets for performance (A* is run for each robot-target pair).
            if len(targets) > self.config.max_targets_per_component:
                # Prefer targets with larger immediate information gain.
                scored = [
                    (
                        information_gain_count_unknown(fused_map, target=t, radius=self.config.ig_radius),
                        t,
                    )
                    for t in targets
                ]
                scored.sort(key=lambda it: it[0], reverse=True)
                targets = [t for _, t in scored[: self.config.max_targets_per_component]]

            # Compute utilities for robots in this component only.
            n_robots_sub = len(robots_to_assign)
            n_targets_sub = len(targets)
            utilities_sub = np.full((n_robots_sub, n_targets_sub), -np.inf)
            paths_sub: List[List[Optional[List[Tuple[int, int]]]]] = []

            for local_idx, robot_i in enumerate(robots_to_assign):
                start = robot_positions[robot_i]
                row_paths: List[Optional[List[Tuple[int, int]]]] = []
                for t_idx, t in enumerate(targets):
                    path = astar_path(fused_map, start=start, goal=t)
                    row_paths.append(path)
                    if path is None:
                        # Unreachable => -inf utility (already set)
                        continue

                    cost = max(0, len(path) - 1)
                    ig = information_gain_count_unknown(fused_map, target=t, radius=self.config.ig_radius)
                    utilities_sub[local_idx, t_idx] = self.config.alpha * float(ig) - self.config.beta * float(cost)
                paths_sub.append(row_paths)

            # assignment = hungarian_assignment(utilities_sub)
            assignment = self.assign_frontiers(utilities_sub)

            for local_robot_idx, target_j in assignment.robot_to_target.items():
                robot_i = robots_to_assign[local_robot_idx]
                path = paths_sub[local_robot_idx][target_j]
                if path is None:
                    continue

                new_targets[robot_i] = targets[target_j]
                new_paths[robot_i] = path

        # Commit replanning results.
        self._targets = new_targets
        self._paths = new_paths

        # Commit debug overlays.
        self._debug_frontier_clusters = debug_clusters_all
        self._debug_targets = debug_targets_all
