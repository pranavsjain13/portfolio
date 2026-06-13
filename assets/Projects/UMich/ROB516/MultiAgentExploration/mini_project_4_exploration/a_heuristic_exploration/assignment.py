from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import math

import numpy as np
from sympy import N


@dataclass(frozen=True)
class AssignmentResult:
    """Mapping from robot index -> target index."""

    robot_to_target: Dict[int, int]


def greedy_assignment(utilities: np.ndarray) -> AssignmentResult:
    """Greedy assignment maximizing total utility.

    utilities[i][j] is the utility of assigning robot i to target j.

    Student TODO:
    - Implement a simple greedy assignment that doesn't assign the same target twice.
    - If there are fewer targets than robots, assign as many as possible.

    Returns:
    - AssignmentResult.robot_to_target mapping (may be smaller than num_robots)
    """
    if utilities is None or utilities.size == 0:
        return AssignmentResult(robot_to_target={})

    num_robots, num_targets = utilities.shape

    if num_targets == 0:
        return AssignmentResult(robot_to_target={})

    # ===== STUDENT IMPLEMENTATION BEGIN =====
    robot_to_target: Dict[int, int] = {}
    assigned_targets = set()
    for i in range(num_robots):
        best_target = None
        best_utility = -math.inf
        for j in range(num_targets):
            if j in assigned_targets:
                continue
            if np.isfinite(utilities[i, j]) and utilities[i, j] > best_utility:
                best_utility = utilities[i, j]
                best_target = j
        if best_target is not None:
            robot_to_target[i] = best_target
            assigned_targets.add(best_target)
    # raise NotImplementedError("Implement greedy assignment")

    # ===== STUDENT IMPLEMENTATION END =====
    return AssignmentResult(robot_to_target=robot_to_target)



def _hungarian_min_cost(cost: np.ndarray) -> List[int]:
    """Hungarian algorithm for rectangular matrices (min-cost assignment).

    Returns:
    - assignment: list where assignment[i] = chosen column index for row i

    Requires:
    - cost.ndim == 2 with at least one row
    """

    cost = np.asarray(cost, dtype=float)
    n, m = cost.shape

    # If columns < rows, transpose to satisfy m >= n.
    transposed = False
    if m < n:
        transposed = True
        cost = cost.T
        n, m = m, n

    u = np.zeros(n + 1)
    v = np.zeros(m + 1)
    p = [0] * (m + 1)
    way = [0] * (m + 1)

    for i in range(1, n + 1):
        p[0] = i
        j0 = 0
        minv = np.full(m + 1, math.inf)
        used = [False] * (m + 1)

        while True:
            used[j0] = True
            i0 = p[j0]
            delta = math.inf
            j1 = 0

            for j in range(1, m + 1):
                if used[j]:
                    continue
                cur = cost[i0 - 1, j - 1] - u[i0] - v[j]
                if cur < minv[j]:
                    minv[j] = cur
                    way[j] = j0
                if minv[j] < delta:
                    delta = minv[j]
                    j1 = j

            for j in range(0, m + 1):
                if used[j]:
                    u[p[j]] += delta
                    v[j] -= delta
                else:
                    minv[j] -= delta

            j0 = j1
            if p[j0] == 0:
                break

        # Augmenting path
        while True:
            j1 = way[j0]
            p[j0] = p[j1]
            j0 = j1
            if j0 == 0:
                break

    assignment = [-1] * n
    for j in range(1, m + 1):
        if p[j] != 0:
            assignment[p[j] - 1] = j - 1

    if not transposed:
        return assignment

    # Invert assignment for the transposed problem.
    # assignment is for transposed rows (original columns).
    inv = [-1] * m
    for row_t, col_t in enumerate(assignment):
        if col_t >= 0:
            inv[col_t] = row_t
    return inv


def hungarian_assignment(utilities: np.ndarray) -> AssignmentResult:
    """Optimal assignment using the Hungarian algorithm (maximizes total utility).

    - Unreachable edges should be represented as -inf utility.
    - If there are fewer targets than robots (or some robots have no reachable targets), some robots will be left unassigned.
    - Hungarian algorithm requires a square cost matrix, so we add dummy targets 
    """

    if utilities is None or utilities.size == 0:
        return AssignmentResult(robot_to_target={})

    num_robots, num_targets = utilities.shape
    if num_targets == 0:
        return AssignmentResult(robot_to_target={})

    dummy_util = -1e9

    # Replace non-finite entries with dummy_util, then append dummy columns.
    aug_utils = np.where(np.isfinite(utilities), utilities, dummy_util)
    dummy_cols = np.full((num_robots, num_robots), dummy_util)
    aug_utils = np.concatenate([aug_utils, dummy_cols], axis=1)

    max_u = aug_utils.max()
    if not math.isfinite(max_u):
        # Everything is infeasible
        return AssignmentResult(robot_to_target={})

    # Convert max-utility to min-cost by shifting.
    cost = max_u - aug_utils

    assignment_cols = _hungarian_min_cost(cost)

    robot_to_target: Dict[int, int] = {}
    for i, j in enumerate(assignment_cols[:num_robots]):
        if j < 0 or j >= num_targets:
            continue  # unassigned or dummy
        # Keep only if it was actually feasible.
        if not np.isfinite(utilities[i, j]):
            continue
        robot_to_target[i] = j

    return AssignmentResult(robot_to_target=robot_to_target)
