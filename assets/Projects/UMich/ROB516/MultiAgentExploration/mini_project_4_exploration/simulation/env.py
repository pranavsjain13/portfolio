from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np

from .maps import generate_random_map
from .metrics import compute_coverage
from .sensors import reveal_within_radius
from .types import ACTION_DELTAS, Action, CellState, EnvConfig, Observation, RobotState, StepResult


def _empty_unknown_map(width: int, height: int) -> np.ndarray:
    return np.full((height, width), int(CellState.UNKNOWN), dtype=np.int8)


def _in_bounds(x: int, y: int, width: int, height: int) -> bool:
    return 0 <= x < width and 0 <= y < height


@dataclass
class GridWorldState:
    truth: np.ndarray       # int-encoded grid [y, x] – ground truth (hidden)
    fused_map: np.ndarray   # union of every robot's observations [y, x]
    robot_maps: List[np.ndarray]  # per-robot private maps [y, x], one per robot
    robots: List[RobotState]
    step_index: int


class GridWorldEnv:
    """Discrete multi-robot exploration environment.

    Each robot maintains its own private knowledge map (``robot_maps[i]``).
    A cell in robot *i*'s map is revealed only when robot *i*'s sensor
    sweeps over it.  The shared ``fused_map`` is kept as the element-wise
    union (``np.maximum``) of all individual maps, so it represents the
    collective knowledge of the team.

    - ``truth`` is the fixed ground-truth map (obstacles / free cells).
    - ``robot_maps[i]`` accumulates only robot *i*'s sensor readings.
    - ``fused_map`` = element-wise max over all ``robot_maps``.
    """

    def __init__(self, config: EnvConfig):
        self.config = config
        self.rng = np.random.default_rng(config.seed)
        self.state: GridWorldState | None = None

    @property
    def width(self) -> int:
        return self.config.width

    @property
    def height(self) -> int:
        return self.config.height

    def reset(self) -> Observation:
        generated = generate_random_map(
            width=self.config.width,
            height=self.config.height,
            num_robots=self.config.num_robots,
            obstacle_prob=self.config.obstacle_prob,
            rng=self.rng,
        )

        robots = [RobotState(x=x, y=y) for (x, y) in generated.robot_starts]

        # Each robot gets its own private map, initially fully unknown.
        robot_maps = [
            _empty_unknown_map(self.config.width, self.config.height)
            for _ in robots
        ]

        # Initial reveal: each robot independently reveals into its own map
        # AND into the shared fused map (per user preference).
        fused = _empty_unknown_map(self.config.width, self.config.height)
        for i, robot in enumerate(robots):
            reveal_within_radius(
                truth=generated.truth,
                fused_map=robot_maps[i],
                robot_positions=[(robot.x, robot.y)],
                radius=self.config.sensor_radius,
            )
            reveal_within_radius(
                truth=generated.truth,
                fused_map=fused,
                robot_positions=[(robot.x, robot.y)],
                radius=self.config.sensor_radius,
            )

        self.state = GridWorldState(
            truth=generated.truth,
            fused_map=fused,
            robot_maps=robot_maps,
            robots=robots,
            step_index=0,
        )
        return self.observe()

    def observe(self) -> Observation:
        """Return a global observation using the fused map and all robot positions.

        Preserved for backward-compatibility with existing policies and runner.
        For per-robot private observations use ``observe_robot()`` or
        ``observe_all()``.
        """
        assert self.state is not None
        return Observation(
            fused_map=self.state.fused_map,
            robot_positions=[(r.x, r.y) for r in self.state.robots],
            step_index=self.state.step_index,
            max_steps=self.config.max_steps,
        )

    def observe_robot(self, robot_idx: int) -> Observation:
        """Return a per-robot observation for robot *robot_idx*.

        ``robot_map`` contains only the cells that *this* robot has sensed.
        ``fused_map`` is the shared union map.
        ``robot_positions`` includes only robots visible to robot *robot_idx*
        (i.e. within Chebyshev distance <= sensor_radius), plus itself.
        """
        assert self.state is not None
        rx, ry = self.state.robots[robot_idx].x, self.state.robots[robot_idx].y
        radius = self.config.sensor_radius
        visible_positions = [
            (r.x, r.y)
            for i, r in enumerate(self.state.robots)
            if i == robot_idx
            or (abs(r.x - rx) <= radius and abs(r.y - ry) <= radius)
        ]
        return Observation(
            fused_map=self.state.fused_map,
            robot_map=self.state.robot_maps[robot_idx],
            robot_positions=visible_positions,
            step_index=self.state.step_index,
            max_steps=self.config.max_steps,
        )

    def observe_all(self) -> List[Observation]:
        """Return a list of per-robot observations, one per robot."""
        assert self.state is not None
        return [self.observe_robot(i) for i in range(len(self.state.robots))]

    def step(self, actions: List[Action]) -> StepResult:
        assert self.state is not None
        if len(actions) != len(self.state.robots):
            raise ValueError(f"Expected {len(self.state.robots)} actions, got {len(actions)}")

        width, height = self.config.width, self.config.height
        intended: List[Tuple[int, int]] = []

        # Compute intended positions
        for robot, action in zip(self.state.robots, actions):
            dx, dy = ACTION_DELTAS[action]
            nx, ny = robot.x + dx, robot.y + dy
            if not _in_bounds(nx, ny, width, height):
                nx, ny = robot.x, robot.y
            intended.append((nx, ny))

        collisions = 0

        # Block moves into occupied cells
        for idx, (nx, ny) in enumerate(intended):
            if int(self.state.truth[ny, nx]) == int(CellState.OCCUPIED):
                intended[idx] = (self.state.robots[idx].x, self.state.robots[idx].y)
                collisions += 1
                self.state.robots[idx].collisions += 1

        # Handle robot-robot collisions / overlap
        if not self.config.allow_robot_overlap:
            seen = {}
            for i, pos in enumerate(intended):
                if pos in seen:
                    # Both robots stay
                    j = seen[pos]
                    intended[i] = (self.state.robots[i].x, self.state.robots[i].y)
                    intended[j] = (self.state.robots[j].x, self.state.robots[j].y)
                    collisions += 2
                    self.state.robots[i].collisions += 1
                    self.state.robots[j].collisions += 1
                else:
                    seen[pos] = i

        # Apply moves and distance
        for robot, (nx, ny) in zip(self.state.robots, intended):
            if (nx, ny) != (robot.x, robot.y):
                robot.distance_traveled += 1
            robot.x, robot.y = nx, ny

        # Reveal into each robot's private map individually.
        old_known = int(np.count_nonzero(self.state.fused_map != int(CellState.UNKNOWN)))
        for i, robot in enumerate(self.state.robots):
            reveal_within_radius(
                truth=self.state.truth,
                fused_map=self.state.robot_maps[i],
                robot_positions=[(robot.x, robot.y)],
                radius=self.config.sensor_radius,
            )

        # Recompute the fused map as the element-wise union of all robot maps.
        # np.maximum propagates known values (0 FREE, 1 OCCUPIED) over -1 UNKNOWN.
        np.copyto(self.state.fused_map, np.maximum.reduce(self.state.robot_maps))
        new_known = int(np.count_nonzero(self.state.fused_map != int(CellState.UNKNOWN)))
        newly_revealed = new_known - old_known

        self.state.step_index += 1
        coverage = compute_coverage(self.state.fused_map)
        done = self.state.step_index >= self.config.max_steps or coverage >= 0.999

        return StepResult(
            step_index=self.state.step_index,
            actions=list(actions),
            robot_positions=[(r.x, r.y) for r in self.state.robots],
            newly_revealed=int(newly_revealed),
            collisions=int(collisions),
            coverage=float(coverage),
            done=bool(done),
        )

    def get_truth(self) -> np.ndarray:
        assert self.state is not None
        return self.state.truth

    def get_fused_map(self) -> np.ndarray:
        assert self.state is not None
        return self.state.fused_map

    def get_robot_maps(self) -> List[np.ndarray]:
        """Return the list of per-robot private observation maps."""
        assert self.state is not None
        return self.state.robot_maps

    def get_robot_states(self) -> List[RobotState]:
        assert self.state is not None
        return self.state.robots

    def share_knowledge(self, agent_indices: List[int]):
        """Simulate a knowledge-sharing event between the specified agents.

        All agents in agent_indices share their private maps with eachother.

        The result is that each agent's private map becomes the union of all their maps.
        """
        assert self.state is not None
        assert all(0 <= idx < len(self.state.robots) for idx in agent_indices)

        # Get the maps to be shared
        maps_to_share = [self.state.robot_maps[idx] for idx in agent_indices]
        # Compute the union of the maps to be shared
        shared_map = np.maximum.reduce(maps_to_share)

        # Update each agent's private map with the shared map
        for idx in agent_indices:
            np.copyto(self.state.robot_maps[idx], shared_map)
        




