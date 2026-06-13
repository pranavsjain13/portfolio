from __future__ import annotations

from dataclasses import dataclass
from typing import List, Protocol

import numpy as np

from .types import Action, Observation


class Policy(Protocol):
    def reset(self, obs: Observation) -> None: ...

    def act(self, obs: Observation) -> List[Action]: ...


@dataclass
class RandomPolicy:
    seed: int = 0

    def __post_init__(self) -> None:
        self.rng = np.random.default_rng(self.seed)

    def reset(self, obs: Observation) -> None:
        return None

    def act(self, obs: Observation) -> List[Action]:
        actions = [Action.UP, Action.DOWN, Action.LEFT, Action.RIGHT, Action.STAY]
        return [actions[int(self.rng.integers(0, len(actions)))] for _ in obs.robot_positions]
