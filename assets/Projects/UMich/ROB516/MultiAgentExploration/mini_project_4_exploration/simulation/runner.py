from __future__ import annotations

import argparse
from dataclasses import asdict
from typing import Optional

from .env import GridWorldEnv
from .metrics import summarize_episode
from .policies import RandomPolicy, Policy
from .render import MatplotlibRenderer, render_ascii
from .types import EnvConfig


def run_episode(
    policy: Policy,
    config: EnvConfig,
    render: bool = False,
    render_every: int = 25,
    render_mpl: bool = False,
    plot_frontier_clusters: bool = False,
) -> dict:
    env = GridWorldEnv(config)
    obs = env.reset()
    policy.reset(obs)

    mpl = MatplotlibRenderer() if render_mpl else None

    last = None
    while True:
        actions = policy.act(obs)
        last = env.step(actions)
        obs = env.observe()

        if (last.step_index % render_every == 0 or last.done):
            if render:
                print(
                    f"\nStep {last.step_index} coverage={last.coverage:.3f} newly_revealed={last.newly_revealed} collisions={last.collisions}"
                )
                print(render_ascii(env.get_fused_map(), last.robot_positions))
            if mpl is not None:
                clusters = None
                targets = None
                if plot_frontier_clusters:
                    clusters = getattr(policy, "_debug_frontier_clusters", None)
                    targets = getattr(policy, "_debug_targets", None)
                mpl.update(
                    env.get_fused_map(),
                    last.robot_positions,
                    title=f"Step {last.step_index}  coverage={last.coverage:.3f}",
                    frontier_clusters=clusters,
                    targets=targets,
                )

        if last.done:
            break

    stats = summarize_episode(env.get_robot_states(), env.get_fused_map(), steps=last.step_index)
    return {
        "config": asdict(config),
        "episode": asdict(stats),
    }


def _make_policy(name: str, seed: int) -> Policy:
    if name == "random":
        return RandomPolicy(seed=seed)

    if name == "heuristic":
        from a_heuristic_exploration.policy import HeuristicFrontierPolicy

        return HeuristicFrontierPolicy()

    raise ValueError(f"Unknown policy: {name}")


def main(argv: Optional[list[str]] = None) -> None:
    p = argparse.ArgumentParser(description="Multi-robot exploration runner")
    p.add_argument("--policy", choices=["random", "heuristic"], default="random")
    p.add_argument("--width", type=int, default=32)
    p.add_argument("--height", type=int, default=32)
    p.add_argument("--num-robots", type=int, default=3)
    p.add_argument("--obstacle-prob", type=float, default=0.18)
    p.add_argument("--sensor-radius", type=int, default=4)
    p.add_argument("--max-steps", type=int, default=250)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--render", action="store_true")
    p.add_argument("--render-mpl", action="store_true")
    p.add_argument("--render-every", type=int, default=25)
    p.add_argument(
        "--plot-frontier-clusters",
        action="store_true",
        help="(heuristic + --render-mpl) overlay frontier clusters and target points",
    )

    args = p.parse_args(argv)

    config = EnvConfig(
        width=args.width,
        height=args.height,
        num_robots=args.num_robots,
        obstacle_prob=args.obstacle_prob,
        sensor_radius=args.sensor_radius,
        max_steps=args.max_steps,
        seed=args.seed,
    )

    policy = _make_policy(args.policy, seed=args.seed)
    result = run_episode(
        policy,
        config,
        render=args.render,
        render_every=args.render_every,
        render_mpl=args.render_mpl,
        plot_frontier_clusters=args.plot_frontier_clusters,
    )
    print("\nResult:")
    print(result)


if __name__ == "__main__":
    main()
