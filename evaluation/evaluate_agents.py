"""Evaluate saved racing-agent checkpoints on deterministic tracks.

This script is intentionally separate from compare_agents.py and training.
It ranks checkpoints by full-lap completion first, then track progress, then
reward.
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import (  # noqa: E402
    dqn_env_cfg,
    dqn_train_cfg,
    double_dqn_env_cfg,
    double_dqn_train_cfg,
    env_cfg,
    ppo_env_cfg,
    ppo_train_cfg,
    train_cfg,
)
from env_setup import make_env  # noqa: E402


@dataclass
class EpisodeResult:
    checkpoint: str
    seed: int
    reward: float
    steps: int
    lap_finished: bool
    visited_tiles: int | None
    total_tiles: int | None
    progress_percent: float
    end_reason: str


@dataclass
class AgentSummary:
    checkpoint: str
    finish_rate: float
    avg_progress: float
    avg_reward: float
    avg_steps: float
    episodes: int


def _select_eval_action(agent, obs):
    if hasattr(agent, "select_eval_action"):
        return agent.select_eval_action(obs)

    old_epsilon = getattr(agent, "epsilon", None)
    if old_epsilon is not None:
        agent.epsilon = 0.0

    action = agent.select_action(obs)

    if old_epsilon is not None:
        agent.epsilon = old_epsilon

    return action


def _track_progress(env) -> tuple[int | None, int | None]:
    base_env = env.unwrapped
    visited = getattr(base_env, "tile_visited_count", None)
    track = getattr(base_env, "track", None)
    total = len(track) if track is not None else None
    return visited, total


def _configs_for_checkpoint(checkpoint_path: str):
    basename = os.path.basename(checkpoint_path).lower()

    try:
        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    except Exception:
        if basename.startswith("ppo"):
            return ppo_env_cfg, ppo_train_cfg
        if basename.startswith(("doubledqn", "double_dqn")):
            return double_dqn_env_cfg, double_dqn_train_cfg
        if basename.startswith("dqn"):
            return dqn_env_cfg, dqn_train_cfg
        return env_cfg, train_cfg

    if ckpt.get("agent_name") == "DoubleDQN":
        return double_dqn_env_cfg, double_dqn_train_cfg
    if "ac" in ckpt:
        return ppo_env_cfg, ppo_train_cfg
    if basename.startswith(("doubledqn", "double_dqn")):
        return double_dqn_env_cfg, double_dqn_train_cfg
    if "q_net" in ckpt:
        return dqn_env_cfg, dqn_train_cfg

    return env_cfg, train_cfg


def _generate_seeds(num_episodes: int, seed: int | None) -> list[int]:
    if seed is not None:
        return [seed]

    rng = np.random.default_rng(seed=0)
    return rng.integers(0, 100_000, size=num_episodes).tolist()


def _end_reason(terminated: bool, truncated: bool, info: dict) -> str:
    if terminated and info.get("lap_finished", False):
        return "lap_finished"
    if terminated:
        return "off_track"
    if truncated:
        return "time_limit"
    return "max_steps"


def _run_episode(env, agent, checkpoint_label: str, seed: int, max_steps: int) -> EpisodeResult:
    obs, _ = env.reset(seed=seed)
    episode_reward = 0.0
    steps = 0
    final_info = {}
    terminated = False
    truncated = False

    for _ in range(max_steps):
        action = _select_eval_action(agent, obs)
        obs, reward, terminated, truncated, info = env.step(action)
        final_info = info
        episode_reward += reward
        steps += 1

        if terminated or truncated:
            break

    visited, total = _track_progress(env)
    progress_percent = (visited / total * 100.0) if visited is not None and total else 0.0

    return EpisodeResult(
        checkpoint=checkpoint_label,
        seed=seed,
        reward=episode_reward,
        steps=steps,
        lap_finished=final_info.get("lap_finished", False),
        visited_tiles=visited,
        total_tiles=total,
        progress_percent=progress_percent,
        end_reason=_end_reason(terminated, truncated, final_info),
    )


def evaluate_checkpoint(
    checkpoint_path: str,
    seeds: list[int],
    max_steps: int,
) -> list[EpisodeResult] | None:
    label = os.path.basename(checkpoint_path)

    try:
        selected_env_cfg, selected_train_cfg = _configs_for_checkpoint(checkpoint_path)
        env = make_env(
            selected_env_cfg,
            render_mode=None,
            max_episode_steps=max_steps,
        )
        agent = selected_train_cfg.agent(env.observation_space, env.action_space)
        agent.load(checkpoint_path)
    except Exception as exc:
        print(f"Skipping '{label}': could not load checkpoint/config ({exc})")
        return None

    try:
        return [
            _run_episode(env, agent, label, seed, max_steps)
            for seed in seeds
        ]
    finally:
        env.close()


def summarize_results(results: list[EpisodeResult]) -> AgentSummary:
    finish_rate = float(np.mean([r.lap_finished for r in results])) if results else 0.0
    avg_progress = float(np.mean([r.progress_percent for r in results])) if results else 0.0
    avg_reward = float(np.mean([r.reward for r in results])) if results else 0.0
    avg_steps = float(np.mean([r.steps for r in results])) if results else 0.0

    checkpoint = results[0].checkpoint if results else ""
    return AgentSummary(
        checkpoint=checkpoint,
        finish_rate=finish_rate,
        avg_progress=avg_progress,
        avg_reward=avg_reward,
        avg_steps=avg_steps,
        episodes=len(results),
    )


def _summary_sort_key(summary: AgentSummary):
    return (
        summary.finish_rate,
        summary.avg_progress,
        summary.avg_reward,
    )


def print_details(results_by_checkpoint: dict[str, list[EpisodeResult]]) -> None:
    for checkpoint, results in results_by_checkpoint.items():
        print(f"\n{checkpoint}")
        print("-" * len(checkpoint))
        for result in results:
            tiles = (
                f"{result.visited_tiles}/{result.total_tiles}"
                if result.visited_tiles is not None and result.total_tiles is not None
                else "unknown"
            )
            print(
                f"  seed {result.seed:<6} | "
                f"reward {result.reward:>8.2f} | "
                f"steps {result.steps:>4} | "
                f"lap {str(result.lap_finished):<5} | "
                f"progress {result.progress_percent:>5.1f}% ({tiles}) | "
                f"ended {result.end_reason}"
            )


def print_summary(summaries: list[AgentSummary]) -> None:
    if not summaries:
        print("No checkpoints were evaluated.")
        return

    ranked = sorted(summaries, key=_summary_sort_key, reverse=True)

    print("\nEvaluation Summary")
    print("=" * 88)
    print(
        f"{'Checkpoint':<32} "
        f"{'Finish%':>9} "
        f"{'AvgProgress':>12} "
        f"{'AvgReward':>10} "
        f"{'AvgSteps':>9} "
        f"{'Episodes':>8}"
    )
    print("-" * 88)

    for idx, summary in enumerate(ranked):
        prefix = "* " if idx == 0 else "  "
        print(
            f"{prefix}{summary.checkpoint:<30} "
            f"{summary.finish_rate * 100:>8.1f}% "
            f"{summary.avg_progress:>11.1f}% "
            f"{summary.avg_reward:>10.2f} "
            f"{summary.avg_steps:>9.1f} "
            f"{summary.episodes:>8}"
        )

    print("=" * 88)
    print("* = best by finish rate, then average progress, then average reward")


def write_csv(path: str, results_by_checkpoint: dict[str, list[EpisodeResult]]) -> None:
    output_path = Path(path)
    if output_path.parent != Path("."):
        output_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "checkpoint",
        "seed",
        "reward",
        "steps",
        "lap_finished",
        "visited_tiles",
        "total_tiles",
        "progress_percent",
        "end_reason",
    ]

    with output_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for results in results_by_checkpoint.values():
            for result in results:
                writer.writerow({
                    "checkpoint": result.checkpoint,
                    "seed": result.seed,
                    "reward": f"{result.reward:.6f}",
                    "steps": result.steps,
                    "lap_finished": result.lap_finished,
                    "visited_tiles": result.visited_tiles,
                    "total_tiles": result.total_tiles,
                    "progress_percent": f"{result.progress_percent:.6f}",
                    "end_reason": result.end_reason,
                })

    print(f"\nWrote CSV: {output_path}")


def evaluate(
    checkpoints: list[str],
    num_episodes: int,
    seed: int | None,
    max_steps: int,
    csv_path: str | None,
    details: bool,
) -> None:
    seeds = _generate_seeds(num_episodes, seed)

    if seed is not None:
        print(f"Fixed-seed evaluation: seed {seed}")
    else:
        print(f"Evaluating {num_episodes} deterministic held-out seeds: {seeds}")

    results_by_checkpoint: dict[str, list[EpisodeResult]] = {}
    summaries: list[AgentSummary] = []

    for checkpoint_path in checkpoints:
        if not os.path.exists(checkpoint_path):
            print(f"Skipping '{checkpoint_path}': file not found")
            continue

        print(f"\nEvaluating {os.path.basename(checkpoint_path)}")
        results = evaluate_checkpoint(checkpoint_path, seeds, max_steps)
        if results is None:
            continue

        results_by_checkpoint[os.path.basename(checkpoint_path)] = results
        summaries.append(summarize_results(results))

    if details:
        print_details(results_by_checkpoint)

    print_summary(summaries)

    if csv_path:
        write_csv(csv_path, results_by_checkpoint)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate trained racing agents by lap completion and progress."
    )
    parser.add_argument("checkpoints", nargs="+", help="Checkpoint .pt files to evaluate")
    parser.add_argument(
        "--episodes",
        type=int,
        default=10,
        help="Number of deterministic held-out seeds; ignored when --seed is set",
    )
    parser.add_argument("--seed", type=int, default=None, help="Evaluate one fixed track seed")
    parser.add_argument(
        "--max-steps",
        type=int,
        default=1000,
        help="Maximum steps per evaluation episode",
    )
    parser.add_argument("--csv", dest="csv_path", default=None, help="Optional CSV output path")
    parser.add_argument(
        "--details",
        action="store_true",
        help="Print one line per checkpoint per seed",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    evaluate(
        checkpoints=args.checkpoints,
        num_episodes=args.episodes,
        seed=args.seed,
        max_steps=args.max_steps,
        csv_path=args.csv_path,
        details=args.details,
    )
