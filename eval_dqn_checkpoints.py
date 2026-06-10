import argparse
import csv
import glob
import os
import re
from pathlib import Path

from agents.dqn_agent import DQNAgent
from config import dqn_env_cfg
from env_setup import make_env


def checkpoint_sort_key(path: str) -> tuple[int, str]:
    name = os.path.basename(path)
    match = re.search(r"_ep_(\d+)\.pt$", name)
    if match:
        return int(match.group(1)), name
    if name == "DQN_best.pt":
        return 10**12, name
    if name == "DQN_final.pt":
        return 10**12 + 1, name
    return 10**12 + 2, name


def checkpoint_episode(path: str) -> int | None:
    match = re.search(r"_ep_(\d+)\.pt$", os.path.basename(path))
    if not match:
        return None
    return int(match.group(1))


def track_progress(env) -> tuple[int | None, int | None, float | None]:
    base_env = env.unwrapped
    visited = getattr(base_env, "tile_visited_count", None)
    track = getattr(base_env, "track", None)
    total = len(track) if track is not None else None
    pct = visited / total if visited is not None and total else None
    return visited, total, pct


def evaluate_checkpoint(
    path: str,
    seed: int,
    max_steps: int | None,
    no_progress_limit: int,
) -> dict:
    env = make_env(dqn_env_cfg, render_mode=None, max_episode_steps=max_steps)
    agent = DQNAgent(env.observation_space, env.action_space)
    agent.load(path)

    old_epsilon = agent.epsilon
    agent.epsilon = 0.0

    obs, _ = env.reset(seed=seed)
    total_reward = 0.0
    steps = 0
    end_reason = "max_steps"
    final_info = {}
    steps_without_progress = 0

    while True:
        action = agent.select_action(obs)
        obs, reward, terminated, truncated, info = env.step(action)
        final_info = info
        total_reward += reward
        steps += 1

        if reward > 0:
            steps_without_progress = 0
        else:
            steps_without_progress += 1

        if terminated:
            end_reason = "terminated"
            break
        if no_progress_limit > 0 and steps_without_progress > no_progress_limit:
            end_reason = "no_progress"
            break
        if truncated:
            end_reason = "max_steps"
            break
        if max_steps is not None and steps >= max_steps:
            end_reason = "max_steps"
            break

    agent.epsilon = old_epsilon
    visited, total, pct = track_progress(env)
    env.close()

    return {
        "checkpoint": os.path.basename(path),
        "path": path,
        "episode": checkpoint_episode(path),
        "seed": seed,
        "reward": total_reward,
        "steps": steps,
        "end_reason": end_reason,
        "lap_finished": bool(final_info.get("lap_finished", False)),
        "tiles_visited": visited,
        "tiles_total": total,
        "progress_pct": pct * 100 if pct is not None else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate DQN checkpoints on one CarRacing seed and write CSV logs."
    )
    parser.add_argument(
        "checkpoints",
        nargs="*",
        default=["checkpoints/DQN*.pt"],
        help="Checkpoint paths or glob patterns. Default: checkpoints/DQN*.pt",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--no-progress-limit", type=int, default=120)
    parser.add_argument("--output", default="dqn_checkpoint_eval_seed42.csv")
    args = parser.parse_args()

    paths: list[str] = []
    for pattern in args.checkpoints:
        matches = glob.glob(pattern)
        paths.extend(matches if matches else [pattern])

    paths = sorted(set(paths), key=checkpoint_sort_key)
    paths = [path for path in paths if os.path.exists(path)]

    if not paths:
        raise FileNotFoundError("No DQN checkpoint files found.")

    rows = []
    for index, path in enumerate(paths, 1):
        print(f"[{index}/{len(paths)}] evaluating {path}")
        row = evaluate_checkpoint(
            path,
            args.seed,
            args.max_steps,
            args.no_progress_limit,
        )
        rows.append(row)

        progress = (
            f"{row['progress_pct']:.1f}%"
            if row["progress_pct"] is not None
            else "unknown"
        )
        print(
            f"  reward {row['reward']:.2f} | steps {row['steps']} | "
            f"end {row['end_reason']} | progress {progress} | "
            f"lap_finished {row['lap_finished']}"
        )

    fieldnames = [
        "checkpoint",
        "path",
        "episode",
        "seed",
        "reward",
        "steps",
        "end_reason",
        "lap_finished",
        "tiles_visited",
        "tiles_total",
        "progress_pct",
    ]

    output_path = Path(args.output)
    with output_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nSaved evaluation log to {output_path}")


if __name__ == "__main__":
    main()
