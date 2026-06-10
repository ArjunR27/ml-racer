import argparse

import numpy as np

from agents.dqn_agent import DQNAgent
from config import dqn_env_cfg
from env_setup import make_env


def track_progress(env) -> tuple[int | None, int | None, float | None]:
    base_env = env.unwrapped
    visited = getattr(base_env, "tile_visited_count", None)
    track = getattr(base_env, "track", None)
    total = len(track) if track is not None else None
    pct = visited / total if visited is not None and total else None
    return visited, total, pct


def evaluate_seed(agent: DQNAgent, env, seed: int, max_steps: int | None) -> dict:
    obs, _ = env.reset(seed=seed)
    total_reward = 0.0
    steps = 0
    end_reason = "max_steps"
    final_info = {}

    old_epsilon = agent.epsilon
    agent.epsilon = 0.0

    while True:
        action = agent.select_action(obs)
        obs, reward, terminated, truncated, info = env.step(action)
        final_info = info
        total_reward += reward
        steps += 1

        if terminated:
            end_reason = "terminated"
            break
        if truncated:
            end_reason = "max_steps"
            break
        if max_steps is not None and steps >= max_steps:
            end_reason = "max_steps"
            break

    agent.epsilon = old_epsilon
    visited, total, pct = track_progress(env)
    return {
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
        description="Evaluate DQN checkpoint over random CarRacing seeds."
    )
    parser.add_argument("--checkpoint", default="checkpoints/DQN_best.pt")
    parser.add_argument("--num-seeds", type=int, default=100)
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument("--seed-stop", type=int, default=100_000)
    parser.add_argument("--rng-seed", type=int, default=0)
    parser.add_argument("--max-steps", type=int, default=None)
    args = parser.parse_args()

    env = make_env(dqn_env_cfg, render_mode=None, max_episode_steps=args.max_steps)
    agent = DQNAgent(env.observation_space, env.action_space)
    agent.load(args.checkpoint)

    rng = np.random.default_rng(args.rng_seed)
    seeds = rng.integers(args.seed_start, args.seed_stop, size=args.num_seeds).tolist()

    print(f"Evaluating {args.checkpoint}")
    print(f"Random seeds: {args.num_seeds} from [{args.seed_start}, {args.seed_stop})")
    print("=" * 80)

    rows = []
    for index, seed in enumerate(seeds, 1):
        row = evaluate_seed(agent, env, seed, args.max_steps)
        rows.append(row)

        progress = (
            f"{row['progress_pct']:.1f}%"
            if row["progress_pct"] is not None
            else "unknown"
        )
        print(
            f"{index:>3}/{args.num_seeds} | seed {seed:>6} | "
            f"reward {row['reward']:>8.2f} | "
            f"completion {progress:>7} | "
            f"steps {row['steps']:>4} | "
            f"end {row['end_reason']} | "
            f"lap_finished {row['lap_finished']}"
        )

    env.close()

    rewards = np.array([row["reward"] for row in rows], dtype=np.float32)
    progress_values = np.array(
        [
            row["progress_pct"]
            for row in rows
            if row["progress_pct"] is not None
        ],
        dtype=np.float32,
    )
    finish_rate = np.mean([row["lap_finished"] for row in rows]) * 100

    print("=" * 80)
    print(f"Average reward: {rewards.mean():.2f}")
    print(f"Reward std:     {rewards.std():.2f}")
    if len(progress_values) > 0:
        print(f"Average completion: {progress_values.mean():.1f}%")
        print(f"Completion std:     {progress_values.std():.1f}%")
    else:
        print("Average completion: unknown")
    print(f"Lap finish rate: {finish_rate:.1f}%")


if __name__ == "__main__":
    main()
