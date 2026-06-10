import argparse
from pathlib import Path

import imageio.v2 as imageio

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


def main() -> None:
    parser = argparse.ArgumentParser(description="Record a DQN CarRacing checkpoint.")
    parser.add_argument("--checkpoint", default="checkpoints/DQN_best.pt")
    parser.add_argument("--output", default="dqn_best_seed42.gif")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-steps", type=int, default=1000)
    parser.add_argument("--fps", type=int, default=30)
    args = parser.parse_args()

    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    env = make_env(dqn_env_cfg, render_mode="rgb_array", max_episode_steps=args.max_steps)
    agent = DQNAgent(env.observation_space, env.action_space)
    agent.load(str(checkpoint_path))

    obs, _ = env.reset(seed=args.seed)
    total_reward = 0.0
    frames = []

    old_epsilon = agent.epsilon
    agent.epsilon = 0.0

    for _ in range(args.max_steps):
        action = agent.select_action(obs)
        obs, reward, terminated, truncated, _ = env.step(action)
        total_reward += reward
        frames.append(env.render())

        if terminated or truncated:
            break

    agent.epsilon = old_epsilon
    visited, total, pct = track_progress(env)
    env.close()

    if args.output.lower().endswith(".gif"):
        imageio.mimsave(args.output, frames, duration=1 / args.fps, loop=0)
    else:
        imageio.mimsave(args.output, frames, fps=args.fps)

    print(f"Saved {len(frames)} frames to {args.output}")
    print(f"DQN reward on seed {args.seed}: {total_reward:.2f}")
    if visited is not None and total:
        print(f"Completion: {visited}/{total} tiles ({pct * 100:.1f}%)")
    else:
        print("Completion: unknown")


if __name__ == "__main__":
    main()
