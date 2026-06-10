"""Standalone DQN trainer for Google Colab.

Colab setup:
    1. Runtime > Change runtime type > GPU
    2. Upload colab_dqn_agent.py and this file
    3. Run:
        !pip install "gymnasium[box2d]" swig pygame
        !python colab_train_dqn.py --episodes 500

Optional Google Drive checkpoints:
    from google.colab import drive
    drive.mount("/content/drive")
    !python colab_train_dqn.py --checkpoint-dir /content/drive/MyDrive/ml-racer-dqn
"""

from __future__ import annotations

import argparse
import os
import time
from collections import deque
from pathlib import Path

import gymnasium as gym
import numpy as np
from gymnasium.wrappers import (
    FrameStackObservation,
    GrayscaleObservation,
    ResizeObservation,
)

from colab_dqn_agent import DQNAgent


def make_env(seed: int, render_mode: str | None = None) -> gym.Env:
    """Create the discrete CarRacing environment used by DQN."""
    env = gym.make(
        "CarRacing-v3",
        continuous=False,
        domain_randomize=False,
        render_mode=render_mode,
    )
    env = GrayscaleObservation(env, keep_dim=False)
    env = ResizeObservation(env, (84, 84))
    env = FrameStackObservation(env, 4)
    env.reset(seed=seed)
    return env


def eval_agent(agent: DQNAgent, seed: int, max_steps: int, episodes: int = 1) -> float:
    """Headless evaluation on the fixed seed."""
    env = make_env(seed)
    rewards = []

    for _ in range(episodes):
        obs, _ = env.reset(seed=seed)
        episode_reward = 0.0

        for _ in range(max_steps):
            action = agent.select_eval_action(obs)
            obs, reward, terminated, truncated, _ = env.step(action)
            episode_reward += reward
            if terminated or truncated:
                break

        rewards.append(episode_reward)

    env.close()
    return float(np.mean(rewards))


def save_checkpoint(agent: DQNAgent, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    agent.save(str(path))
    print(f"saved checkpoint: {path}")


def train(args: argparse.Namespace) -> None:
    env = make_env(args.seed)
    agent = DQNAgent(
        env.observation_space,
        env.action_space,
        lr=args.lr,
        gamma=args.gamma,
        epsilon_start=args.epsilon_start,
        epsilon_min=args.epsilon_min,
        epsilon_decay=args.epsilon_decay,
        buffer_size=args.buffer_size,
        batch_size=args.batch_size,
        update_every=args.update_every,
        target_sync=args.target_sync,
        min_buffer=args.min_buffer,
    )

    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    best_path = checkpoint_dir / "DQN_best.pt"
    final_path = checkpoint_dir / "DQN_final.pt"

    if args.resume:
        resume_path = Path(args.resume)
    else:
        resume_path = best_path

    if resume_path.exists():
        agent.load(str(resume_path))
        print(f"loaded checkpoint: {resume_path}")
    else:
        print("no checkpoint loaded; starting fresh")

    reward_window = deque(maxlen=args.log_interval)
    best_avg_reward = float("-inf")
    start_time = time.time()

    print("=" * 60)
    print("DQN training started")
    print(f"episodes: {args.episodes}")
    print(f"max_steps_per_episode: {args.max_steps}")
    print(f"seed: {args.seed}")
    print(f"obs_space: {env.observation_space}")
    print(f"action_space: {env.action_space}")
    print(f"checkpoint_dir: {checkpoint_dir}")
    print("=" * 60)

    try:
        for episode in range(1, args.episodes + 1):
            obs, _ = env.reset(seed=args.seed)
            episode_reward = 0.0
            episode_metrics: dict[str, list[float]] = {}
            steps_without_progress = 0

            for step in range(1, args.max_steps + 1):
                action = agent.select_action(obs)
                next_obs, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated

                if reward > 0:
                    steps_without_progress = 0
                else:
                    steps_without_progress += 1

                if args.no_progress_limit > 0 and steps_without_progress > args.no_progress_limit:
                    done = True

                metrics = agent.update(obs, action, reward, next_obs, done)
                for key, value in metrics.items():
                    episode_metrics.setdefault(key, []).append(value)

                obs = next_obs
                episode_reward += reward

                if args.progress_interval > 0 and step % args.progress_interval == 0:
                    print(
                        f"[progress] ep {episode} | step {step}/{args.max_steps} | "
                        f"reward {episode_reward:.2f} | buffer {len(agent.buffer)} | "
                        f"epsilon {agent.epsilon:.4f}"
                    )

                if done:
                    break

            reward_window.append(episode_reward)

            if episode % args.log_interval == 0:
                avg_reward = float(np.mean(reward_window))
                elapsed = time.time() - start_time
                metric_str = " ".join(
                    f"{key}={np.mean(values):.4f}"
                    for key, values in episode_metrics.items()
                )
                print(
                    f"ep {episode:>5} | reward {episode_reward:>8.2f} | "
                    f"avg({args.log_interval}) {avg_reward:>8.2f} | "
                    f"steps {step:>4} | elapsed {elapsed:>7.0f}s"
                    + (f" | {metric_str}" if metric_str else "")
                )

                if avg_reward > best_avg_reward:
                    best_avg_reward = avg_reward
                    save_checkpoint(agent, best_path)

            if args.eval_interval > 0 and episode % args.eval_interval == 0:
                eval_reward = eval_agent(agent, args.seed, args.max_steps, args.eval_episodes)
                print(f"[eval] ep {episode} | seed {args.seed} | reward {eval_reward:.2f}")

            if episode % args.save_interval == 0:
                save_checkpoint(agent, checkpoint_dir / f"DQN_ep_{episode}.pt")

    except KeyboardInterrupt:
        print("\ntraining interrupted; saving final checkpoint")
        save_checkpoint(agent, final_path)
        raise
    finally:
        env.close()

    save_checkpoint(agent, final_path)
    print(f"training complete in {time.time() - start_time:.1f}s")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train DQN on Gymnasium CarRacing-v3.")
    parser.add_argument("--episodes", type=int, default=500)
    parser.add_argument("--max-steps", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints_dqn")
    parser.add_argument("--resume", type=str, default="")
    parser.add_argument("--log-interval", type=int, default=10)
    parser.add_argument("--save-interval", type=int, default=500)
    parser.add_argument("--eval-interval", type=int, default=50)
    parser.add_argument("--eval-episodes", type=int, default=1)
    parser.add_argument("--progress-interval", type=int, default=100)
    parser.add_argument("--no-progress-limit", type=int, default=120)

    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--epsilon-start", type=float, default=1.0)
    parser.add_argument("--epsilon-min", type=float, default=0.05)
    parser.add_argument("--epsilon-decay", type=float, default=0.9999)
    parser.add_argument("--buffer-size", type=int, default=50_000)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--update-every", type=int, default=4)
    parser.add_argument("--target-sync", type=int, default=1_000)
    parser.add_argument("--min-buffer", type=int, default=1_000)
    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())
