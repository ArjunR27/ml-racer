"""compare_agents.py — run multiple saved checkpoints on the same track(s).

Each agent plays every episode on an identical track (fixed seed per episode)

Usage:
    python compare_agents.py checkpoints/DQN_ep_100.pt checkpoints/DQN_ep_500.pt
    python compare_agents.py checkpoints/*.pt --episodes 20 --no-render
    python compare_agents.py checkpoints/*.pt --seed 42 
"""

import argparse
import os

import numpy as np
import torch

from config import (
    dqn_env_cfg,
    dqn_train_cfg,
    double_dqn_env_cfg,
    double_dqn_train_cfg,
    env_cfg,
    ppo_env_cfg,
    ppo_train_cfg,
    train_cfg,
)
from env_setup import make_env

try:
    import pygame
    _PYGAME_AVAILABLE = True
except ImportError:
    _PYGAME_AVAILABLE = False


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


def _configs_for_checkpoint(checkpoint_path: str, agent_type: str):
    if agent_type == "ppo":
        return ppo_env_cfg, ppo_train_cfg
    if agent_type == "double_dqn":
        return double_dqn_env_cfg, double_dqn_train_cfg
    if agent_type == "dqn":
        return dqn_env_cfg, dqn_train_cfg

    basename = os.path.basename(checkpoint_path).lower()

    try:
        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    except Exception:
        if basename.startswith("ppo"):
            return ppo_env_cfg, ppo_train_cfg
        if basename.startswith("doubledqn") or basename.startswith("double_dqn"):
            return double_dqn_env_cfg, double_dqn_train_cfg
        if basename.startswith("dqn"):
            return dqn_env_cfg, dqn_train_cfg
        return env_cfg, train_cfg

    if ckpt.get("agent_name") == "DoubleDQN":
        return double_dqn_env_cfg, double_dqn_train_cfg
    if "ac" in ckpt:
        return ppo_env_cfg, ppo_train_cfg
    if basename.startswith("doubledqn") or basename.startswith("double_dqn"):
        return double_dqn_env_cfg, double_dqn_train_cfg
    if "q_net" in ckpt:
        return dqn_env_cfg, dqn_train_cfg

    return env_cfg, train_cfg


def run_agent(checkpoint_path: str, seeds: list[int], render: bool) -> list[float]:
    """Load one agent and run it on each seed. Returns list of episode rewards."""
    if render and _PYGAME_AVAILABLE:
        pygame.init()

    selected_env_cfg, selected_train_cfg = _configs_for_checkpoint(
        checkpoint_path, agent_type="auto"
    )
    render_mode = "human" if render else None
    env = make_env(selected_env_cfg, render_mode=render_mode)
    agent = selected_train_cfg.agent(env.observation_space, env.action_space)
    agent.load(checkpoint_path)

    clock = pygame.time.Clock() if (render and _PYGAME_AVAILABLE) else None

    rewards = []
    for ep, seed in enumerate(seeds, 1):
        obs, _ = env.reset(seed=seed)
        episode_reward = 0.0
        steps = 0

        while True:
            if clock is not None:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        env.close()
                        return rewards

            action = _select_eval_action(agent, obs)
            obs, reward, terminated, truncated, _ = env.step(action)
            episode_reward += reward
            steps += 1

            if clock is not None:
                clock.tick(60)

            if terminated or truncated:
                break

        rewards.append(episode_reward)
        print(f"  ep {ep:>3} (seed {seed}) | steps {steps:>4} | reward {episode_reward:>8.2f}")

    env.close()
    return rewards


def compare(checkpoints: list[str], num_episodes: int, render: bool, seed: int | None) -> None:
    # Fixed seed mode: one episode on one track, no variance expected
    if seed is not None:
        seeds = [seed]
        print(f"Fixed seed mode: all agents play seed {seed} once.")
    else:
        rng = np.random.default_rng(seed=0)
        seeds = rng.integers(0, 100_000, size=num_episodes).tolist()

    all_results: dict[str, list[float]] = {}

    for path in checkpoints:
        if not os.path.exists(path):
            print(f"Skipping '{path}' — file not found.")
            continue

        label = os.path.basename(path)
        print(f"\n{'='*55}")
        print(f"  {label}")
        print(f"{'='*55}")

        rewards = run_agent(path, seeds, render)
        all_results[label] = rewards

    if seed is not None:
        _print_fixed_seed_comparison(all_results, seed)
    else:
        _print_comparison(all_results)


def _print_fixed_seed_comparison(results: dict[str, list[float]], seed: int) -> None:
    """Single-episode comparison — no stats, just ranked scores."""
    if not results:
        return

    col = 28
    print(f"\n{'='*50}")
    print(f"  Fixed track (seed={seed})")
    print(f"  {'Checkpoint':<{col}} {'Reward':>8}")
    print(f"  {'-'*col} {'--------':>8}")

    ranked = sorted(results.items(), key=lambda x: x[1][0], reverse=True)
    for i, (label, rewards) in enumerate(ranked):
        prefix = "* " if i == 0 else "  "
        print(f"{prefix}{label:<{col}} {rewards[0]:>8.2f}")

    print(f"{'='*50}")
    print(f"  * = best reward on this track")


def _print_comparison(results: dict[str, list[float]]) -> None:
    """Multi-episode comparison with full stats."""
    if not results:
        return

    col = 28
    print(f"\n{'='*65}")
    print(f"  {'Checkpoint':<{col}} {'Mean':>8}  {'Std':>7}  {'Min':>8}  {'Max':>8}")
    print(f"  {'-'*col} {'--------':>8}  {'-------':>7}  {'--------':>8}  {'--------':>8}")

    ranked = sorted(results.items(), key=lambda x: np.mean(x[1]), reverse=True)
    for i, (label, rewards) in enumerate(ranked):
        prefix = "* " if i == 0 else "  "
        print(
            f"{prefix}{label:<{col}} "
            f"{np.mean(rewards):>8.2f}  "
            f"{np.std(rewards):>7.2f}  "
            f"{np.min(rewards):>8.2f}  "
            f"{np.max(rewards):>8.2f}"
        )

    print(f"{'='*65}")
    print(f"  * = best mean reward across {len(next(iter(results.values())))} episodes")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare saved RL agents on identical tracks.")
    parser.add_argument("checkpoints", nargs="+",  help="Paths to checkpoint files")
    parser.add_argument("--episodes",  type=int,   default=10,   help="Episodes per agent, ignored if --seed is set (default: 10)")
    parser.add_argument("--seed",      type=int,   default=None, help="Pin a single track seed — runs once per agent, no variance")
    parser.add_argument("--no-render", action="store_true",      help="Disable the pygame window")
    args = parser.parse_args()

    compare(
        checkpoints=args.checkpoints,
        num_episodes=args.episodes,
        render=not args.no_render,
        seed=args.seed,
    )
