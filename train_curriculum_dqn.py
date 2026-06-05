"""Curriculum training loop for DQN-style agents.

The curriculum starts on one track, expands to a small fixed seed set, then
optionally trains on random tracks. This keeps early learning stable while
still allowing later generalization.
"""

from __future__ import annotations

import os
import random
import sys
import time
from collections import deque
from dataclasses import dataclass

import numpy as np

from config import (
    EnvConfig,
    TrainConfig,
    dqn_env_cfg,
    dqn_train_cfg,
    double_dqn_env_cfg,
    double_dqn_train_cfg,
)
from env_setup import make_env


AGENT_NAME = "double_dqn"
SINGLE_SEED = 42
SINGLE_SEED_EPISODES = 1000
SEED_SET = (42, 1001, 2027, 3003, 8080)
SEED_SET_EPISODES = 3000
RANDOM_EPISODES = 0
MAX_STEPS_PER_EPISODE = 500
NO_PROGRESS_LIMIT = 200
RESUME_FROM = None
FRESH_START = False

try:
    import pygame

    _PYGAME_AVAILABLE = True
except ImportError:
    _PYGAME_AVAILABLE = False


@dataclass(frozen=True)
class CurriculumPhase:
    name: str
    episodes: int
    seeds: tuple[int, ...] | None


def _track_progress(env) -> tuple[int, int]:
    base_env = env.unwrapped
    visited = getattr(base_env, "tile_visited_count", 0)
    track = getattr(base_env, "track", None)
    total = len(track) if track is not None else 0
    return int(visited), int(total)


def _progress_fraction(visited: int, total: int) -> float:
    return visited / total if total else 0.0


def _progress_text(visited: int, total: int) -> str:
    if not total:
        return "unknown"
    return f"{visited}/{total} ({_progress_fraction(visited, total):.1%})"


def _episode_seed(phase: CurriculumPhase, phase_episode: int) -> int | None:
    if phase.seeds is None:
        return random.randint(0, sys.maxsize)
    return phase.seeds[(phase_episode - 1) % len(phase.seeds)]


def _build_curriculum() -> list[CurriculumPhase]:
    phases = [
        CurriculumPhase("single_seed", SINGLE_SEED_EPISODES, (SINGLE_SEED,)),
        CurriculumPhase("seed_set", SEED_SET_EPISODES, SEED_SET),
        CurriculumPhase("random", RANDOM_EPISODES, None),
    ]
    return [phase for phase in phases if phase.episodes > 0]


def _select_config(agent_name: str) -> tuple[EnvConfig, TrainConfig]:
    if agent_name == "dqn":
        return dqn_env_cfg, dqn_train_cfg
    if agent_name == "double_dqn":
        return double_dqn_env_cfg, double_dqn_train_cfg
    raise ValueError(f"Unsupported agent: {agent_name}")


def _load_checkpoint(agent, checkpoint_path: str, fresh: bool) -> None:
    if fresh:
        print("Starting fresh because --fresh was passed.")
        return

    if os.path.exists(checkpoint_path):
        agent.load(checkpoint_path)
        print(f"Loaded checkpoint: {checkpoint_path}")
    else:
        print("No checkpoint found, starting fresh.")


def _save_best(agent, checkpoint_dir: str, agent_name: str) -> None:
    best_path = os.path.join(checkpoint_dir, f"{agent_name}_curriculum_best.pt")
    agent.save(best_path)
    print(f"           ^ new best avg reward -- saved to {best_path}")


def _save_periodic(agent, checkpoint_dir: str, agent_name: str, episode: int) -> None:
    ckpt_path = os.path.join(checkpoint_dir, f"{agent_name}_curriculum_ep_{episode}.pt")
    agent.save(ckpt_path)


def train_curriculum() -> None:
    env_cfg, train_cfg = _select_config(AGENT_NAME)
    curriculum = _build_curriculum()
    if not curriculum:
        raise ValueError("At least one curriculum phase must have episodes > 0.")

    train_cfg.max_steps_per_episode = MAX_STEPS_PER_EPISODE

    os.makedirs(train_cfg.checkpoint_dir, exist_ok=True)

    clock = None
    if train_cfg.render and _PYGAME_AVAILABLE:
        pygame.init()
        clock = pygame.time.Clock()

    render_mode = "human" if train_cfg.render else None
    env = make_env(env_cfg, render_mode=render_mode)
    agent = train_cfg.agent(env.observation_space, env.action_space)

    resume_path = RESUME_FROM or os.path.join(
        train_cfg.checkpoint_dir, f"{agent.name}_curriculum_best.pt"
    )
    _load_checkpoint(agent, resume_path, FRESH_START)

    reward_window = deque(maxlen=train_cfg.log_interval)
    progress_window = deque(maxlen=train_cfg.log_interval)
    best_avg_reward = float("-inf")
    total_episodes = sum(phase.episodes for phase in curriculum)
    global_episode = 0
    start_time = time.time()

    print(f"\n{'=' * 65}")
    print("  Curriculum training started")
    print(f"  Agent: {agent.name}")
    print(f"  obs_space: {env.observation_space}")
    print(f"  action_space: {env.action_space}")
    print(f"  total episodes: {total_episodes}")
    for phase in curriculum:
        seeds = "random" if phase.seeds is None else ",".join(str(seed) for seed in phase.seeds)
        print(f"  phase {phase.name}: {phase.episodes} episodes | seeds: {seeds}")
    print(f"{'=' * 65}\n")

    try:
        for phase in curriculum:
            for phase_episode in range(1, phase.episodes + 1):
                global_episode += 1
                seed = _episode_seed(phase, phase_episode)
                obs, _ = env.reset(seed=seed)
                episode_reward = 0.0
                episode_steps = 0
                episode_metrics: dict[str, list] = {}
                steps_without_progress = 0
                last_visited, _ = _track_progress(env)
                final_visited = last_visited
                final_total = 0
                final_info = {}
                end_reason = "max_steps"

                for _ in range(train_cfg.max_steps_per_episode):
                    if clock is not None:
                        for event in pygame.event.get():
                            if event.type == pygame.QUIT:
                                return

                    action = agent.select_action(obs)
                    next_obs, reward, terminated, truncated, info = env.step(action)
                    final_info = info
                    done = terminated or truncated

                    visited, total = _track_progress(env)
                    final_visited, final_total = visited, total
                    if visited > last_visited:
                        steps_without_progress = 0
                        last_visited = visited
                    else:
                        steps_without_progress += 1

                    if steps_without_progress > NO_PROGRESS_LIMIT:
                        done = True
                        end_reason = "no_progress"
                    elif terminated and info.get("lap_finished", False):
                        end_reason = "lap_finished"
                    elif terminated:
                        end_reason = "off_track"
                    elif truncated:
                        end_reason = "time_limit"

                    metrics = agent.update(obs, action, reward, next_obs, done)
                    for key, value in metrics.items():
                        episode_metrics.setdefault(key, []).append(value)

                    obs = next_obs
                    episode_reward += reward
                    episode_steps += 1

                    if clock is not None:
                        clock.tick(60)

                    if done:
                        break

                reward_window.append(episode_reward)
                progress_window.append(_progress_fraction(final_visited, final_total))

                if global_episode % train_cfg.log_interval == 0:
                    avg_reward = float(np.mean(reward_window))
                    avg_progress = float(np.mean(progress_window))
                    elapsed = time.time() - start_time
                    metric_str = "  ".join(
                        f"{key}={np.mean(value):.4f}"
                        for key, value in episode_metrics.items()
                    )
                    seed_text = "random" if seed is None else str(seed)
                    print(
                        f"ep {global_episode:>6}/{total_episodes:<6} | "
                        f"phase {phase.name:<11} | "
                        f"seed {seed_text:<8} | "
                        f"steps {episode_steps:>4} | "
                        f"reward {episode_reward:>8.2f} | "
                        f"avg({train_cfg.log_interval}) {avg_reward:>8.2f} | "
                        f"progress {_progress_text(final_visited, final_total)} | "
                        f"avg_progress {avg_progress:.1%} | "
                        f"ended {end_reason} | "
                        f"lap_finished {final_info.get('lap_finished', False)} | "
                        f"elapsed {elapsed:>6.0f}s"
                        + (f" | {metric_str}" if metric_str else "")
                    )

                    if avg_reward > best_avg_reward:
                        best_avg_reward = avg_reward
                        _save_best(agent, train_cfg.checkpoint_dir, agent.name)

                if global_episode % train_cfg.save_interval == 0:
                    _save_periodic(agent, train_cfg.checkpoint_dir, agent.name, global_episode)
    finally:
        env.close()

    total_time = time.time() - start_time
    print(f"\nCurriculum training complete in {total_time:.1f}s")


if __name__ == "__main__":
    train_curriculum()
