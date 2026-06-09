import csv
import os
import time
from collections import deque

import numpy as np

from config import EnvConfig, TrainConfig, env_cfg, train_cfg
from env_setup import make_env

try:
    import pygame
    _PYGAME_AVAILABLE = True
except ImportError:
    _PYGAME_AVAILABLE = False


NO_PROGRESS_LIMIT = 120
TRAINING_LOG_DIR = "training_logs"
TRAINING_LOG_FIELDS = [
    "episode",
    "agent",
    "seed",
    "reward",
    "avg_reward",
    "steps",
    "lap_finished",
    "visited_tiles",
    "total_tiles",
    "progress_percent",
    "avg_progress_percent",
    "end_reason",
    "elapsed_seconds",
    "loss",
    "epsilon",
    "pg_loss",
    "vf_loss",
    "entropy",
]


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


def _training_log_path(agent_name: str, seed_text: str) -> str:
    safe_seed = seed_text.replace(os.sep, "_")
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    return os.path.join(TRAINING_LOG_DIR, f"{agent_name}_seed_{safe_seed}_{timestamp}.csv")


def _mean_metrics(episode_metrics: dict[str, list]) -> dict[str, float]:
    return {
        key: float(np.mean(values))
        for key, values in episode_metrics.items()
        if values
    }


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


def _render_evaluation(env_cfg: EnvConfig, train_cfg: TrainConfig, agent, episode: int) -> None:
    if not _PYGAME_AVAILABLE:
        print("Skipping render evaluation because pygame is not installed.")
        return

    pygame.init()
    clock = pygame.time.Clock()
    env = make_env(env_cfg, render_mode="human")

    print(f"\n[render eval] episode {episode} -- watching current {agent.name} policy")

    for eval_ep in range(1, train_cfg.render_eval_episodes + 1):
        obs, _ = env.reset(seed=env_cfg.seed if env_cfg.seed != -1 else None)
        episode_reward = 0.0
        steps = 0

        for _ in range(train_cfg.max_steps_per_episode):
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    env.close()
                    return

            action = _select_eval_action(agent, obs)
            obs, reward, terminated, truncated, _ = env.step(action)
            episode_reward += reward
            steps += 1

            clock.tick(60)

            if terminated or truncated:
                break

        print(
            f"[render eval] eval_ep {eval_ep} | "
            f"steps {steps} | reward {episode_reward:.2f}"
        )

    env.close()


def train(env_cfg: EnvConfig, train_cfg: TrainConfig) -> None:
    """Main training loop."""

    clock = None
    if train_cfg.render and _PYGAME_AVAILABLE:
        pygame.init()
        clock = pygame.time.Clock()

    render_mode = "human" if train_cfg.render else None
    env = make_env(env_cfg, render_mode=render_mode)
    agent = train_cfg.agent(env.observation_space, env.action_space)

    checkpoint_path = f"checkpoints/{agent.name}_best.pt"

    if os.path.exists(checkpoint_path):
        agent.load(checkpoint_path)
        print(f"Loaded checkpoint: {checkpoint_path}")
    else:
        print("No checkpoint found, starting fresh.")

    os.makedirs(train_cfg.checkpoint_dir, exist_ok=True)

    reward_window = deque(maxlen=train_cfg.log_interval)
    progress_window = deque(maxlen=train_cfg.log_interval)
    best_avg_reward = float("-inf")
    training_seed = env_cfg.seed if env_cfg.seed != -1 else None
    seed_text = "random" if training_seed is None else str(training_seed)
    training_log_path = _training_log_path(agent.name, seed_text)
    os.makedirs(TRAINING_LOG_DIR, exist_ok=True)

    print(f"\n{'='*55}")
    print(f"  Training started")
    print(f"  Agent: {agent.name}")
    print(f"  obs_space: {env.observation_space}")
    print(f"  action_space: {env.action_space}")
    print(f"  episodes: {train_cfg.num_episodes}")
    print(f"  seed: {seed_text}")
    print(f"  training_log: {training_log_path}")
    print(f"{'='*55}\n")

    start_time = time.time()

    with open(training_log_path, "w", newline="") as log_file:
        log_writer = csv.DictWriter(log_file, fieldnames=TRAINING_LOG_FIELDS)
        log_writer.writeheader()

        for episode in range(1, train_cfg.num_episodes + 1):
            obs, _ = env.reset(seed=training_seed)
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
                            env.close()
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

                for k, v in metrics.items():
                    episode_metrics.setdefault(k, []).append(v)

                obs = next_obs
                episode_reward += reward
                episode_steps += 1

                if clock is not None:
                    clock.tick(60)

                if done:
                    break

            progress = _progress_fraction(final_visited, final_total)
            reward_window.append(episode_reward)
            progress_window.append(progress)
            avg_reward = float(np.mean(reward_window))
            avg_progress = float(np.mean(progress_window))
            elapsed = time.time() - start_time
            metric_averages = _mean_metrics(episode_metrics)

            log_writer.writerow({
                "episode": episode,
                "agent": agent.name,
                "seed": seed_text,
                "reward": f"{episode_reward:.6f}",
                "avg_reward": f"{avg_reward:.6f}",
                "steps": episode_steps,
                "lap_finished": final_info.get("lap_finished", False),
                "visited_tiles": final_visited,
                "total_tiles": final_total,
                "progress_percent": f"{progress * 100:.6f}",
                "avg_progress_percent": f"{avg_progress * 100:.6f}",
                "end_reason": end_reason,
                "elapsed_seconds": f"{elapsed:.6f}",
                "loss": f"{metric_averages['loss']:.6f}" if "loss" in metric_averages else "",
                "epsilon": f"{metric_averages['epsilon']:.6f}" if "epsilon" in metric_averages else "",
                "pg_loss": f"{metric_averages['pg_loss']:.6f}" if "pg_loss" in metric_averages else "",
                "vf_loss": f"{metric_averages['vf_loss']:.6f}" if "vf_loss" in metric_averages else "",
                "entropy": f"{metric_averages['entropy']:.6f}" if "entropy" in metric_averages else "",
            })
            log_file.flush()

            if episode % train_cfg.log_interval == 0:
                metric_str = "  ".join(
                    f"{k}={v:.4f}"
                    for k, v in metric_averages.items()
                )
                print(
                    f"ep {episode:>6} | "
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
                    best_path = os.path.join(train_cfg.checkpoint_dir, f"{agent.name}_best.pt")
                    agent.save(best_path)
                    print(f"           ^ new best avg reward -- saved to {best_path}")

            if episode % train_cfg.save_interval == 0:
                ckpt_path = os.path.join(train_cfg.checkpoint_dir, f"{agent.name}_ep_{episode}.pt")
                agent.save(ckpt_path)

            if (
                train_cfg.render_eval_interval > 0
                and episode % train_cfg.render_eval_interval == 0
            ):
                _render_evaluation(env_cfg, train_cfg, agent, episode)

    if (
        train_cfg.render_eval_interval > 0
        and train_cfg.num_episodes % train_cfg.render_eval_interval != 0
    ):
        _render_evaluation(env_cfg, train_cfg, agent, train_cfg.num_episodes)

    env.close()
    total_time = time.time() - start_time
    print(f"\nTraining complete in {total_time:.1f}s")
    print(f"Training log saved to {training_log_path}")


if __name__ == "__main__":

    train(env_cfg, train_cfg)
