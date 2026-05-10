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
    best_avg_reward = float("-inf")

    print(f"\n{'='*55}")
    print(f"  Training started")
    print(f"  Agent: {agent.name}")
    print(f"  obs_space: {env.observation_space}")
    print(f"  action_space: {env.action_space}")
    print(f"  episodes: {train_cfg.num_episodes}")
    print(f"{'='*55}\n")

    start_time = time.time()

    for episode in range(1, train_cfg.num_episodes + 1):
        obs, _ = env.reset()
        episode_reward = 0.0
        episode_steps = 0
        episode_metrics: dict[str, list] = {}

        for _ in range(train_cfg.max_steps_per_episode):
            if clock is not None:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        env.close()
                        return

            action = agent.select_action(obs)
            next_obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

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

        reward_window.append(episode_reward)

        if episode % train_cfg.log_interval == 0:
            avg_reward = np.mean(reward_window)
            elapsed = time.time() - start_time
            metric_str = "  ".join(
                f"{k}={np.mean(v):.4f}"
                for k, v in episode_metrics.items()
            )
            print(
                f"ep {episode:>6} | "
                f"steps {episode_steps:>4} | "
                f"reward {episode_reward:>8.2f} | "
                f"avg({train_cfg.log_interval}) {avg_reward:>8.2f} | "
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

    env.close()
    total_time = time.time() - start_time
    print(f"\nTraining complete in {total_time:.1f}s")


if __name__ == "__main__":

    train(env_cfg, train_cfg)