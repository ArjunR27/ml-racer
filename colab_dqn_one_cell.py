# Paste this whole file into one Google Colab cell after installing dependencies.
#
# Dependency cell to run first:
#   !pip -q install "gymnasium[box2d]" swig pygame
#
# Then paste this full file into the next cell and press Run.

from __future__ import annotations

import os
import random
import time
from collections import deque
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from gymnasium import spaces
from gymnasium.wrappers import FrameStackObservation, GrayscaleObservation, ResizeObservation


# ---------------------------------------------------------------------------
# Colab training settings
# ---------------------------------------------------------------------------
EPISODES = 500
MAX_STEPS = 500
SEED = 42
CHECKPOINT_DIR = "checkpoints_dqn"

LOG_INTERVAL = 10
SAVE_INTERVAL = 500
EVAL_INTERVAL = 50
PROGRESS_INTERVAL = 100
NO_PROGRESS_LIMIT = 120

LR = 1e-4
GAMMA = 0.99
EPSILON_START = 1.0
EPSILON_MIN = 0.05
EPSILON_DECAY = 0.9999
BUFFER_SIZE = 50_000
BATCH_SIZE = 32
UPDATE_EVERY = 4
TARGET_SYNC = 1_000
MIN_BUFFER = 1_000


class QNetwork(nn.Module):
    def __init__(self, obs_shape: tuple[int, ...], n_actions: int):
        super().__init__()

        if len(obs_shape) == 2:
            in_channels = 1
            height, width = obs_shape
            self.needs_permute = False
        elif len(obs_shape) == 3 and obs_shape[0] <= 16:
            in_channels, height, width = obs_shape
            self.needs_permute = False
        else:
            height, width, in_channels = obs_shape
            self.needs_permute = True

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=8, stride=4), nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2), nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1), nn.ReLU(),
        )

        with torch.no_grad():
            dummy = torch.zeros(1, in_channels, height, width)
            conv_out = int(np.prod(self.conv(dummy).shape))

        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(conv_out, 512), nn.ReLU(),
            nn.Linear(512, n_actions),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.needs_permute:
            x = x.permute(0, 3, 1, 2)
        if x.dim() == 3:
            x = x.unsqueeze(1)
        x = x / 255.0
        return self.fc(self.conv(x))


class ReplayBuffer:
    def __init__(self, capacity: int):
        self.buffer = deque(maxlen=capacity)

    def push(self, obs, action, reward, next_obs, done) -> None:
        self.buffer.append((obs, action, reward, next_obs, done))

    def sample(self, batch_size: int):
        batch = random.sample(self.buffer, batch_size)
        obs, actions, rewards, next_obs, dones = zip(*batch)
        return (
            np.array(obs, dtype=np.float32),
            np.array(actions, dtype=np.int64),
            np.array(rewards, dtype=np.float32),
            np.array(next_obs, dtype=np.float32),
            np.array(dones, dtype=np.float32),
        )

    def __len__(self) -> int:
        return len(self.buffer)


class DQNAgent:
    def __init__(
        self,
        obs_space: spaces.Space,
        action_space: spaces.Discrete,
        lr: float = LR,
        gamma: float = GAMMA,
        epsilon_start: float = EPSILON_START,
        epsilon_min: float = EPSILON_MIN,
        epsilon_decay: float = EPSILON_DECAY,
        buffer_size: int = BUFFER_SIZE,
        batch_size: int = BATCH_SIZE,
        update_every: int = UPDATE_EVERY,
        target_sync: int = TARGET_SYNC,
        min_buffer: int = MIN_BUFFER,
    ):
        if not isinstance(action_space, spaces.Discrete):
            raise ValueError("DQNAgent requires a discrete action space.")

        self.name = "DQN"
        self.obs_shape = obs_space.shape
        self.action_space = action_space
        self.n_actions = int(action_space.n)

        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.update_every = update_every
        self.target_sync = target_sync
        self.min_buffer = min_buffer
        self.step_count = 0

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[DQNAgent] device={self.device} obs_shape={self.obs_shape}")

        self.q_net = QNetwork(self.obs_shape, self.n_actions).to(self.device)
        self.target_net = QNetwork(self.obs_shape, self.n_actions).to(self.device)
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.q_net.parameters(), lr=lr)
        self.buffer = ReplayBuffer(buffer_size)

    def select_action(self, obs: np.ndarray) -> int:
        if random.random() < self.epsilon:
            return int(self.action_space.sample())

        obs_t = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_values = self.q_net(obs_t)
        return int(q_values.argmax(dim=1).item())

    def select_eval_action(self, obs: np.ndarray) -> int:
        old_epsilon = self.epsilon
        self.epsilon = 0.0
        action = self.select_action(obs)
        self.epsilon = old_epsilon
        return action

    def update(self, obs, action, reward, next_obs, done) -> dict[str, float]:
        self.buffer.push(obs, action, reward, next_obs, done)
        self.step_count += 1

        if len(self.buffer) < self.min_buffer:
            return {}
        if self.step_count % self.update_every != 0:
            return {}

        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        loss = self._learn()

        if self.step_count % self.target_sync == 0:
            self.target_net.load_state_dict(self.q_net.state_dict())

        return {"loss": loss, "epsilon": self.epsilon}

    def save(self, path: str) -> None:
        torch.save({
            "agent_name": self.name,
            "q_net": self.q_net.state_dict(),
            "target_net": self.target_net.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "epsilon": self.epsilon,
            "step_count": self.step_count,
            "buffer": self.buffer.buffer,
        }, path)

    def load(self, path: str) -> None:
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        self.q_net.load_state_dict(ckpt["q_net"])
        self.target_net.load_state_dict(ckpt["target_net"])
        self.optimizer.load_state_dict(ckpt["optimizer"])
        self.epsilon = ckpt["epsilon"]
        self.step_count = ckpt["step_count"]
        self.buffer.buffer = ckpt["buffer"]

    def _learn(self) -> float:
        obs, actions, rewards, next_obs, dones = self.buffer.sample(self.batch_size)

        obs = torch.tensor(obs, dtype=torch.float32).to(self.device)
        actions = torch.tensor(actions, dtype=torch.int64).to(self.device)
        rewards = torch.tensor(rewards, dtype=torch.float32).to(self.device)
        next_obs = torch.tensor(next_obs, dtype=torch.float32).to(self.device)
        dones = torch.tensor(dones, dtype=torch.float32).to(self.device)

        q_values = self.q_net(obs).gather(1, actions.unsqueeze(1)).squeeze(1)

        with torch.no_grad():
            next_q = self.target_net(next_obs).max(dim=1).values
            targets = rewards + self.gamma * next_q * (1.0 - dones)

        loss = nn.functional.smooth_l1_loss(q_values, targets)

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.q_net.parameters(), max_norm=10)
        self.optimizer.step()

        return float(loss.item())


def make_env(seed: int, render_mode: str | None = None) -> gym.Env:
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


def save_checkpoint(agent: DQNAgent, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    agent.save(str(path))
    print(f"saved checkpoint: {path}")


def eval_agent(agent: DQNAgent, seed: int, max_steps: int) -> float:
    env = make_env(seed)
    obs, _ = env.reset(seed=seed)
    episode_reward = 0.0

    for _ in range(max_steps):
        action = agent.select_eval_action(obs)
        obs, reward, terminated, truncated, _ = env.step(action)
        episode_reward += reward
        if terminated or truncated:
            break

    env.close()
    return float(episode_reward)


def train() -> None:
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    checkpoint_dir = Path(CHECKPOINT_DIR)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    best_path = checkpoint_dir / "DQN_best.pt"
    final_path = checkpoint_dir / "DQN_final.pt"

    env = make_env(SEED)
    agent = DQNAgent(env.observation_space, env.action_space)

    if best_path.exists():
        agent.load(str(best_path))
        print(f"loaded checkpoint: {best_path}")
    else:
        print("no checkpoint loaded; starting fresh")

    reward_window = deque(maxlen=LOG_INTERVAL)
    best_avg_reward = float("-inf")
    start_time = time.time()

    print("=" * 60)
    print("DQN training started")
    print(f"episodes: {EPISODES}")
    print(f"max_steps: {MAX_STEPS}")
    print(f"seed: {SEED}")
    print(f"checkpoint_dir: {checkpoint_dir}")
    print(f"obs_space: {env.observation_space}")
    print(f"action_space: {env.action_space}")
    print("=" * 60)

    try:
        for episode in range(1, EPISODES + 1):
            obs, _ = env.reset(seed=SEED)
            episode_reward = 0.0
            episode_metrics: dict[str, list[float]] = {}
            steps_without_progress = 0

            for step in range(1, MAX_STEPS + 1):
                action = agent.select_action(obs)
                next_obs, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated

                if reward > 0:
                    steps_without_progress = 0
                else:
                    steps_without_progress += 1
                if NO_PROGRESS_LIMIT > 0 and steps_without_progress > NO_PROGRESS_LIMIT:
                    done = True

                metrics = agent.update(obs, action, reward, next_obs, done)
                for key, value in metrics.items():
                    episode_metrics.setdefault(key, []).append(value)

                obs = next_obs
                episode_reward += reward

                if PROGRESS_INTERVAL > 0 and step % PROGRESS_INTERVAL == 0:
                    print(
                        f"[progress] ep {episode} | step {step}/{MAX_STEPS} | "
                        f"reward {episode_reward:.2f} | buffer {len(agent.buffer)} | "
                        f"epsilon {agent.epsilon:.4f}"
                    )

                if done:
                    break

            reward_window.append(episode_reward)

            if episode % LOG_INTERVAL == 0:
                avg_reward = float(np.mean(reward_window))
                elapsed = time.time() - start_time
                metric_str = " ".join(
                    f"{key}={np.mean(values):.4f}"
                    for key, values in episode_metrics.items()
                )
                print(
                    f"ep {episode:>5} | reward {episode_reward:>8.2f} | "
                    f"avg({LOG_INTERVAL}) {avg_reward:>8.2f} | "
                    f"steps {step:>4} | elapsed {elapsed:>7.0f}s"
                    + (f" | {metric_str}" if metric_str else "")
                )

                if avg_reward > best_avg_reward:
                    best_avg_reward = avg_reward
                    save_checkpoint(agent, best_path)

            if EVAL_INTERVAL > 0 and episode % EVAL_INTERVAL == 0:
                eval_reward = eval_agent(agent, SEED, MAX_STEPS)
                print(f"[eval] ep {episode} | seed {SEED} | reward {eval_reward:.2f}")

            if episode % SAVE_INTERVAL == 0:
                save_checkpoint(agent, checkpoint_dir / f"DQN_ep_{episode}.pt")

    except KeyboardInterrupt:
        print("\ntraining interrupted; saving final checkpoint")
        save_checkpoint(agent, final_path)
        raise
    finally:
        env.close()

    save_checkpoint(agent, final_path)
    print(f"training complete in {time.time() - start_time:.1f}s")


train()
