"""Standalone DQN agent for Google Colab CarRacing training.

Upload this file together with colab_train_dqn.py. It has no dependency on the
rest of this repository.
"""

from __future__ import annotations

import random
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from gymnasium import spaces


class QNetwork(nn.Module):
    """CNN that maps image observations to Q-values for each discrete action."""

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
    """Fixed-size replay buffer with random batch sampling."""

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
    """DQN with replay buffer, target network, and epsilon-greedy exploration."""

    def __init__(
        self,
        obs_space: spaces.Space,
        action_space: spaces.Discrete,
        lr: float = 1e-4,
        gamma: float = 0.99,
        epsilon_start: float = 1.0,
        epsilon_min: float = 0.05,
        epsilon_decay: float = 0.9999,
        buffer_size: int = 50_000,
        batch_size: int = 32,
        update_every: int = 4,
        target_sync: int = 1_000,
        min_buffer: int = 1_000,
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
