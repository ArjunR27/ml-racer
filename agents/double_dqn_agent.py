"""Double DQN agent.

Double DQN keeps the same CNN, replay buffer, and epsilon-greedy exploration
as regular DQN, but reduces Q-value overestimation by splitting next-action
selection and next-action evaluation across the online and target networks.
"""

import random

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from gymnasium import spaces

from agents.base_agent import BaseAgent
from agents.dqn_agent import QNetwork, ReplayBuffer


class DoubleDQNAgent(BaseAgent):
    """DQN variant using Double DQN targets."""

    def __init__(
        self,
        obs_space: spaces.Space,
        action_space: spaces.Space,
        lr: float            = 1e-4,
        gamma: float         = 0.99,
        epsilon_start: float = 1.0,
        epsilon_min: float   = 0.05,
        epsilon_decay: float = 0.9999,
        buffer_size: int     = 50_000,
        batch_size: int      = 32,
        update_every: int    = 4,
        target_sync: int     = 1_000,
        min_buffer: int      = 1_000,
    ):
        super().__init__(obs_space, action_space, "DoubleDQN")

        self.gamma         = gamma
        self.epsilon       = epsilon_start
        self.epsilon_min   = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.batch_size    = batch_size
        self.update_every  = update_every
        self.target_sync   = target_sync
        self.min_buffer    = min_buffer
        self.step_count    = 0

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[DoubleDQNAgent] device={self.device}  epsilon_start={epsilon_start}")

        self.q_net      = QNetwork(self.obs_shape, self.n_actions).to(self.device)
        self.target_net = QNetwork(self.obs_shape, self.n_actions).to(self.device)
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.q_net.parameters(), lr=lr)
        self.buffer    = ReplayBuffer(buffer_size)

    def select_action(self, obs: np.ndarray) -> int:
        if random.random() < self.epsilon:
            return self.action_space.sample()

        obs_t = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_values = self.q_net(obs_t)
        return int(q_values.argmax(dim=1).item())

    def update(self, obs, action, reward, next_obs, done) -> dict:
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
            "q_net":      self.q_net.state_dict(),
            "target_net": self.target_net.state_dict(),
            "optimizer":  self.optimizer.state_dict(),
            "epsilon":    self.epsilon,
            "step_count": self.step_count,
            "buffer":     self.buffer.buffer,
        }, path)

    def load(self, path: str) -> None:
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        self.q_net.load_state_dict(ckpt["q_net"])
        self.target_net.load_state_dict(ckpt["target_net"])
        self.optimizer.load_state_dict(ckpt["optimizer"])
        self.epsilon    = ckpt["epsilon"]
        self.step_count = ckpt["step_count"]
        self.buffer.buffer = ckpt["buffer"]

    def _learn(self) -> float:
        obs, actions, rewards, next_obs, dones = self.buffer.sample(self.batch_size)

        obs      = torch.tensor(obs,      dtype=torch.float32).to(self.device)
        actions  = torch.tensor(actions,  dtype=torch.int64).to(self.device)
        rewards  = torch.tensor(rewards,  dtype=torch.float32).to(self.device)
        next_obs = torch.tensor(next_obs, dtype=torch.float32).to(self.device)
        dones    = torch.tensor(dones,    dtype=torch.float32).to(self.device)

        q_values = self.q_net(obs).gather(1, actions.unsqueeze(1)).squeeze(1)

        with torch.no_grad():
            next_actions = self.q_net(next_obs).argmax(dim=1)
            next_q = self.target_net(next_obs).gather(
                1, next_actions.unsqueeze(1)
            ).squeeze(1)
            targets = rewards + self.gamma * next_q * (1 - dones)

        loss = nn.functional.smooth_l1_loss(q_values, targets)

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.q_net.parameters(), max_norm=10)
        self.optimizer.step()

        return loss.item()
