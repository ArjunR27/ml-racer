"""DQN (Deep Q-Network) agent.

The simplest RL agent that can learn from pixel observations.

Core idea:
    Q(s, a) = expected total future reward when taking action a in state s.
    Train a neural network to predict Q(s, a) for all actions at once.
    Always pick the action with the highest Q value (with some random exploration).

Three components:
    1. Q-Network      -- CNN that maps an obs to Q-values for each action
    2. Replay Buffer  -- stores past transitions; sample random batches to train on
    3. Target Network -- a delayed copy of the Q-network for stable training targets
"""

import random
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from gymnasium import spaces

from agents.base_agent import BaseAgent


# ---------------------------------------------------------------------------
# Q-Network
# ---------------------------------------------------------------------------

class QNetwork(nn.Module):
    """CNN that takes an obs and outputs one Q-value per action.

    Handles all obs shapes produced by EnvConfig:
        grayscale + frame_stack=4  -> (4, 84, 84)   channels-first already
        colour, no stack           -> (96, 96, 3)   permuted to (3, 96, 96)
    """

    def __init__(self, obs_shape: tuple, n_actions: int):
        super().__init__()

        if len(obs_shape) == 2:
            # (H, W) single grayscale frame
            in_channels = 1
            h, w = obs_shape
            self.needs_permute = False
        elif len(obs_shape) == 3 and obs_shape[0] <= 16:
            # (frames, H, W) grayscale stack — already channels-first
            in_channels, h, w = obs_shape
            self.needs_permute = False
        else:
            # (H, W, C) colour — needs permuting to (C, H, W)
            h, w, in_channels = obs_shape
            self.needs_permute = True

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=8, stride=4), nn.ReLU(),
            nn.Conv2d(32,          64, kernel_size=4, stride=2), nn.ReLU(),
            nn.Conv2d(64,          64, kernel_size=3, stride=1), nn.ReLU(),
        )

        # Compute flattened conv output size with a dummy forward pass
        with torch.no_grad():
            dummy = torch.zeros(1, in_channels, h, w)
            conv_out = int(np.prod(self.conv(dummy).shape))

        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(conv_out, 512), nn.ReLU(),
            nn.Linear(512, n_actions),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.needs_permute:
            x = x.permute(0, 3, 1, 2)  # (B, H, W, C) -> (B, C, H, W)
        return self.fc(self.conv(x))


# ---------------------------------------------------------------------------
# Replay Buffer
# ---------------------------------------------------------------------------

class ReplayBuffer:
    """Ring buffer of (obs, action, reward, next_obs, done) transitions.

    Training on random batches rather than sequential steps breaks
    correlations between consecutive frames, which stabilises learning.
    """

    def __init__(self, capacity: int):
        self.buffer = deque(maxlen=capacity)

    def push(self, obs, action, reward, next_obs, done):
        self.buffer.append((obs, action, reward, next_obs, done))

    def sample(self, batch_size: int):
        batch = random.sample(self.buffer, batch_size)
        obs, actions, rewards, next_obs, dones = zip(*batch)
        return (
            np.array(obs,      dtype=np.float32),
            np.array(actions,  dtype=np.int64),
            np.array(rewards,  dtype=np.float32),
            np.array(next_obs, dtype=np.float32),
            np.array(dones,    dtype=np.float32),
        )

    def __len__(self):
        return len(self.buffer)


# ---------------------------------------------------------------------------
# DQN Agent
# ---------------------------------------------------------------------------

class DQNAgent(BaseAgent):
    """DQN with experience replay and a target network.

    Hyperparameters:
        lr             learning rate for the Q-network
        gamma          discount factor (how much future rewards matter; 0-1)
        epsilon_start  initial exploration rate (1.0 = fully random)
        epsilon_min    floor for exploration rate
        epsilon_decay  multiplicative decay applied after every step
        buffer_size    max transitions stored in the replay buffer
        batch_size     transitions sampled per gradient update
        update_every   run a gradient update every N steps
        target_sync    copy Q-net weights to target net every N steps
        min_buffer     don't start training until buffer has this many transitions
    """

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
        super().__init__(obs_space, action_space, "DQN")

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
        print(f"[DQNAgent] device={self.device}  epsilon_start={epsilon_start}")

        # Two identical networks: one we train, one we use for stable targets
        self.q_net      = QNetwork(self.obs_shape, self.n_actions).to(self.device)
        self.target_net = QNetwork(self.obs_shape, self.n_actions).to(self.device)
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.q_net.parameters(), lr=lr)
        self.buffer    = ReplayBuffer(buffer_size)

    # ------------------------------------------------------------------
    # BaseAgent interface
    # ------------------------------------------------------------------

    def select_action(self, obs: np.ndarray) -> int:
        """Epsilon-greedy: random action with probability epsilon,
        otherwise the action with the highest predicted Q-value."""
        if random.random() < self.epsilon:
            return self.action_space.sample()

        obs_t = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_values = self.q_net(obs_t)
        return int(q_values.argmax(dim=1).item())

    def update(self, obs, action, reward, next_obs, done) -> dict:
        self.buffer.push(obs, action, reward, next_obs, done)
        self.step_count += 1

        # Wait until the buffer has enough transitions
        if len(self.buffer) < self.min_buffer:
            return {}
        # Only update every N steps
        if self.step_count % self.update_every != 0:
            return {}
        
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

        loss = self._learn()

        # Periodically copy learned weights into the target network
        if self.step_count % self.target_sync == 0:
            self.target_net.load_state_dict(self.q_net.state_dict())

        return {"loss": loss, "epsilon": self.epsilon}

    def save(self, path: str) -> None:
        torch.save({
            "q_net":      self.q_net.state_dict(),
            "target_net": self.target_net.state_dict(),
            "optimizer":  self.optimizer.state_dict(),
            "epsilon":    self.epsilon,
            "step_count": self.step_count,
            "buffer": self.buffer.buffer
        }, path)

    def load(self, path: str) -> None:
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        self.q_net.load_state_dict(ckpt["q_net"])
        self.target_net.load_state_dict(ckpt["target_net"])
        self.optimizer.load_state_dict(ckpt["optimizer"])
        self.epsilon    = ckpt["epsilon"]
        self.step_count = ckpt["step_count"]
        self.buffer.buffer = ckpt["buffer"]

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _learn(self) -> float:
        """One gradient step using the Bellman equation as a target.

        Bellman equation:
            Q(s, a) = r + gamma * max_a' Q_target(s', a')

        We minimise the difference between what our Q-net predicts and
        what the Bellman equation says it should be.
        """
        obs, actions, rewards, next_obs, dones = self.buffer.sample(self.batch_size)

        obs      = torch.tensor(obs,      dtype=torch.float32).to(self.device)
        actions  = torch.tensor(actions,  dtype=torch.int64).to(self.device)
        rewards  = torch.tensor(rewards,  dtype=torch.float32).to(self.device)
        next_obs = torch.tensor(next_obs, dtype=torch.float32).to(self.device)
        dones    = torch.tensor(dones,    dtype=torch.float32).to(self.device)

        # Q-values for the specific actions that were taken
        q_values = self.q_net(obs).gather(1, actions.unsqueeze(1)).squeeze(1)

        # Target Q-values: r + gamma * max Q_target(s')  (zero if episode ended)
        with torch.no_grad():
            next_q  = self.target_net(next_obs).max(dim=1).values
            targets = rewards + self.gamma * next_q * (1 - dones)

        # Smooth L1 (Huber) loss — less sensitive to outlier rewards than MSE
        loss = nn.functional.smooth_l1_loss(q_values, targets)

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.q_net.parameters(), max_norm=10)
        self.optimizer.step()

        return loss.item()