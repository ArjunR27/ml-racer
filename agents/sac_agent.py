"""SAC (Soft Actor-Critic) agent for continuous CarRacing control.

SAC is an off-policy actor-critic algorithm. It learns a stochastic continuous
policy, two Q-functions, and target Q-functions. The entropy term encourages
exploration while training from replayed transitions.
"""

from collections import deque
import random

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from gymnasium import spaces

from agents.base_agent import BaseAgent


LOG_STD_MIN = -20
LOG_STD_MAX = 2
EPS = 1e-6


class CNNEncoder(nn.Module):
    """CNN feature extractor for pixel observations."""

    def __init__(self, obs_shape: tuple, out_features: int = 512):
        super().__init__()

        if len(obs_shape) == 2:
            in_channels = 1
            h, w = obs_shape
            self.needs_permute = False
        elif len(obs_shape) == 3 and obs_shape[0] <= 16:
            in_channels, h, w = obs_shape
            self.needs_permute = False
        else:
            h, w, in_channels = obs_shape
            self.needs_permute = True

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=8, stride=4), nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2), nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1), nn.ReLU(),
        )

        with torch.no_grad():
            dummy = torch.zeros(1, in_channels, h, w)
            conv_out = int(np.prod(self.conv(dummy).shape))

        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(conv_out, out_features),
            nn.ReLU(),
        )
        self.out_features = out_features

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.needs_permute:
            x = x.permute(0, 3, 1, 2)
        if x.dim() == 3:
            x = x.unsqueeze(1)
        x = x / 255.0
        return self.fc(self.conv(x))


class SquashedGaussianActor(nn.Module):
    """Gaussian policy squashed through tanh and scaled to action bounds."""

    def __init__(self, obs_shape: tuple, action_space: spaces.Box):
        super().__init__()

        action_dim = int(np.prod(action_space.shape))
        self.encoder = CNNEncoder(obs_shape, out_features=512)
        self.mean = nn.Linear(self.encoder.out_features, action_dim)
        self.log_std = nn.Linear(self.encoder.out_features, action_dim)

        action_low = torch.tensor(action_space.low, dtype=torch.float32)
        action_high = torch.tensor(action_space.high, dtype=torch.float32)
        self.register_buffer("action_scale", (action_high - action_low) / 2.0)
        self.register_buffer("action_offset", (action_high + action_low) / 2.0)

    def forward(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.encoder(obs)
        mean = self.mean(features)
        log_std = self.log_std(features).clamp(LOG_STD_MIN, LOG_STD_MAX)
        return mean, log_std

    def sample(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        mean, log_std = self(obs)
        std = log_std.exp()
        normal = torch.distributions.Normal(mean, std)
        raw_action = normal.rsample()
        squashed = torch.tanh(raw_action)
        action = squashed * self.action_scale + self.action_offset

        log_prob = normal.log_prob(raw_action)
        log_prob -= torch.log(1.0 - squashed.pow(2) + EPS)
        log_prob = log_prob.sum(dim=-1, keepdim=True)
        return action, log_prob

    def mean_action(self, obs: torch.Tensor) -> torch.Tensor:
        mean, _ = self(obs)
        squashed = torch.tanh(mean)
        return squashed * self.action_scale + self.action_offset


class SoftQNetwork(nn.Module):
    """Q(s, a) network with its own visual encoder."""

    def __init__(self, obs_shape: tuple, action_dim: int):
        super().__init__()
        self.encoder = CNNEncoder(obs_shape, out_features=512)
        self.q = nn.Sequential(
            nn.Linear(self.encoder.out_features + action_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 1),
        )

    def forward(self, obs: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        features = self.encoder(obs)
        x = torch.cat([features, action], dim=1)
        return self.q(x)


class SACReplayBuffer:
    """Replay buffer for continuous-action transitions."""

    def __init__(self, capacity: int):
        self.buffer = deque(maxlen=capacity)

    def push(self, obs, action, reward, next_obs, done):
        self.buffer.append((obs, action, reward, next_obs, done))

    def sample(self, batch_size: int):
        batch = random.sample(self.buffer, batch_size)
        obs, actions, rewards, next_obs, dones = zip(*batch)
        return (
            np.array(obs, dtype=np.float32),
            np.array(actions, dtype=np.float32),
            np.array(rewards, dtype=np.float32).reshape(-1, 1),
            np.array(next_obs, dtype=np.float32),
            np.array(dones, dtype=np.float32).reshape(-1, 1),
        )

    def __len__(self):
        return len(self.buffer)


class SACAgent(BaseAgent):
    """Soft Actor-Critic for continuous action spaces."""

    def __init__(
        self,
        obs_space: spaces.Space,
        action_space: spaces.Space,
        actor_lr: float = 3e-4,
        critic_lr: float = 3e-4,
        alpha_lr: float = 3e-4,
        gamma: float = 0.99,
        tau: float = 0.005,
        buffer_size: int = 100_000,
        batch_size: int = 64,
        min_buffer: int = 2_000,
        update_every: int = 1,
        policy_delay: int = 1,
    ):
        if not isinstance(action_space, spaces.Box):
            raise ValueError("SACAgent requires a continuous Box action space.")

        super().__init__(obs_space, action_space, "SAC")

        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size
        self.min_buffer = min_buffer
        self.update_every = update_every
        self.policy_delay = policy_delay
        self.step_count = 0

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[SACAgent] device={self.device}  action_dim={self.n_actions}")

        self.actor = SquashedGaussianActor(self.obs_shape, action_space).to(self.device)
        self.q1 = SoftQNetwork(self.obs_shape, self.n_actions).to(self.device)
        self.q2 = SoftQNetwork(self.obs_shape, self.n_actions).to(self.device)
        self.target_q1 = SoftQNetwork(self.obs_shape, self.n_actions).to(self.device)
        self.target_q2 = SoftQNetwork(self.obs_shape, self.n_actions).to(self.device)
        self.target_q1.load_state_dict(self.q1.state_dict())
        self.target_q2.load_state_dict(self.q2.state_dict())
        self.target_q1.eval()
        self.target_q2.eval()

        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.q1_optimizer = optim.Adam(self.q1.parameters(), lr=critic_lr)
        self.q2_optimizer = optim.Adam(self.q2.parameters(), lr=critic_lr)

        self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
        self.alpha_optimizer = optim.Adam([self.log_alpha], lr=alpha_lr)
        self.target_entropy = -float(self.n_actions)

        self.buffer = SACReplayBuffer(buffer_size)

    @property
    def alpha(self) -> torch.Tensor:
        return self.log_alpha.exp()

    def select_action(self, obs: np.ndarray) -> np.ndarray:
        obs_t = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(self.device)
        with torch.no_grad():
            action, _ = self.actor.sample(obs_t)
        return action.squeeze(0).cpu().numpy().astype(np.float32)

    def select_eval_action(self, obs: np.ndarray) -> np.ndarray:
        obs_t = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(self.device)
        with torch.no_grad():
            action = self.actor.mean_action(obs_t)
        return action.squeeze(0).cpu().numpy().astype(np.float32)

    def update(self, obs, action, reward, next_obs, done) -> dict:
        self.buffer.push(obs, action, reward, next_obs, done)
        self.step_count += 1

        if len(self.buffer) < self.min_buffer:
            return {}
        if self.step_count % self.update_every != 0:
            return {}

        metrics = self._learn()
        self._soft_update(self.target_q1, self.q1)
        self._soft_update(self.target_q2, self.q2)
        return metrics

    def save(self, path: str) -> None:
        torch.save({
            "agent_name": self.name,
            "actor": self.actor.state_dict(),
            "q1": self.q1.state_dict(),
            "q2": self.q2.state_dict(),
            "target_q1": self.target_q1.state_dict(),
            "target_q2": self.target_q2.state_dict(),
            "actor_optimizer": self.actor_optimizer.state_dict(),
            "q1_optimizer": self.q1_optimizer.state_dict(),
            "q2_optimizer": self.q2_optimizer.state_dict(),
            "alpha_optimizer": self.alpha_optimizer.state_dict(),
            "log_alpha": self.log_alpha.detach().cpu(),
            "step_count": self.step_count,
            "buffer": self.buffer.buffer,
        }, path)

    def load(self, path: str) -> None:
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        self.actor.load_state_dict(ckpt["actor"])
        self.q1.load_state_dict(ckpt["q1"])
        self.q2.load_state_dict(ckpt["q2"])
        self.target_q1.load_state_dict(ckpt["target_q1"])
        self.target_q2.load_state_dict(ckpt["target_q2"])
        self.actor_optimizer.load_state_dict(ckpt["actor_optimizer"])
        self.q1_optimizer.load_state_dict(ckpt["q1_optimizer"])
        self.q2_optimizer.load_state_dict(ckpt["q2_optimizer"])
        self.alpha_optimizer.load_state_dict(ckpt["alpha_optimizer"])
        self.log_alpha.data.copy_(ckpt["log_alpha"].to(self.device))
        self.step_count = ckpt["step_count"]
        self.buffer.buffer = ckpt["buffer"]

    def _learn(self) -> dict:
        obs, actions, rewards, next_obs, dones = self.buffer.sample(self.batch_size)

        obs = torch.tensor(obs, dtype=torch.float32).to(self.device)
        actions = torch.tensor(actions, dtype=torch.float32).to(self.device)
        rewards = torch.tensor(rewards, dtype=torch.float32).to(self.device)
        next_obs = torch.tensor(next_obs, dtype=torch.float32).to(self.device)
        dones = torch.tensor(dones, dtype=torch.float32).to(self.device)

        with torch.no_grad():
            next_actions, next_log_probs = self.actor.sample(next_obs)
            target_q1 = self.target_q1(next_obs, next_actions)
            target_q2 = self.target_q2(next_obs, next_actions)
            target_q = torch.min(target_q1, target_q2) - self.alpha * next_log_probs
            q_target = rewards + self.gamma * (1.0 - dones) * target_q

        q1_loss = nn.functional.mse_loss(self.q1(obs, actions), q_target)
        q2_loss = nn.functional.mse_loss(self.q2(obs, actions), q_target)

        self.q1_optimizer.zero_grad()
        q1_loss.backward()
        nn.utils.clip_grad_norm_(self.q1.parameters(), max_norm=10)
        self.q1_optimizer.step()

        self.q2_optimizer.zero_grad()
        q2_loss.backward()
        nn.utils.clip_grad_norm_(self.q2.parameters(), max_norm=10)
        self.q2_optimizer.step()

        actor_loss_value = 0.0
        alpha_loss_value = 0.0
        if self.step_count % self.policy_delay == 0:
            new_actions, log_probs = self.actor.sample(obs)
            q_new = torch.min(self.q1(obs, new_actions), self.q2(obs, new_actions))
            actor_loss = (self.alpha.detach() * log_probs - q_new).mean()

            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=10)
            self.actor_optimizer.step()
            actor_loss_value = actor_loss.item()

            alpha_loss = -(self.log_alpha * (log_probs + self.target_entropy).detach()).mean()
            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()
            alpha_loss_value = alpha_loss.item()

        return {
            "q1_loss": q1_loss.item(),
            "q2_loss": q2_loss.item(),
            "actor_loss": actor_loss_value,
            "alpha_loss": alpha_loss_value,
            "alpha": self.alpha.item(),
        }

    def _soft_update(self, target: nn.Module, source: nn.Module) -> None:
        for target_param, source_param in zip(target.parameters(), source.parameters()):
            target_param.data.mul_(1.0 - self.tau)
            target_param.data.add_(self.tau * source_param.data)
