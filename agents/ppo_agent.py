import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical, Normal
from gymnasium import spaces

from agents.base_agent import BaseAgent


# ---------------------------------------------------------------------------
# Shared feature extractor
# ---------------------------------------------------------------------------

class CNNExtractor(nn.Module):
    """CNN maps pixel observations to a flat feature vector"""

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
            nn.Conv2d(32,          64, kernel_size=4, stride=2), nn.ReLU(),
            nn.Conv2d(64,          64, kernel_size=3, stride=1), nn.ReLU(),
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
            x = x.permute(0, 3, 1, 2)  # (B, H, W, C) -> (B, C, H, W)
        if x.dim() == 3:               # (B, H, W) -> (B, 1, H, W)
            x = x.unsqueeze(1)
        x = x / 255.0  # normalize pixels to [0, 1]
        return self.fc(self.conv(x))


# ---------------------------------------------------------------------------
# Actor-Critic network
# ---------------------------------------------------------------------------

class ActorCritic(nn.Module):

    def __init__(self, obs_shape: tuple, n_actions: int, continuous: bool,
                 action_space: spaces.Space):
        super().__init__()

        self.continuous = continuous
        self.n_actions  = n_actions

        self.extractor = CNNExtractor(obs_shape, out_features=512)
        feat_dim = self.extractor.out_features

        if continuous:
            self.actor_mean = nn.Linear(feat_dim, n_actions)
            self.log_std    = nn.Parameter(torch.zeros(n_actions))

            low  = torch.tensor(action_space.low,  dtype=torch.float32)
            high = torch.tensor(action_space.high, dtype=torch.float32)
            self.register_buffer("action_scale",  (high - low) / 2.0)
            self.register_buffer("action_offset", (high + low) / 2.0)
        else:
            self.actor_logits = nn.Linear(feat_dim, n_actions)

        self.critic = nn.Linear(feat_dim, 1)

        self._init_weights()

    def _init_weights(self):
        """Orthogonal init keeps gradients well-scaled at the start."""
        for m in self.modules():
            if isinstance(m, (nn.Linear, nn.Conv2d)):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        if self.continuous:
            nn.init.orthogonal_(self.actor_mean.weight, gain=0.01)
        else:
            nn.init.orthogonal_(self.actor_logits.weight, gain=0.01)
        nn.init.orthogonal_(self.critic.weight, gain=1.0)

    def forward(self, obs: torch.Tensor):
        """Returns (distribution, value)."""
        features = self.extractor(obs)
        value = self.critic(features).squeeze(-1)

        if self.continuous:
            mean = torch.tanh(self.actor_mean(features))
            mean = mean * self.action_scale + self.action_offset
            std  = self.log_std.exp().expand_as(mean)
            dist = Normal(mean, std)
        else:
            logits = self.actor_logits(features)
            dist   = Categorical(logits=logits)

        return dist, value

    def get_value(self, obs: torch.Tensor) -> torch.Tensor:
        features = self.extractor(obs)
        return self.critic(features).squeeze(-1)

    def actor_head_parameters(self):
        """Actor head parameters only (excludes trunk)."""
        if self.continuous:
            return list(self.actor_mean.parameters()) + [self.log_std]
        else:
            return list(self.actor_logits.parameters())

    def critic_head_parameters(self):
        """Critic head parameters only (excludes trunk)."""
        return list(self.critic.parameters())


# ---------------------------------------------------------------------------
# Rollout Buffer
# ---------------------------------------------------------------------------

class RolloutBuffer:
    """Stores a fixed window of on-policy transitions collected between updates"""

    def __init__(self, capacity: int, obs_shape: tuple, n_actions: int,
                 continuous: bool, device: torch.device):
        self.capacity   = capacity
        self.obs_shape  = obs_shape
        self.n_actions  = n_actions
        self.continuous = continuous
        self.device     = device
        self.reset()

    def reset(self):
        self.obs      = np.zeros((self.capacity, *self.obs_shape), dtype=np.float32)
        self.actions  = np.zeros((self.capacity, self.n_actions)
                                 if self.continuous else (self.capacity,), dtype=np.float32)
        self.rewards  = np.zeros(self.capacity, dtype=np.float32)
        self.dones    = np.zeros(self.capacity, dtype=np.float32)
        self.values   = np.zeros(self.capacity, dtype=np.float32)
        self.log_probs = np.zeros(self.capacity, dtype=np.float32)
        self.ptr      = 0
        self.full     = False

    def push(self, obs, action, reward, done, value, log_prob):
        self.obs[self.ptr]       = obs
        self.actions[self.ptr]   = action
        self.rewards[self.ptr]   = reward
        self.dones[self.ptr]     = float(done)
        self.values[self.ptr]    = value
        self.log_probs[self.ptr] = log_prob
        self.ptr += 1
        if self.ptr >= self.capacity:
            self.full = True
            self.ptr  = 0

    def is_ready(self) -> bool:
        return self.full or self.ptr == self.capacity

    def compute_returns_and_advantages(self, last_value: float, gamma: float, gae_lambda: float) -> tuple:
        
        size = self.capacity if self.full else self.ptr
        advantages = np.zeros(size, dtype=np.float32)
        returns    = np.zeros(size, dtype=np.float32)

        gae        = 0.0
        next_value = last_value

        for t in reversed(range(size)):
            mask       = 1.0 - self.dones[t]
            delta      = self.rewards[t] + gamma * next_value * mask - self.values[t]
            gae        = delta + gamma * gae_lambda * mask * gae
            advantages[t] = gae
            returns[t]    = gae + self.values[t]
            next_value = self.values[t]

        return (
            torch.tensor(self.obs[:size],       dtype=torch.float32).to(self.device),
            torch.tensor(self.actions[:size],   dtype=torch.float32 if self.continuous
                                                      else torch.int64).to(self.device),
            torch.tensor(self.log_probs[:size], dtype=torch.float32).to(self.device),
            torch.tensor(advantages,            dtype=torch.float32).to(self.device),
            torch.tensor(returns,               dtype=torch.float32).to(self.device),
        )


# ---------------------------------------------------------------------------
# PPO Agent
# ---------------------------------------------------------------------------

class PPOAgent(BaseAgent):
    """
    Hyperparameters:
        lr_actor        learning rate for the trunk + actor head
        lr_critic       learning rate for the critic head only
        gamma           discount factor; how far into the future to care (0–1)
        gae_lambda      GAE smoothing; 0=TD, 1=Monte-Carlo (0–1)
        clip_eps        PPO clip range; how much the policy can shift per update
        vf_coef         weight of the value-function loss in the total loss
        ent_coef        weight of the entropy bonus (encourages exploration)
        max_grad_norm   gradient clipping threshold (prevents exploding gradients)
        n_steps         transitions collected before each PPO update
        n_epochs        gradient passes over the collected batch
        batch_size      mini-batch size used inside each epoch
    """

    def __init__(
        self,
        obs_space: spaces.Space,
        action_space: spaces.Space,
        lr_actor:      float = 2e-4,
        lr_critic:     float = 1e-3,
        gamma:         float = 0.99,
        gae_lambda:    float = 0.95,
        clip_eps:      float = 0.15,
        vf_coef:       float = 0.25,
        ent_coef:      float = 0.05,
        max_grad_norm: float = 0.5,
        n_steps:       int   = 4096,
        n_epochs:      int   = 4,
        batch_size:    int   = 64,
    ):
        super().__init__(obs_space, action_space, "PPO")

        self.gamma         = gamma
        self.gae_lambda    = gae_lambda
        self.clip_eps      = clip_eps
        self.vf_coef       = vf_coef
        self.ent_coef      = ent_coef
        self.max_grad_norm = max_grad_norm
        self.n_steps       = n_steps
        self.n_epochs      = n_epochs
        self.batch_size    = batch_size
        self.step_count    = 0

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[PPOAgent] device={self.device}  continuous={self.continuous}  "
              f"lr_actor={lr_actor}  lr_critic={lr_critic}")

        self.ac = ActorCritic(
            self.obs_shape, self.n_actions, self.continuous, action_space
        ).to(self.device)

    
        self.optimizer = optim.Adam([
            {"params": list(self.ac.extractor.parameters()) +
                       list(self.ac.actor_head_parameters()),
             "lr": lr_actor},
            {"params": self.ac.critic_head_parameters(),
             "lr": lr_critic},
        ], eps=1e-5)

        self.buffer = RolloutBuffer(
            n_steps, self.obs_shape, self.n_actions, self.continuous, self.device
        )

        self._ret_mean  = 0.0
        self._ret_var   = 1.0
        self._ret_count = 0

    def _update_return_stats(self, returns: torch.Tensor):
        """Welford online update of running return mean and variance."""
        batch      = returns.cpu().numpy().astype(np.float64)
        batch_n    = len(batch)
        batch_mean = batch.mean()
        batch_var  = batch.var()

        total_n           = self._ret_count + batch_n
        delta             = batch_mean - self._ret_mean
        self._ret_mean   += delta * batch_n / total_n
        self._ret_var     = (
            self._ret_var   * self._ret_count +
            batch_var       * batch_n +
            delta**2        * self._ret_count * batch_n / total_n
        ) / total_n
        self._ret_count   = total_n

    # ------------------------------------------------------------------
    # BaseAgent interface
    # ------------------------------------------------------------------

    def select_action(self, obs: np.ndarray) -> np.ndarray | int:
        obs_t = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(self.device)
        with torch.no_grad():
            dist, value = self.ac(obs_t)
            action      = dist.sample()
            log_prob    = dist.log_prob(action)
            if self.continuous:
                log_prob = log_prob.sum(-1)

        self._last_value    = value.item()
        self._last_log_prob = log_prob.item()

        if self.continuous:
            return action.squeeze(0).cpu().numpy()
        else:
            return int(action.item())

    def update(self, obs, action, reward, next_obs, done) -> dict:
        self.buffer.push(
            obs, action, reward, done,
            self._last_value, self._last_log_prob
        )
        self.step_count += 1

        if not self.buffer.is_ready():
            return {}

        next_obs_t = torch.tensor(next_obs, dtype=torch.float32).unsqueeze(0).to(self.device)
        with torch.no_grad():
            last_value = self.ac.get_value(next_obs_t).item() * (1.0 - float(done))

        obs_b, acts_b, old_lp_b, adv_b, ret_b = \
            self.buffer.compute_returns_and_advantages(last_value, self.gamma, self.gae_lambda)

        adv_b = (adv_b - adv_b.mean()) / (adv_b.std() + 1e-8)

        metrics = self._ppo_update(obs_b, acts_b, old_lp_b, adv_b, ret_b)
        self.buffer.reset()
        return metrics

    def save(self, path: str) -> None:
        torch.save({
            "ac":         self.ac.state_dict(),
            "optimizer":  self.optimizer.state_dict(),
            "step_count": self.step_count,
            "ret_mean":   self._ret_mean,
            "ret_var":    self._ret_var,
            "ret_count":  self._ret_count,
        }, path)

    def load(self, path: str) -> None:
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        self.ac.load_state_dict(ckpt["ac"])
        self.optimizer.load_state_dict(ckpt["optimizer"])
        self.step_count    = ckpt["step_count"]
        self._ret_mean  = ckpt.get("ret_mean",  0.0)
        self._ret_var   = ckpt.get("ret_var",   1.0)
        self._ret_count = ckpt.get("ret_count", 0)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _ppo_update(self, obs, actions, old_log_probs, advantages, returns) -> dict:
        """Run n_epochs passes over the collected batch in random mini-batches"""
        n = obs.shape[0]
        losses, pg_losses, vf_losses, ent_losses = [], [], [], []

        self._update_return_stats(returns)

        for _ in range(self.n_epochs):
            indices = torch.randperm(n, device=self.device)

            for start in range(0, n, self.batch_size):
                idx = indices[start: start + self.batch_size]

                obs_mb    = obs[idx]
                acts_mb   = actions[idx]
                old_lp_mb = old_log_probs[idx]
                adv_mb    = advantages[idx]
                ret_mb    = returns[idx]

                dist, values = self.ac(obs_mb)

                if self.continuous:
                    new_log_probs = dist.log_prob(acts_mb).sum(-1)
                else:
                    new_log_probs = dist.log_prob(acts_mb)

                entropy = dist.entropy().mean()

                ratio    = torch.exp(new_log_probs - old_lp_mb)
                pg_loss1 = ratio * adv_mb
                pg_loss2 = torch.clamp(ratio, 1 - self.clip_eps,
                                               1 + self.clip_eps) * adv_mb
                pg_loss  = -torch.min(pg_loss1, pg_loss2).mean()

                ret_std     = np.sqrt(self._ret_var) + 1e-8
                ret_mb_norm = (ret_mb - self._ret_mean) / ret_std
                vf_loss = nn.functional.mse_loss(values, ret_mb_norm)

                loss = pg_loss + self.vf_coef * vf_loss - self.ent_coef * entropy

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.ac.parameters(), self.max_grad_norm)
                self.optimizer.step()

                losses.append(loss.item())
                pg_losses.append(pg_loss.item())
                vf_losses.append(vf_loss.item())
                ent_losses.append(entropy.item())

        return {
            "loss":     np.mean(losses),
            "pg_loss":  np.mean(pg_losses),
            "vf_loss":  np.mean(vf_losses),
            "entropy":  np.mean(ent_losses),
        }