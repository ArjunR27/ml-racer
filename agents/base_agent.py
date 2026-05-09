from abc import ABC, abstractmethod

import numpy as np
from gymnasium import spaces


class BaseAgent(ABC):
    """Common interface that every agent must implement.

    Agents receive the environment's observation and action spaces at
    construction time, so they can adapt their networks/logic without
    any hardcoded assumptions about shape, size, or action type.

    The training loop only ever calls:
        select_action / update / save / load
    """

    def __init__(self, obs_space: spaces.Space, action_space: spaces.Space, name: str):
        self.obs_space = obs_space
        self.action_space = action_space
        self.name = name

        # Derived convenience flags — use these inside subclasses
        self.continuous = isinstance(action_space, spaces.Box)

        if self.continuous:
            self.n_actions = int(np.prod(action_space.shape))
        else:
            self.n_actions = int(action_space.n)  # e.g. 5 for discrete CarRacing

        # Observation shape, channel count, flat size — whichever is relevant
        self.obs_shape = obs_space.shape           # e.g. (84, 84, 4)
        self.obs_size = int(np.prod(obs_space.shape))

        print(f"[Agent] obs_shape={self.obs_shape}  "
              f"n_actions={self.n_actions}  continuous={self.continuous}")

    # ------------------------------------------------------------------
    # Required interface
    # ------------------------------------------------------------------

    @abstractmethod
    def select_action(self, obs: np.ndarray) -> np.ndarray | int:
        """Choose an action for the given observation.

        Args:
            obs: A single observation from the environment.

        Returns:
            An integer for discrete action spaces, or a float array for
            continuous ones. Match the env's action_space dtype/shape.
        """

    @abstractmethod
    def update(
        self,
        obs: np.ndarray,
        action: np.ndarray | int,
        reward: float,
        next_obs: np.ndarray,
        done: bool,
    ) -> dict:
        """Learn from one transition.

        Called after every env step. Agents that learn in batches (DQN,
        PPO) should store the transition in a buffer here and only update
        when ready; the return dict lets them signal that.

        Args:
            obs:      Observation before the action.
            action:   Action taken.
            reward:   Scalar reward received.
            next_obs: Observation after the action.
            done:     True if the episode ended.

        Returns:
            Dict of training metrics (e.g. {"loss": 0.42}).
            Return an empty dict if no update was performed this step.
        """

    @abstractmethod
    def save(self, path: str) -> None:
        """Persist agent state (weights, buffers, etc.) to disk."""

    @abstractmethod
    def load(self, path: str) -> None:
        """Restore agent state from a checkpoint saved by save()."""