import numpy as np
from gymnasium import spaces

from agents.base_agent import BaseAgent


class RandomAgent(BaseAgent):
    """ Makes random actions. Doesn't learn. Just a baseline """

    def __init__(self, obs_space: spaces.Space, action_space: spaces.Space):
        super().__init__(obs_space, action_space, "Random")

    def select_action(self, obs: np.ndarray) -> np.ndarray | int:
        return self.action_space.sample()

    def update(self, obs, action, reward, next_obs, done) -> dict:
        return {}

    def save(self, path: str) -> None:
        pass

    def load(self, path: str) -> None:
        pass