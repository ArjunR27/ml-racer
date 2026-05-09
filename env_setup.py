import gymnasium as gym
from gymnasium.wrappers import (
    GrayscaleObservation,
    ResizeObservation,
    FrameStackObservation,
)
import random
import sys

from config import EnvConfig


def make_env(cfg: EnvConfig, render_mode: str = None) -> gym.Env:
    """Build the CarRacing environment from an EnvConfig.

    Designed to abstract env information from the agent so the env can
    change without needing to change any agent code

    Args:
        cfg:         EnvConfig dataclass controlling all env options.
        render_mode: Override render mode (e.g. "human" to watch, None for training).

    Returns:
        A fully wrapped gymnasium.Env.
    """
    env = gym.make(
        "CarRacing-v3",
        continuous=cfg.continuous,
        domain_randomize=False,
        render_mode=render_mode,
    )

    if cfg.grayscale:
        env = GrayscaleObservation(env, keep_dim=False)

    if cfg.resize is not None:
        env = ResizeObservation(env, (cfg.resize, cfg.resize))

    if cfg.frame_stack > 1:
        env = FrameStackObservation(env, cfg.frame_stack)

    if cfg.seed == -1:
        env.reset(seed=random.randint(0, sys.maxsize))
    else :
        env.reset(seed=cfg.seed)
    return env