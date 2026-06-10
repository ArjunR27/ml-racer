from dataclasses import dataclass
from typing import Sequence
from agents.base_agent import BaseAgent
from agents.random_agent import RandomAgent
from agents.dqn_agent import DQNAgent
from agents.ppo_agent import PPOAgent
from agents.sac_agent import SACAgent

try:
    from agents.double_dqn_agent import DoubleDQNAgent
except ModuleNotFoundError:
    DoubleDQNAgent = None


@dataclass
class EnvConfig:
    continuous: bool     # False = 5 discrete actions, True = 3 continuous
    grayscale: bool      # convert RGB frames to single-channel
    frame_stack: int     # number of consecutive frames stacked (1 = no stacking)
    resize: int          # resize obs to (resize x resize); None = keep 96x96
    seed: int            # -1 for random seed (general racing). Set a value for mastering a track


@dataclass
class TrainConfig:
    agent: type[BaseAgent]
    num_episodes: int 
    max_steps_per_episode: int 
    log_interval: int 
    save_interval: int
    checkpoint_dir: str
    training_seeds: Sequence[int] = (42,)
    progress_interval_steps: int = 100
    render: bool = False
    render_eval_interval: int = 0  # 0 disables periodic visual evaluations
    render_eval_episodes: int = 1


# -----------------------------------------------------------------------
# Named configs
# -----------------------------------------------------------------------
ppo_env_cfg = EnvConfig(
    continuous = False,
    grayscale = True,
    frame_stack = 4,
    resize = 84,
    seed = 42,
)

ppo_train_cfg = TrainConfig(
    agent = PPOAgent,
    num_episodes = 10000,
    max_steps_per_episode = 500,
    log_interval = 10,
    save_interval = 100,
    checkpoint_dir = "checkpoints",
    render = False,
    render_eval_interval = 1000,
    render_eval_episodes = 1,
)

sac_env_cfg = EnvConfig(
    continuous = True,
    grayscale = True,
    frame_stack = 4,
    resize = 84,
    seed = 42,
)

sac_train_cfg = TrainConfig(
    agent = SACAgent,
    num_episodes = 10000,
    max_steps_per_episode = 500,
    log_interval = 10,
    save_interval = 100,
    checkpoint_dir = "checkpoints",
    render = False,
    render_eval_interval = 1000,
    render_eval_episodes = 1,
)

dqn_env_cfg = EnvConfig(
    continuous = False,
    grayscale = False,
    frame_stack = 1,
    resize = 96,
    seed = 42,
)

dqn_train_cfg = TrainConfig(
    agent = DQNAgent,
    num_episodes = 10000,
    max_steps_per_episode = 500,
    log_interval = 10,
    save_interval = 100,
    checkpoint_dir = "checkpoints",
    render = False,
    render_eval_interval = 1000,
    render_eval_episodes = 1,
)

double_dqn_env_cfg = EnvConfig(
    continuous = False,
    grayscale = False,
    frame_stack = 1,
    resize = 96,
    seed = 42,
)

double_dqn_train_cfg = TrainConfig(
    agent = DoubleDQNAgent,
    num_episodes = 10000,
    max_steps_per_episode = 500,
    log_interval = 10,
    save_interval = 100,
    checkpoint_dir = "checkpoints",
    render = False,
    render_eval_interval = 1000,
    render_eval_episodes = 1,
)


# -----------------------------------------------------------------------
# Active training config. Change these two lines to switch algorithms.
# -----------------------------------------------------------------------
env_cfg = sac_env_cfg
train_cfg = sac_train_cfg
