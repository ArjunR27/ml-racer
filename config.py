from dataclasses import dataclass
from agents.base_agent import BaseAgent
from agents.random_agent import RandomAgent
from agents.dqn_agent import DQNAgent


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
    render: bool = False


# -----------------------------------------------------------------------
# Edit these — All files read from here
# -----------------------------------------------------------------------
env_cfg = EnvConfig(
    continuous = False,
    grayscale = False,
    frame_stack = 1,
    resize = 96,
    seed = -1,
)

train_cfg = TrainConfig(
    agent = DQNAgent,
    num_episodes = 10000,
    max_steps_per_episode = 500,
    log_interval = 10,
    save_interval = 100,
    checkpoint_dir = "checkpoints",
    render = True,
)