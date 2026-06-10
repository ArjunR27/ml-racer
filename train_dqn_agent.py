"""Train the DQN agent regardless of the active config.

This is a small entry point that reuses the shared training loop from
train_agent.py, but passes the DQN-specific environment and training configs.
"""

from config import dqn_env_cfg, dqn_train_cfg
from train_agent import train


if __name__ == "__main__":
    train(dqn_env_cfg, dqn_train_cfg)
