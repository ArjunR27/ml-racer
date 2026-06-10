from config import dqn_env_cfg
from env_setup import make_env
from agents.random_agent import RandomAgent


def main() -> None:
    env = make_env(dqn_env_cfg, render_mode="human")
    agent = RandomAgent(env.observation_space, env.action_space)

    obs, _ = env.reset(seed=42)
    total_reward = 0.0

    for _ in range(1000):
        action = agent.select_action(obs)
        obs, reward, terminated, truncated, _ = env.step(action)
        total_reward += reward

        if terminated or truncated:
            break

    env.close()
    print(f"Random agent reward: {total_reward:.2f}")


if __name__ == "__main__":
    main()
