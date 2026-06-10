import argparse

import imageio.v2 as imageio

from config import dqn_env_cfg
from env_setup import make_env
from agents.random_agent import RandomAgent


def main() -> None:
    parser = argparse.ArgumentParser(description="Record a random CarRacing agent to GIF or MP4.")
    parser.add_argument("--output", default="random_agent.gif")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-steps", type=int, default=1000)
    parser.add_argument("--fps", type=int, default=30)
    args = parser.parse_args()

    env = make_env(dqn_env_cfg, render_mode="rgb_array")
    agent = RandomAgent(env.observation_space, env.action_space)

    obs, _ = env.reset(seed=args.seed)
    total_reward = 0.0
    frames = []

    for _ in range(args.max_steps):
        action = agent.select_action(obs)
        obs, reward, terminated, truncated, _ = env.step(action)
        total_reward += reward
        frames.append(env.render())

        if terminated or truncated:
            break

    env.close()

    if args.output.lower().endswith(".gif"):
        imageio.mimsave(args.output, frames, duration=1 / args.fps, loop=0)
    else:
        imageio.mimsave(args.output, frames, fps=args.fps)
    print(f"Saved {len(frames)} frames to {args.output}")
    print(f"Random agent reward: {total_reward:.2f}")


if __name__ == "__main__":
    main()
