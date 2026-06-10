import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot DQN checkpoint evaluation CSV.")
    parser.add_argument("--input", default="dqn_checkpoint_eval_seed42.csv")
    parser.add_argument("--output-dir", default="dqn_eval_graphs")
    parser.add_argument("--title", default="DQN Training Progress on Seed 42")
    args = parser.parse_args()

    df = pd.read_csv(args.input)
    df = df.dropna(subset=["episode"]).copy()
    df["episode"] = df["episode"].astype(int)
    df["end_reason"] = df["end_reason"].replace({
        "terminated": "no_progress",
        "time_limit": "max_steps",
        "max_limit": "max_steps",
        "no progress": "no_progress",
    })
    df = df.sort_values("episode")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(10, 5))
    plt.plot(df["episode"], df["reward"], marker="o", linewidth=2)
    plt.title(f"{args.title}: Reward")
    plt.xlabel("Training Episode")
    plt.ylabel("Evaluation Reward on Seed 42")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    reward_path = output_dir / "dqn_training_reward.png"
    plt.savefig(reward_path, dpi=200)
    plt.close()

    plt.figure(figsize=(10, 5))
    plt.plot(df["episode"], df["progress_pct"], marker="o", linewidth=2, color="green")
    plt.title(f"{args.title}: Track Progress")
    plt.xlabel("Training Episode")
    plt.ylabel("Track Progress (%) on Seed 42")
    plt.ylim(0, 105)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    progress_path = output_dir / "dqn_training_progress.png"
    plt.savefig(progress_path, dpi=200)
    plt.close()

    end_counts = df["end_reason"].value_counts()
    plt.figure(figsize=(8, 5))
    end_counts.plot(kind="bar", color="slateblue")
    plt.title(f"{args.title}: Episode End Reasons")
    plt.xlabel("End Reason")
    plt.ylabel("Checkpoint Count")
    plt.xticks(rotation=0)
    plt.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    end_path = output_dir / "dqn_training_end_reasons.png"
    plt.savefig(end_path, dpi=200)
    plt.close()

    print(f"Saved reward graph: {reward_path}")
    print(f"Saved progress graph: {progress_path}")
    print(f"Saved end reason graph: {end_path}")


if __name__ == "__main__":
    main()
