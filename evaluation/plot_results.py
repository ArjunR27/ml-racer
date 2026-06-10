"""Create performance graphs from evaluate_agents.py CSV output."""

from __future__ import annotations

import argparse
import os
import tempfile
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "ml-racer-matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(Path(tempfile.gettempdir()) / "ml-racer-cache"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


SUMMARY_METRICS = [
    ("finish_rate", "Finish Rate", "finish_rate.png", "Completion rate"),
    ("avg_progress", "Average Progress (%)", "avg_progress.png", "Average track progress"),
    ("avg_reward", "Average Reward", "avg_reward.png", "Average reward"),
    ("avg_steps", "Average Steps", "avg_steps.png", "Average episode length"),
]
AGENT_SUMMARY_METRICS = [
    ("finish_rate", "Finish Rate", "agent_finish_rate.png", "Completion rate by agent"),
    ("avg_progress", "Average Progress (%)", "agent_avg_progress.png", "Average track progress by agent"),
    ("avg_reward", "Average Reward", "agent_avg_reward.png", "Average reward by agent"),
    ("avg_steps", "Average Steps", "agent_avg_steps.png", "Average episode length by agent"),
]
AGENT_COLORS = {
    "PPO": "#f28e2b",
    "DQN": "#4e79a7",
    "DoubleDQN": "#59a14f",
    "Unknown": "#9c755f",
}


def _load_results(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    required_columns = {
        "checkpoint",
        "seed",
        "reward",
        "steps",
        "lap_finished",
        "progress_percent",
        "end_reason",
    }
    missing = required_columns - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(sorted(missing))}")

    df["lap_finished"] = df["lap_finished"].astype(str).str.lower().isin(["true", "1"])
    df["agent"] = df["checkpoint"].map(_agent_from_checkpoint)
    return df


def _agent_from_checkpoint(checkpoint: str) -> str:
    name = checkpoint.lower()
    if name.startswith(("doubledqn", "double_dqn")):
        return "DoubleDQN"
    if name.startswith("ppo"):
        return "PPO"
    if name.startswith("dqn"):
        return "DQN"
    return "Unknown"


def _summary(df: pd.DataFrame, group_col: str = "checkpoint") -> pd.DataFrame:
    return (
        df.groupby(group_col, as_index=False)
        .agg(
            finish_rate=("lap_finished", "mean"),
            avg_progress=("progress_percent", "mean"),
            avg_reward=("reward", "mean"),
            avg_steps=("steps", "mean"),
            episodes=("seed", "count"),
        )
        .sort_values(
            ["finish_rate", "avg_progress", "avg_reward"],
            ascending=False,
        )
    )


def _write_summary(summary_df: pd.DataFrame, output_dir: Path) -> None:
    summary_path = output_dir / "summary.csv"
    summary_df.to_csv(summary_path, index=False)


def _write_agent_summary(summary_df: pd.DataFrame, output_dir: Path) -> None:
    summary_path = output_dir / "agent_summary.csv"
    summary_df.to_csv(summary_path, index=False)


def _save_bar_chart(
    summary_df: pd.DataFrame,
    metric: str,
    ylabel: str,
    title: str,
    output_path: Path,
    x_col: str = "checkpoint",
    palette: dict[str, str] | None = None,
) -> None:
    plt.figure(figsize=(10, 5))
    if palette is None:
        ax = sns.barplot(data=summary_df, x=x_col, y=metric, color="#3f7f93")
    else:
        ax = sns.barplot(data=summary_df, x=x_col, y=metric, hue=x_col, palette=palette, legend=False)
    ax.set_title(title)
    ax.set_xlabel("Agent" if x_col == "agent" else "Checkpoint")
    ax.set_ylabel(ylabel)
    ax.tick_params(axis="x", rotation=30)
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()


def _save_reward_distribution(df: pd.DataFrame, output_path: Path) -> None:
    plt.figure(figsize=(10, 5))
    ax = sns.boxplot(data=df, x="checkpoint", y="reward", color="#9c6b54")
    ax.set_title("Reward Distribution")
    ax.set_xlabel("Checkpoint")
    ax.set_ylabel("Reward")
    ax.tick_params(axis="x", rotation=30)
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()


def _save_agent_reward_distribution(df: pd.DataFrame, output_path: Path) -> None:
    plt.figure(figsize=(10, 5))
    ax = sns.boxplot(data=df, x="agent", y="reward", hue="agent", palette=AGENT_COLORS, legend=False)
    ax.set_title("Reward Distribution by Agent")
    ax.set_xlabel("Agent")
    ax.set_ylabel("Reward")
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()


def _save_progress_by_seed(df: pd.DataFrame, output_path: Path) -> None:
    plt.figure(figsize=(11, 5))
    ax = sns.lineplot(
        data=df,
        x="seed",
        y="progress_percent",
        hue="checkpoint",
        marker="o",
    )
    ax.set_title("Track Progress by Seed")
    ax.set_xlabel("Seed")
    ax.set_ylabel("Progress (%)")
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()


def _save_agent_progress_by_seed(df: pd.DataFrame, output_path: Path) -> None:
    plt.figure(figsize=(11, 5))
    ax = sns.lineplot(
        data=df,
        x="seed",
        y="progress_percent",
        hue="agent",
        palette=AGENT_COLORS,
        marker="o",
    )
    ax.set_title("Track Progress by Seed and Agent")
    ax.set_xlabel("Seed")
    ax.set_ylabel("Progress (%)")
    ax.legend(title="Agent")
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()


def _save_end_reason_chart(df: pd.DataFrame, output_path: Path) -> None:
    counts = (
        df.groupby(["checkpoint", "end_reason"])
        .size()
        .reset_index(name="episodes")
    )

    plt.figure(figsize=(10, 5))
    ax = sns.barplot(
        data=counts,
        x="checkpoint",
        y="episodes",
        hue="end_reason",
    )
    ax.set_title("Episode End Reasons")
    ax.set_xlabel("Checkpoint")
    ax.set_ylabel("Episodes")
    ax.tick_params(axis="x", rotation=30)
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()


def _save_agent_end_reason_chart(df: pd.DataFrame, output_path: Path) -> None:
    counts = (
        df.groupby(["agent", "end_reason"])
        .size()
        .reset_index(name="episodes")
    )

    plt.figure(figsize=(10, 5))
    ax = sns.barplot(
        data=counts,
        x="agent",
        y="episodes",
        hue="end_reason",
    )
    ax.set_title("Episode End Reasons by Agent")
    ax.set_xlabel("Agent")
    ax.set_ylabel("Episodes")
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()


def create_graphs(csv_path: str, output_dir: str) -> None:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    sns.set_theme(style="whitegrid")
    df = _load_results(csv_path)
    summary_df = _summary(df)
    agent_summary_df = _summary(df, group_col="agent")
    _write_summary(summary_df, out)
    _write_agent_summary(agent_summary_df, out)

    for metric, ylabel, filename, title in SUMMARY_METRICS:
        _save_bar_chart(summary_df, metric, ylabel, title, out / filename)
    for metric, ylabel, filename, title in AGENT_SUMMARY_METRICS:
        _save_bar_chart(
            agent_summary_df,
            metric,
            ylabel,
            title,
            out / filename,
            x_col="agent",
            palette=AGENT_COLORS,
        )

    _save_reward_distribution(df, out / "reward_distribution.png")
    _save_agent_reward_distribution(df, out / "agent_reward_distribution.png")
    _save_progress_by_seed(df, out / "progress_by_seed.png")
    _save_agent_progress_by_seed(df, out / "agent_progress_by_seed.png")
    _save_end_reason_chart(df, out / "end_reasons.png")
    _save_agent_end_reason_chart(df, out / "agent_end_reasons.png")

    print(f"Wrote graphs and summary to: {out}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate graphs from evaluation/evaluate_agents.py CSV results."
    )
    parser.add_argument("csv", help="CSV produced by evaluation/evaluate_agents.py")
    parser.add_argument(
        "--out",
        default="evaluation/graphs",
        help="Output directory for PNG graphs and summary.csv",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    create_graphs(args.csv, args.out)
