"""Create presentation-friendly training curves from training_logs CSV files."""

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


DEFAULT_AGENTS = ("PPO", "DQN", "DoubleDQN")
END_REASON_COLORS = {
    "max_steps": "#4e79a7",
    "no_progress": "#e15759",
    "lap_finished": "#59a14f",
    "off_track": "#f28e2b",
    "time_limit": "#76b7b2",
}
REQUIRED_COLUMNS = {
    "episode",
    "agent",
    "reward",
    "avg_reward",
    "progress_percent",
    "avg_progress_percent",
    "steps",
    "end_reason",
}


def _episode_count(path: Path) -> int:
    try:
        return max(sum(1 for _ in path.open()) - 1, 0)
    except OSError:
        return 0


def _logs_for_agent(log_dir: Path, agent: str) -> list[Path]:
    return sorted(log_dir.glob(f"{agent}_seed_*.csv"))


def _best_log_for_agent(log_dir: Path, agent: str) -> Path:
    matches = sorted(
        _logs_for_agent(log_dir, agent),
        key=lambda path: (_episode_count(path), path.stat().st_mtime),
        reverse=True,
    )
    if not matches:
        raise FileNotFoundError(f"No training log found for agent '{agent}' in {log_dir}")
    return matches[0]


def _load_log(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"{path} is missing required columns: {', '.join(sorted(missing))}")

    numeric_columns = [
        "episode",
        "reward",
        "avg_reward",
        "progress_percent",
        "avg_progress_percent",
        "steps",
    ]
    for column in numeric_columns:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    df = df.dropna(subset=["episode", "reward", "progress_percent"])
    df["source_log"] = path.name
    return df


def _stitch_logs(paths: list[Path]) -> pd.DataFrame:
    frames = []
    episode_offset = 0

    for run_index, path in enumerate(paths, start=1):
        df = _load_log(path)
        df = df.sort_values("episode").copy()
        df["run_index"] = run_index
        df["original_episode"] = df["episode"]
        df["episode"] = df["episode"] + episode_offset
        episode_offset = int(df["episode"].max())
        frames.append(df)

    return pd.concat(frames, ignore_index=True)


def _load_logs(
    log_dir: Path,
    agents: list[str],
    logs: list[str] | None,
    stitch_runs: bool,
) -> pd.DataFrame:
    if logs:
        paths = [Path(path) for path in logs]
        if stitch_runs:
            by_agent: dict[str, list[Path]] = {}
            for path in paths:
                agent = pd.read_csv(path, usecols=["agent"], nrows=1)["agent"].iloc[0]
                by_agent.setdefault(agent, []).append(path)
            frames = [_stitch_logs(sorted(agent_paths)) for agent_paths in by_agent.values()]
        else:
            frames = [_load_log(path) for path in paths]
    else:
        frames = []
        for agent in agents:
            if stitch_runs:
                paths = _logs_for_agent(log_dir, agent)
                if not paths:
                    raise FileNotFoundError(f"No training log found for agent '{agent}' in {log_dir}")
                frames.append(_stitch_logs(paths))
            else:
                frames.append(_load_log(_best_log_for_agent(log_dir, agent)))

    return pd.concat(frames, ignore_index=True)


def _add_smoothed_columns(df: pd.DataFrame, window: int) -> pd.DataFrame:
    df = df.sort_values(["agent", "episode"]).copy()
    grouped = df.groupby("agent", group_keys=False)
    df["reward_smooth"] = grouped["reward"].transform(
        lambda series: series.rolling(window, min_periods=1).mean()
    )
    df["progress_smooth"] = grouped["progress_percent"].transform(
        lambda series: series.rolling(window, min_periods=1).mean()
    )
    return df


def _save_line_plot(
    df: pd.DataFrame,
    y: str,
    ylabel: str,
    title: str,
    output_path: Path,
) -> None:
    plt.figure(figsize=(11, 6))
    ax = sns.lineplot(
        data=df,
        x="episode",
        y=y,
        hue="agent",
        linewidth=2.4,
    )
    ax.set_title(title)
    ax.set_xlabel("Episode")
    ax.set_ylabel(ylabel)
    ax.legend(title="Agent")
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()


def _axis_limits(df: pd.DataFrame, x: str, y: str) -> tuple[tuple[float, float], tuple[float, float]]:
    x_min = float(df[x].min())
    x_max = float(df[x].max())
    y_min = float(df[y].min())
    y_max = float(df[y].max())

    if y_min == y_max:
        y_padding = 1.0
    else:
        y_padding = (y_max - y_min) * 0.08

    return (x_min, x_max), (y_min - y_padding, y_max + y_padding)


def _safe_filename(value: str) -> str:
    return "".join(char if char.isalnum() else "_" for char in value).strip("_")


def _save_individual_line_plots(
    df: pd.DataFrame,
    y: str,
    ylabel: str,
    title_prefix: str,
    output_dir: Path,
    filename_prefix: str,
) -> None:
    x_limits, y_limits = _axis_limits(df, "episode", y)

    for agent, agent_df in df.groupby("agent"):
        plt.figure(figsize=(11, 6))
        ax = sns.lineplot(
            data=agent_df,
            x="episode",
            y=y,
            linewidth=1.6,
            color="#3f7f93",
        )
        ax.set_title(f"{title_prefix}: {agent}")
        ax.set_xlabel("Episode")
        ax.set_ylabel(ylabel)
        ax.set_xlim(*x_limits)
        ax.set_ylim(*y_limits)
        plt.tight_layout()
        plt.savefig(output_dir / f"{filename_prefix}_{_safe_filename(agent)}.png", dpi=180)
        plt.close()


def _save_end_reason_plot(df: pd.DataFrame, output_path: Path) -> None:
    counts = (
        df.groupby(["agent", "end_reason"])
        .size()
        .reset_index(name="episodes")
    )

    plt.figure(figsize=(10, 5.5))
    ax = sns.barplot(
        data=counts,
        x="agent",
        y="episodes",
        hue="end_reason",
        palette=END_REASON_COLORS,
    )
    ax.set_title("Training Episode End Reasons")
    ax.set_xlabel("Agent")
    ax.set_ylabel("Episodes")
    ax.legend(title="End reason")
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()


def _save_individual_end_reason_plots(df: pd.DataFrame, output_dir: Path) -> None:
    counts = (
        df.groupby(["agent", "end_reason"])
        .size()
        .reset_index(name="episodes")
    )
    y_max = counts["episodes"].max()

    for agent, agent_counts in counts.groupby("agent"):
        plt.figure(figsize=(10, 5.5))
        ax = sns.barplot(
            data=agent_counts,
            x="end_reason",
            y="episodes",
            hue="end_reason",
            palette=END_REASON_COLORS,
            legend=False,
            width=0.55,
        )
        ax.set_title(f"Training Episode End Reasons: {agent}")
        ax.set_xlabel("End reason")
        ax.set_ylabel("Episodes")
        ax.set_ylim(0, y_max * 1.08)
        ax.tick_params(axis="x", rotation=20)
        plt.tight_layout()
        plt.savefig(output_dir / f"training_end_reasons_{_safe_filename(agent)}.png", dpi=180)
        plt.close()


def _write_summary(df: pd.DataFrame, output_path: Path, window: int) -> None:
    summary = (
        df.groupby("agent", as_index=False)
        .agg(
            source_log=("source_log", "last"),
            episodes=("episode", "max"),
            final_reward_smooth=("reward_smooth", "last"),
            best_reward_smooth=("reward_smooth", "max"),
            final_progress_smooth=("progress_smooth", "last"),
            best_progress_smooth=("progress_smooth", "max"),
            avg_steps=("steps", "mean"),
            runs=("source_log", "nunique"),
        )
        .sort_values("agent")
    )
    summary.insert(1, "smoothing_window", window)
    summary.to_csv(output_path, index=False)


def create_training_graphs(
    log_dir: str,
    output_dir: str,
    agents: list[str],
    logs: list[str] | None,
    window: int,
    stitch_runs: bool,
) -> None:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    sns.set_theme(style="whitegrid")
    df = _load_logs(Path(log_dir), agents, logs, stitch_runs)
    df = _add_smoothed_columns(df, window)

    _save_line_plot(
        df,
        "reward_smooth",
        f"Reward ({window}-episode rolling average)",
        "Training Reward Over Time",
        out / "training_reward.png",
    )
    _save_line_plot(
        df,
        "progress_smooth",
        f"Track progress % ({window}-episode rolling average)",
        "Training Track Progress Over Time",
        out / "training_progress.png",
    )
    _save_individual_line_plots(
        df,
        "reward_smooth",
        f"Reward ({window}-episode rolling average)",
        "Training Reward Over Time",
        out,
        "training_reward",
    )
    _save_individual_line_plots(
        df,
        "progress_smooth",
        f"Track progress % ({window}-episode rolling average)",
        "Training Track Progress Over Time",
        out,
        "training_progress",
    )
    _save_end_reason_plot(df, out / "training_end_reasons.png")
    _save_individual_end_reason_plots(df, out)
    _write_summary(df, out / "training_summary.csv", window)

    print(f"Wrote training graphs and summary to: {out}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate training graphs from training_logs CSV files."
    )
    parser.add_argument(
        "--log-dir",
        default="training_logs",
        help="Directory containing training log CSV files.",
    )
    parser.add_argument(
        "--out",
        default="training_logs/graphs",
        help="Output directory for PNG graphs and summary CSV.",
    )
    parser.add_argument(
        "--agents",
        nargs="+",
        default=list(DEFAULT_AGENTS),
        help="Agent names to plot when --logs is not provided.",
    )
    parser.add_argument(
        "--logs",
        nargs="+",
        help="Specific training log CSV files to plot. Overrides --agents.",
    )
    parser.add_argument(
        "--window",
        type=int,
        default=100,
        help="Rolling average window size in episodes.",
    )
    parser.add_argument(
        "--single-run",
        action="store_true",
        help="Use only the longest log per agent instead of stitching multiple runs.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    create_training_graphs(
        log_dir=args.log_dir,
        output_dir=args.out,
        agents=args.agents,
        logs=args.logs,
        window=args.window,
        stitch_runs=not args.single_run,
    )
