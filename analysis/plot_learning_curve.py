from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


"""
train_monitor.csv と eval_monitor.csv を読み込んで、各種指標を時系列でプロットするスクリプト。

使い方:
    uv run python -m analysis.plot_learning_curve  --run-dir outputs/intrinsic_sp_base_dqn_none      
"""


@dataclass
class PlotConfig:
    smoothing_window: int = 20
    dpi: int = 140
    show_raw: bool = True
    show_smoothed: bool = True
    show_eval_npz: bool = True


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot train/eval learning curves from monitor CSV and EvalCallback outputs."
    )
    parser.add_argument(
        "--run-dir",
        type=str,
        required=True,
        help="Run directory created by pglp_rl.training.train",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Figure output directory. Default: <run-dir>/figures",
    )
    parser.add_argument(
        "--smoothing-window",
        type=int,
        default=20,
        help="Rolling average window size.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=140,
        help="Saved figure DPI.",
    )
    parser.add_argument(
        "--no-raw",
        action="store_true",
        help="Do not draw raw per-episode curves.",
    )
    parser.add_argument(
        "--no-smoothed",
        action="store_true",
        help="Do not draw smoothed curves.",
    )
    parser.add_argument(
        "--no-eval-npz",
        action="store_true",
        help="Do not draw EvalCallback evaluations.npz curves.",
    )
    return parser.parse_args()


def resolve_monitor_file(logs_dir: Path, stem: str) -> Optional[Path]:
    candidates = [
        logs_dir / stem,
        logs_dir / f"{stem}.csv",
        logs_dir / f"{stem}.monitor.csv",
    ]
    for path in candidates:
        if path.exists():
            return path

    pattern_candidates = sorted(logs_dir.glob(f"{stem}*monitor.csv"))
    if pattern_candidates:
        return pattern_candidates[0]

    pattern_candidates = sorted(logs_dir.glob(f"{stem}*.csv"))
    if pattern_candidates:
        return pattern_candidates[0]

    return None


def load_monitor_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Monitor file not found: {path}")

    df = pd.read_csv(path, comment="#")
    required = {"r", "l", "t"}
    if not required.issubset(df.columns):
        raise ValueError(f"Monitor CSV must contain {sorted(required)}. Found: {list(df.columns)}")

    out = df.copy()
    out["episode"] = np.arange(1, len(out) + 1)
    out["timestep_end"] = out["l"].cumsum()

    # 追加列が無い場合も動くようにする
    if "episode_total_return" not in out.columns:
        out["episode_total_return"] = out["r"]
    if "episode_external_return" not in out.columns:
        out["episode_external_return"] = np.nan
    if "episode_intrinsic_return" not in out.columns:
        out["episode_intrinsic_return"] = np.nan

    return out


def load_eval_npz(path: Path) -> Optional[dict[str, np.ndarray]]:
    if not path.exists():
        return None
    data = np.load(path, allow_pickle=True)
    return {key: data[key] for key in data.files}


def smooth_series(values: pd.Series, window: int) -> pd.Series:
    window = max(int(window), 1)
    return values.rolling(window=window, min_periods=1).mean()


def save_plot(fig: plt.Figure, path: Path, dpi: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def _plot_single_series(
    *,
    x,
    y,
    title: str,
    xlabel: str,
    ylabel: str,
    label: str,
    output_path: Path,
    cfg: PlotConfig,
) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    if cfg.show_raw:
        ax.plot(x, y, alpha=0.35, label=f"{label} raw")
    if cfg.show_smoothed:
        smoothed = smooth_series(pd.Series(y), cfg.smoothing_window)
        ax.plot(x, smoothed, linewidth=2.0, label=f"{label} smoothed")
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)
    ax.legend()
    save_plot(fig, output_path, cfg.dpi)


def plot_train_total_external_intrinsic(
    train_df: pd.DataFrame,
    figures_dir: Path,
    cfg: PlotConfig,
) -> None:
    _plot_single_series(
        x=train_df["episode"],
        y=train_df["episode_total_return"],
        title="Train total return vs episode",
        xlabel="Episode",
        ylabel="Total return",
        label="train total",
        output_path=figures_dir / "train_total_return_vs_episode.png",
        cfg=cfg,
    )

    _plot_single_series(
        x=train_df["episode"],
        y=train_df["episode_external_return"],
        title="Train external return vs episode",
        xlabel="Episode",
        ylabel="External return",
        label="train external",
        output_path=figures_dir / "train_external_return_vs_episode.png",
        cfg=cfg,
    )

    _plot_single_series(
        x=train_df["episode"],
        y=train_df["episode_intrinsic_return"],
        title="Train intrinsic return vs episode",
        xlabel="Episode",
        ylabel="Intrinsic return",
        label="train intrinsic",
        output_path=figures_dir / "train_intrinsic_return_vs_episode.png",
        cfg=cfg,
    )


def plot_eval_total_external_intrinsic(
    eval_df: pd.DataFrame,
    figures_dir: Path,
    cfg: PlotConfig,
) -> None:
    _plot_single_series(
        x=eval_df["episode"],
        y=eval_df["episode_total_return"],
        title="Eval total return vs episode",
        xlabel="Episode",
        ylabel="Total return",
        label="eval total",
        output_path=figures_dir / "eval_total_return_vs_episode.png",
        cfg=cfg,
    )

    _plot_single_series(
        x=eval_df["episode"],
        y=eval_df["episode_external_return"],
        title="Eval external return vs episode",
        xlabel="Episode",
        ylabel="External return",
        label="eval external",
        output_path=figures_dir / "eval_external_return_vs_episode.png",
        cfg=cfg,
    )

    _plot_single_series(
        x=eval_df["episode"],
        y=eval_df["episode_intrinsic_return"],
        title="Eval intrinsic return vs episode",
        xlabel="Episode",
        ylabel="Intrinsic return",
        label="eval intrinsic",
        output_path=figures_dir / "eval_intrinsic_return_vs_episode.png",
        cfg=cfg,
    )


def plot_episode_length(
    df: pd.DataFrame,
    title: str,
    label: str,
    output_path: Path,
    cfg: PlotConfig,
) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(df["episode"], df["l"], label=label)
    ax.set_title(title)
    ax.set_xlabel("Episode")
    ax.set_ylabel("Episode length")
    ax.grid(True, alpha=0.3)
    ax.legend()
    save_plot(fig, output_path, cfg.dpi)


def plot_combined_train(
    train_df: pd.DataFrame,
    figures_dir: Path,
    cfg: PlotConfig,
) -> None:
    fig, ax = plt.subplots(figsize=(9, 5))

    for col, label in [
        ("episode_total_return", "train total"),
        ("episode_external_return", "train external"),
        ("episode_intrinsic_return", "train intrinsic"),
    ]:
        if cfg.show_raw:
            ax.plot(train_df["timestep_end"], train_df[col], alpha=0.20, label=f"{label} raw")
        if cfg.show_smoothed:
            smoothed = smooth_series(train_df[col], cfg.smoothing_window)
            ax.plot(train_df["timestep_end"], smoothed, linewidth=2.0, label=f"{label} smoothed")

    ax.set_title("Train returns vs timestep")
    ax.set_xlabel("Timestep")
    ax.set_ylabel("Return")
    ax.grid(True, alpha=0.3)
    ax.legend()
    save_plot(fig, figures_dir / "train_returns_combined_vs_timestep.png", cfg.dpi)


def plot_combined_eval(
    eval_df: pd.DataFrame,
    figures_dir: Path,
    cfg: PlotConfig,
) -> None:
    fig, ax = plt.subplots(figsize=(9, 5))

    for col, label in [
        ("episode_total_return", "eval total"),
        ("episode_external_return", "eval external"),
        ("episode_intrinsic_return", "eval intrinsic"),
    ]:
        if cfg.show_raw:
            ax.plot(eval_df["episode"], eval_df[col], alpha=0.35, label=f"{label} raw")
        if cfg.show_smoothed:
            smoothed = smooth_series(eval_df[col], cfg.smoothing_window)
            ax.plot(eval_df["episode"], smoothed, linewidth=2.0, label=f"{label} smoothed")

    ax.set_title("Eval returns vs episode")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Return")
    ax.grid(True, alpha=0.3)
    ax.legend()
    save_plot(fig, figures_dir / "eval_returns_combined_vs_episode.png", cfg.dpi)


def plot_eval_callback_npz(
    eval_data: dict[str, np.ndarray],
    figures_dir: Path,
    cfg: PlotConfig,
) -> None:
    timesteps = np.asarray(eval_data.get("timesteps", []))
    results = np.asarray(eval_data.get("results", []))
    if timesteps.size == 0 or results.size == 0:
        return

    mean_rewards = results.mean(axis=1)
    std_rewards = results.std(axis=1)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(timesteps, mean_rewards, label="EvalCallback mean reward")
    ax.fill_between(
        timesteps,
        mean_rewards - std_rewards,
        mean_rewards + std_rewards,
        alpha=0.2,
        label="±1 std",
    )
    ax.set_title("EvalCallback mean reward vs timestep")
    ax.set_xlabel("Timestep")
    ax.set_ylabel("Mean reward")
    ax.grid(True, alpha=0.3)
    ax.legend()
    save_plot(fig, figures_dir / "eval_callback_mean_reward_vs_timestep.png", cfg.dpi)


def main() -> None:
    args = parse_args()

    run_dir = Path(args.run_dir)
    logs_dir = run_dir / "logs"
    eval_dir = run_dir / "eval"
    figures_dir = Path(args.output_dir) if args.output_dir is not None else run_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    cfg = PlotConfig(
        smoothing_window=args.smoothing_window,
        dpi=args.dpi,
        show_raw=not args.no_raw,
        show_smoothed=not args.no_smoothed,
        show_eval_npz=not args.no_eval_npz,
    )

    train_monitor_path = resolve_monitor_file(logs_dir, "train_monitor")
    eval_monitor_path = resolve_monitor_file(logs_dir, "eval_monitor")

    if train_monitor_path is None:
        raise FileNotFoundError(f"Could not find train monitor CSV in {logs_dir}")
    if eval_monitor_path is None:
        raise FileNotFoundError(f"Could not find eval monitor CSV in {logs_dir}")

    train_df = load_monitor_csv(train_monitor_path)
    eval_df = load_monitor_csv(eval_monitor_path)

    plot_train_total_external_intrinsic(train_df, figures_dir, cfg)
    plot_eval_total_external_intrinsic(eval_df, figures_dir, cfg)

    plot_episode_length(
        train_df,
        title="Train episode length vs episode",
        label="train episode length",
        output_path=figures_dir / "train_episode_length_vs_episode.png",
        cfg=cfg,
    )
    plot_episode_length(
        eval_df,
        title="Eval episode length vs episode",
        label="eval episode length",
        output_path=figures_dir / "eval_episode_length_vs_episode.png",
        cfg=cfg,
    )

    plot_combined_train(train_df, figures_dir, cfg)
    plot_combined_eval(eval_df, figures_dir, cfg)

    if cfg.show_eval_npz:
        eval_npz_path = eval_dir / "evaluations.npz"
        eval_data = load_eval_npz(eval_npz_path)
        if eval_data is not None:
            plot_eval_callback_npz(eval_data, figures_dir, cfg)

    summary_path = figures_dir / "plot_summary.txt"
    with summary_path.open("w", encoding="utf-8") as fh:
        fh.write(f"train_monitor: {train_monitor_path}\n")
        fh.write(f"eval_monitor: {eval_monitor_path}\n")
        fh.write(f"smoothing_window: {cfg.smoothing_window}\n")
        fh.write(f"dpi: {cfg.dpi}\n")


if __name__ == "__main__":
    main()