from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


"""
This script aggregates multiple simulation runs (with train/eval monitor CSVs) under a common group directory and plots mean learning curves with optional raw runs and std bands.


Example usage:
python analysis/plot_learning_curve_multi.py \
  --group-dir outputs/multiseed_sp_base_dqn_none_10sim_20260423-012700
"""


@dataclass
class PlotConfig:
    smoothing_window: int = 20
    dpi: int = 140
    show_raw_runs: bool = False
    show_std_band: bool = True
    show_eval_npz: bool = True


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Aggregate multiple run directories and plot mean learning curves."
    )
    parser.add_argument(
        "--group-dir",
        type=str,
        required=True,
        help="Directory containing multiple simulation run directories. The script searches recursively.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory. Default: <group-dir>/figures_multi",
    )
    parser.add_argument(
        "--smoothing-window",
        type=int,
        default=20,
        help="Rolling mean window for aggregated curves.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=140,
        help="Saved figure DPI.",
    )
    parser.add_argument(
        "--show-raw-runs",
        action="store_true",
        help="Overlay each run as a faint line.",
    )
    parser.add_argument(
        "--no-std-band",
        action="store_true",
        help="Disable mean±std shaded band.",
    )
    parser.add_argument(
        "--no-eval-npz",
        action="store_true",
        help="Do not aggregate eval/evaluations.npz.",
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
    df = pd.read_csv(path, comment="#")
    required = {"r", "l", "t"}
    if not required.issubset(df.columns):
        raise ValueError(f"Monitor CSV must contain {sorted(required)}. Found: {list(df.columns)}")

    out = df.copy()
    out["episode"] = np.arange(1, len(out) + 1)
    out["timestep_end"] = out["l"].cumsum()

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


def find_run_dirs(group_dir: Path) -> List[Path]:
    run_dirs: List[Path] = []
    for p in sorted(group_dir.rglob("*")):
        if not p.is_dir():
            continue
        logs_dir = p / "logs"
        if not logs_dir.is_dir():
            continue
        train_monitor = resolve_monitor_file(logs_dir, "train_monitor")
        eval_monitor = resolve_monitor_file(logs_dir, "eval_monitor")
        if train_monitor is not None and eval_monitor is not None:
            run_dirs.append(p)
    unique = []
    seen = set()
    for p in run_dirs:
        rp = p.resolve()
        if rp not in seen:
            unique.append(p)
            seen.add(rp)
    return unique


def smooth_series(values: pd.Series, window: int) -> pd.Series:
    window = max(int(window), 1)
    return values.rolling(window=window, min_periods=1).mean()


def aggregate_on_x(dfs: List[pd.DataFrame], x_col: str, y_col: str) -> pd.DataFrame:
    series_list = []
    for idx, df in enumerate(dfs):
        s = df[[x_col, y_col]].copy()
        s = s.rename(columns={y_col: f"run_{idx}"})
        s = s.groupby(x_col, as_index=False).mean(numeric_only=True)
        series_list.append(s)

    merged = series_list[0]
    for s in series_list[1:]:
        merged = merged.merge(s, on=x_col, how="outer")

    merged = merged.sort_values(x_col).reset_index(drop=True)
    run_cols = [c for c in merged.columns if c != x_col]
    merged["mean"] = merged[run_cols].mean(axis=1, skipna=True)
    merged["std"] = merged[run_cols].std(axis=1, ddof=0, skipna=True)
    merged["count"] = merged[run_cols].notna().sum(axis=1)
    return merged


def save_plot(fig: plt.Figure, path: Path, dpi: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def plot_aggregated(
    *,
    run_dfs: List[pd.DataFrame],
    x_col: str,
    y_col: str,
    title: str,
    xlabel: str,
    ylabel: str,
    output_path: Path,
    cfg: PlotConfig,
) -> pd.DataFrame:
    agg = aggregate_on_x(run_dfs, x_col=x_col, y_col=y_col)
    agg["mean_smoothed"] = smooth_series(agg["mean"], cfg.smoothing_window)
    agg["std_smoothed"] = smooth_series(agg["std"], cfg.smoothing_window)

    fig, ax = plt.subplots(figsize=(8, 5))

    if cfg.show_raw_runs:
        for df in run_dfs:
            ax.plot(df[x_col], df[y_col], alpha=0.12, linewidth=1.0)

    ax.plot(agg[x_col], agg["mean_smoothed"], linewidth=2.2, label="mean")
    if cfg.show_std_band:
        lower = agg["mean_smoothed"] - agg["std_smoothed"]
        upper = agg["mean_smoothed"] + agg["std_smoothed"]
        ax.fill_between(agg[x_col], lower, upper, alpha=0.2, label="±1 std")

    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)
    ax.legend()
    save_plot(fig, output_path, cfg.dpi)
    return agg


def plot_eval_npz_aggregated(
    eval_datas: List[dict[str, np.ndarray]],
    figures_dir: Path,
    cfg: PlotConfig,
) -> Optional[pd.DataFrame]:
    rows = []
    for ridx, data in enumerate(eval_datas):
        timesteps = np.asarray(data.get("timesteps", []))
        results = np.asarray(data.get("results", []))
        if timesteps.size == 0 or results.size == 0:
            continue
        mean_rewards = results.mean(axis=1)
        for t, m in zip(timesteps, mean_rewards):
            rows.append({"run": ridx, "timestep": int(t), "mean_reward": float(m)})

    if not rows:
        return None

    df = pd.DataFrame(rows)
    pivot = df.pivot_table(index="timestep", columns="run", values="mean_reward", aggfunc="mean")
    pivot = pivot.sort_index().reset_index()
    run_cols = [c for c in pivot.columns if c != "timestep"]
    pivot["mean"] = pivot[run_cols].mean(axis=1, skipna=True)
    pivot["std"] = pivot[run_cols].std(axis=1, ddof=0, skipna=True)
    pivot["mean_smoothed"] = smooth_series(pivot["mean"], cfg.smoothing_window)
    pivot["std_smoothed"] = smooth_series(pivot["std"], cfg.smoothing_window)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(pivot["timestep"], pivot["mean_smoothed"], linewidth=2.2, label="EvalCallback mean reward")
    if cfg.show_std_band:
        ax.fill_between(
            pivot["timestep"],
            pivot["mean_smoothed"] - pivot["std_smoothed"],
            pivot["mean_smoothed"] + pivot["std_smoothed"],
            alpha=0.2,
            label="±1 std",
        )
    ax.set_title("Aggregated EvalCallback mean reward vs timestep")
    ax.set_xlabel("Timestep")
    ax.set_ylabel("Mean reward")
    ax.grid(True, alpha=0.3)
    ax.legend()
    save_plot(fig, figures_dir / "eval_callback_mean_reward_vs_timestep_multi.png", cfg.dpi)

    return pivot


def main() -> None:
    args = parse_args()
    group_dir = Path(args.group_dir)
    if not group_dir.exists():
        raise FileNotFoundError(f"group directory not found: {group_dir}")

    output_dir = Path(args.output_dir) if args.output_dir else group_dir / "figures_multi"
    output_dir.mkdir(parents=True, exist_ok=True)

    cfg = PlotConfig(
        smoothing_window=args.smoothing_window,
        dpi=args.dpi,
        show_raw_runs=args.show_raw_runs,
        show_std_band=not args.no_std_band,
        show_eval_npz=not args.no_eval_npz,
    )

    run_dirs = find_run_dirs(group_dir)
    if not run_dirs:
        raise FileNotFoundError(f"No run directories with train/eval monitor CSVs found under: {group_dir}")

    train_dfs: List[pd.DataFrame] = []
    eval_dfs: List[pd.DataFrame] = []
    eval_npz_datas: List[dict[str, np.ndarray]] = []

    for run_dir in run_dirs:
        logs_dir = run_dir / "logs"
        eval_dir = run_dir / "eval"
        train_monitor_path = resolve_monitor_file(logs_dir, "train_monitor")
        eval_monitor_path = resolve_monitor_file(logs_dir, "eval_monitor")
        if train_monitor_path is None or eval_monitor_path is None:
            continue
        train_dfs.append(load_monitor_csv(train_monitor_path))
        eval_dfs.append(load_monitor_csv(eval_monitor_path))
        if cfg.show_eval_npz:
            d = load_eval_npz(eval_dir / "evaluations.npz")
            if d is not None:
                eval_npz_datas.append(d)

    if not train_dfs or not eval_dfs:
        raise RuntimeError("Could not load aggregated monitor CSVs.")

    summary: Dict[str, pd.DataFrame] = {}

    summary["train_total"] = plot_aggregated(
        run_dfs=train_dfs,
        x_col="episode",
        y_col="episode_total_return",
        title="Aggregated train total return vs episode",
        xlabel="Episode",
        ylabel="Total return",
        output_path=output_dir / "train_total_return_vs_episode_multi.png",
        cfg=cfg,
    )
    summary["train_external"] = plot_aggregated(
        run_dfs=train_dfs,
        x_col="episode",
        y_col="episode_external_return",
        title="Aggregated train external return vs episode",
        xlabel="Episode",
        ylabel="External return",
        output_path=output_dir / "train_external_return_vs_episode_multi.png",
        cfg=cfg,
    )
    summary["train_intrinsic"] = plot_aggregated(
        run_dfs=train_dfs,
        x_col="episode",
        y_col="episode_intrinsic_return",
        title="Aggregated train intrinsic return vs episode",
        xlabel="Episode",
        ylabel="Intrinsic return",
        output_path=output_dir / "train_intrinsic_return_vs_episode_multi.png",
        cfg=cfg,
    )
    summary["train_length"] = plot_aggregated(
        run_dfs=train_dfs,
        x_col="episode",
        y_col="l",
        title="Aggregated train episode length vs episode",
        xlabel="Episode",
        ylabel="Episode length",
        output_path=output_dir / "train_episode_length_vs_episode_multi.png",
        cfg=cfg,
    )
    summary["train_total_timestep"] = plot_aggregated(
        run_dfs=train_dfs,
        x_col="timestep_end",
        y_col="episode_total_return",
        title="Aggregated train total return vs timestep",
        xlabel="Timestep",
        ylabel="Total return",
        output_path=output_dir / "train_total_return_vs_timestep_multi.png",
        cfg=cfg,
    )
    summary["train_external_timestep"] = plot_aggregated(
        run_dfs=train_dfs,
        x_col="timestep_end",
        y_col="episode_external_return",
        title="Aggregated train external return vs timestep",
        xlabel="Timestep",
        ylabel="External return",
        output_path=output_dir / "train_external_return_vs_timestep_multi.png",
        cfg=cfg,
    )
    summary["train_intrinsic_timestep"] = plot_aggregated(
        run_dfs=train_dfs,
        x_col="timestep_end",
        y_col="episode_intrinsic_return",
        title="Aggregated train intrinsic return vs timestep",
        xlabel="Timestep",
        ylabel="Intrinsic return",
        output_path=output_dir / "train_intrinsic_return_vs_timestep_multi.png",
        cfg=cfg,
    )

    summary["eval_total"] = plot_aggregated(
        run_dfs=eval_dfs,
        x_col="episode",
        y_col="episode_total_return",
        title="Aggregated eval total return vs episode",
        xlabel="Episode",
        ylabel="Total return",
        output_path=output_dir / "eval_total_return_vs_episode_multi.png",
        cfg=cfg,
    )
    summary["eval_external"] = plot_aggregated(
        run_dfs=eval_dfs,
        x_col="episode",
        y_col="episode_external_return",
        title="Aggregated eval external return vs episode",
        xlabel="Episode",
        ylabel="External return",
        output_path=output_dir / "eval_external_return_vs_episode_multi.png",
        cfg=cfg,
    )
    summary["eval_intrinsic"] = plot_aggregated(
        run_dfs=eval_dfs,
        x_col="episode",
        y_col="episode_intrinsic_return",
        title="Aggregated eval intrinsic return vs episode",
        xlabel="Episode",
        ylabel="Intrinsic return",
        output_path=output_dir / "eval_intrinsic_return_vs_episode_multi.png",
        cfg=cfg,
    )
    summary["eval_length"] = plot_aggregated(
        run_dfs=eval_dfs,
        x_col="episode",
        y_col="l",
        title="Aggregated eval episode length vs episode",
        xlabel="Episode",
        ylabel="Episode length",
        output_path=output_dir / "eval_episode_length_vs_episode_multi.png",
        cfg=cfg,
    )

    if cfg.show_eval_npz and eval_npz_datas:
        agg_npz = plot_eval_npz_aggregated(eval_npz_datas, output_dir, cfg)
        if agg_npz is not None:
            agg_npz.to_csv(output_dir / "eval_callback_mean_reward_multi.csv", index=False)

    for name, df in summary.items():
        df.to_csv(output_dir / f"{name}_multi.csv", index=False)

    with (output_dir / "plot_summary_multi.txt").open("w", encoding="utf-8") as fh:
        fh.write(f"group_dir: {group_dir}\n")
        fh.write(f"num_runs: {len(run_dirs)}\n")
        fh.write(f"smoothing_window: {cfg.smoothing_window}\n")
        fh.write(f"dpi: {cfg.dpi}\n")
        fh.write(f"show_raw_runs: {cfg.show_raw_runs}\n")
        fh.write(f"show_std_band: {cfg.show_std_band}\n")
        fh.write("run_dirs:\n")
        for rd in run_dirs:
            fh.write(f"- {rd}\n")

    print(f"Loaded {len(run_dirs)} run directories from: {group_dir}")
    print(f"Saved aggregated plots to: {output_dir}")


if __name__ == "__main__":
    main()
