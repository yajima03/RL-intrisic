from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

"""
Create eight diagnostic plots for LPM CSV logs.

Example:
python analysis/plot_lpm_diagnostics.py \
  --group-dir outputs/multiseed_sp_base_dqn_lpm_10sim_20260423-012700
"""


@dataclass
class RunLogs:
    run_id: str
    run_dir: Path
    probe: Optional[pd.DataFrame]
    transition: Optional[pd.DataFrame]
    update: Optional[pd.DataFrame]


@dataclass
class PlotConfig:
    dpi: int
    smoothing_window: int
    bin_size: int
    scatter_sample: int
    show_raw_runs: bool
    show_std_band: bool
    clamp_max: Optional[float]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot eight LPM diagnostics from lpm_probe/transition/update CSV logs."
    )
    parser.add_argument(
        "--group-dir",
        type=str,
        required=True,
        help="Directory containing one run or multiple simulation run directories.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directory to save PNG files. Default: <group-dir>/plots_lpm_diagnostics",
    )
    parser.add_argument("--dpi", type=int, default=150, help="Saved figure DPI.")
    parser.add_argument(
        "--smoothing-window",
        type=int,
        default=3,
        help="Rolling mean window used on aggregated time-series plots.",
    )
    parser.add_argument(
        "--bin-size",
        type=int,
        default=1000,
        help="Env-step bin size for transition plots. Use 0 to keep exact env_step.",
    )
    parser.add_argument(
        "--scatter-sample",
        type=int,
        default=50000,
        help="Maximum total points for scatter plots before deterministic downsampling.",
    )
    parser.add_argument(
        "--show-raw-runs",
        action="store_true",
        help="Overlay individual runs where a plot supports it.",
    )
    parser.add_argument(
        "--no-std-band",
        action="store_true",
        help="Disable mean±std shaded bands.",
    )
    parser.add_argument(
        "--clamp-max",
        type=float,
        default=None,
        help=(
            "Reward clamp upper bound. If omitted, plot 7 estimates saturation "
            "from max lpm_raw_used."
        ),
    )
    return parser.parse_args()


def read_csv_numeric(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    for col in df.columns:
        converted = pd.to_numeric(df[col], errors="coerce")
        if converted.notna().sum() > 0:
            df[col] = converted
    if "env_step" in df.columns:
        df = df.sort_values("env_step").reset_index(drop=True)
    return df


def find_run_logs(group_dir: Path) -> list[RunLogs]:
    candidates: set[Path] = set()
    for csv_name in ("lpm_probe.csv", "lpm_transition.csv", "lpm_update.csv"):
        for path in group_dir.rglob(csv_name):
            if path.is_file():
                candidates.add(path.parent)

    if not candidates:
        direct = {
            name: group_dir / name
            for name in ("lpm_probe.csv", "lpm_transition.csv", "lpm_update.csv")
        }
        if any(path.exists() for path in direct.values()):
            candidates.add(group_dir)

    logs: list[RunLogs] = []
    for idx, logs_dir in enumerate(sorted(candidates)):
        run_dir = logs_dir.parent if logs_dir.name == "logs" else logs_dir
        rel = run_dir.relative_to(group_dir) if run_dir != group_dir else Path(f"run_{idx}")
        run_id = str(rel).replace("/", "_")
        if not run_id or run_id == ".":
            run_id = f"run_{idx}"

        def load(name: str, base_dir: Path = logs_dir) -> Optional[pd.DataFrame]:
            path = base_dir / name
            return read_csv_numeric(path) if path.exists() else None

        logs.append(
            RunLogs(
                run_id=run_id,
                run_dir=run_dir,
                probe=load("lpm_probe.csv"),
                transition=load("lpm_transition.csv"),
                update=load("lpm_update.csv"),
            )
        )
    return logs


def require_columns(df: pd.DataFrame, columns: Iterable[str]) -> bool:
    return all(col in df.columns for col in columns)


def save_plot(fig: plt.Figure, output_path: Path, dpi: int) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def smooth(values: pd.Series, window: int) -> pd.Series:
    window = max(int(window), 1)
    return values.rolling(window=window, min_periods=1).mean()


def bin_env_step(df: pd.DataFrame, bin_size: int) -> pd.Series:
    if bin_size <= 0:
        return df["env_step"]
    return (np.floor(df["env_step"] / bin_size) * bin_size + bin_size / 2).astype(float)


def aggregate_run_series(
    tables: list[pd.DataFrame],
    *,
    x_col: str,
    y_col: str,
) -> pd.DataFrame:
    merged: Optional[pd.DataFrame] = None
    for idx, df in enumerate(tables):
        if df.empty or not require_columns(df, [x_col, y_col]):
            continue
        s = df[[x_col, y_col]].dropna().groupby(x_col, as_index=False).mean()
        s = s.rename(columns={y_col: f"run_{idx}"})
        merged = s if merged is None else merged.merge(s, on=x_col, how="outer")

    if merged is None:
        return pd.DataFrame(columns=[x_col, "mean", "std", "count"])

    merged = merged.sort_values(x_col).reset_index(drop=True)
    run_cols = [col for col in merged.columns if col != x_col]
    merged["mean"] = merged[run_cols].mean(axis=1, skipna=True)
    merged["std"] = merged[run_cols].std(axis=1, ddof=0, skipna=True)
    merged["count"] = merged[run_cols].notna().sum(axis=1)
    return merged


def line_mean_std(
    ax: plt.Axes,
    agg: pd.DataFrame,
    *,
    x_col: str,
    label: str,
    cfg: PlotConfig,
    color: Optional[str] = None,
) -> None:
    y = smooth(agg["mean"], cfg.smoothing_window)
    std = smooth(agg["std"], cfg.smoothing_window)
    line = ax.plot(agg[x_col], y, linewidth=2.2, label=label, color=color)
    if cfg.show_std_band:
        band_color = color or line[0].get_color()
        ax.fill_between(agg[x_col], y - std, y + std, alpha=0.18, color=band_color)


def concat_probe(logs: list[RunLogs]) -> pd.DataFrame:
    dfs = []
    for run in logs:
        if run.probe is None:
            continue
        df = run.probe.copy()
        df["run_id"] = run.run_id
        dfs.append(df)
    return pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()


def concat_transition(logs: list[RunLogs]) -> pd.DataFrame:
    dfs = []
    for run in logs:
        if run.transition is None:
            continue
        df = run.transition.copy()
        df["run_id"] = run.run_id
        dfs.append(df)
    return pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()


def deterministic_sample(df: pd.DataFrame, max_rows: int) -> pd.DataFrame:
    if max_rows <= 0 or len(df) <= max_rows:
        return df
    return df.sample(n=max_rows, random_state=0)


def add_identity_line(ax: plt.Axes) -> None:
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    lo = min(xlim[0], ylim[0])
    hi = max(xlim[1], ylim[1])
    ax.plot([lo, hi], [lo, hi], color="black", linestyle="--", linewidth=1.0, alpha=0.7)
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)


def plot_scatter(
    df: pd.DataFrame,
    *,
    x: str,
    y: str,
    title: str,
    output_path: Path,
    cfg: PlotConfig,
    zero_lines: bool = False,
) -> bool:
    if df.empty or not require_columns(df, [x, y]):
        return False

    data = deterministic_sample(df[[x, y, "env_step", "run_id"]].dropna(), cfg.scatter_sample)
    if data.empty:
        return False

    fig, ax = plt.subplots(figsize=(6.5, 6))
    if data["run_id"].nunique() <= 12:
        for run_id, part in data.groupby("run_id"):
            ax.scatter(part[x], part[y], s=10, alpha=0.35, label=run_id)
        ax.legend(markerscale=2, fontsize=8, frameon=False)
    else:
        sc = ax.scatter(data[x], data[y], c=data["env_step"], s=8, alpha=0.35, cmap="viridis")
        fig.colorbar(sc, ax=ax, label="env_step")
    add_identity_line(ax)
    if zero_lines:
        ax.axhline(0, color="gray", linewidth=1.0, alpha=0.7)
        ax.axvline(0, color="gray", linewidth=1.0, alpha=0.7)
    ax.set_title(title)
    ax.set_xlabel(x)
    ax.set_ylabel(y)
    ax.grid(True, alpha=0.25)
    save_plot(fig, output_path, cfg.dpi)
    return True


def probe_metric_tables(logs: list[RunLogs]) -> list[pd.DataFrame]:
    tables = []
    required = ["env_step", "previous_error_log", "predicted_previous_error"]
    lp_required = ["oracle_pointwise_progress", "estimated_lpm_signed"]
    for run in logs:
        if run.probe is None or not require_columns(run.probe, required + lp_required):
            continue
        df = run.probe.copy()
        df["error_mae"] = (df["predicted_previous_error"] - df["previous_error_log"]).abs()
        df["error_bias"] = df["predicted_previous_error"] - df["previous_error_log"]
        oracle_sign = np.sign(df["oracle_pointwise_progress"])
        estimated_sign = np.sign(df["estimated_lpm_signed"])
        df["lp_sign_agreement"] = (oracle_sign == estimated_sign).astype(float)
        tables.append(
            df.groupby("env_step", as_index=False)[
                ["error_mae", "error_bias", "lp_sign_agreement"]
            ].mean()
        )
    return tables


def plot_probe_accuracy(logs: list[RunLogs], output_path: Path, cfg: PlotConfig) -> bool:
    tables = probe_metric_tables(logs)
    if not tables:
        return False

    fig, axes = plt.subplots(2, 1, figsize=(8, 7), sharex=True)
    for metric, ax in [
        ("error_mae", axes[0]),
        ("error_bias", axes[0]),
        ("lp_sign_agreement", axes[1]),
    ]:
        agg = aggregate_run_series(tables, x_col="env_step", y_col=metric)
        if agg.empty:
            continue
        line_mean_std(ax, agg, x_col="env_step", label=metric, cfg=cfg)
        if cfg.show_raw_runs:
            for table in tables:
                ax.plot(table["env_step"], table[metric], alpha=0.12, linewidth=1.0)

    axes[0].axhline(0, color="gray", linewidth=1.0, alpha=0.7)
    axes[0].set_title("Probe error-model accuracy over time")
    axes[0].set_ylabel("log-error residual")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(frameon=False)
    axes[1].set_ylim(-0.02, 1.02)
    axes[1].set_xlabel("env_step")
    axes[1].set_ylabel("sign agreement")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(frameon=False)
    save_plot(fig, output_path, cfg.dpi)
    return True


def probe_lp_tables(logs: list[RunLogs], reducer: str) -> list[pd.DataFrame]:
    tables = []
    required = ["env_step", "oracle_pointwise_progress", "estimated_lpm_signed"]
    agg_func: str | Callable[[pd.Series], float] = "median" if reducer == "median" else "mean"
    for run in logs:
        if run.probe is None or not require_columns(run.probe, required):
            continue
        grouped = run.probe.groupby("env_step", as_index=False).agg(
            oracle_pointwise_progress=("oracle_pointwise_progress", agg_func),
            oracle_q25=("oracle_pointwise_progress", lambda s: float(s.quantile(0.25))),
            oracle_q75=("oracle_pointwise_progress", lambda s: float(s.quantile(0.75))),
            estimated_lpm_signed=("estimated_lpm_signed", agg_func),
            estimated_q25=("estimated_lpm_signed", lambda s: float(s.quantile(0.25))),
            estimated_q75=("estimated_lpm_signed", lambda s: float(s.quantile(0.75))),
        )
        tables.append(grouped)
    return tables


def plot_lp_time(logs: list[RunLogs], output_path: Path, cfg: PlotConfig) -> bool:
    tables = probe_lp_tables(logs, reducer="median")
    if not tables:
        return False

    fig, ax = plt.subplots(figsize=(8, 5))
    for metric, label in [
        ("oracle_pointwise_progress", "oracle LP"),
        ("estimated_lpm_signed", "estimated LP"),
    ]:
        agg = aggregate_run_series(tables, x_col="env_step", y_col=metric)
        if not agg.empty:
            line_mean_std(ax, agg, x_col="env_step", label=label, cfg=cfg)
    if cfg.show_raw_runs:
        for table in tables:
            ax.plot(
                table["env_step"],
                table["oracle_pointwise_progress"],
                alpha=0.10,
                linewidth=1.0,
            )
            ax.plot(table["env_step"], table["estimated_lpm_signed"], alpha=0.10, linewidth=1.0)
    ax.axhline(0, color="gray", linewidth=1.0, alpha=0.7)
    ax.set_title("Oracle LP and estimated LP over time")
    ax.set_xlabel("env_step")
    ax.set_ylabel("median LP over probe set")
    ax.grid(True, alpha=0.3)
    ax.legend(frameon=False)
    save_plot(fig, output_path, cfg.dpi)
    return True


def plot_depth_heatmap(df: pd.DataFrame, output_path: Path, cfg: PlotConfig) -> bool:
    required = ["env_step", "depth", "oracle_pointwise_progress", "estimated_lpm_signed"]
    if df.empty or not require_columns(df, required):
        return False
    data = df[required].dropna().copy()
    if data.empty:
        return False
    data["lp_mae"] = (data["estimated_lpm_signed"] - data["oracle_pointwise_progress"]).abs()
    data["env_step_bin"] = bin_env_step(data, cfg.bin_size)
    pivot = data.pivot_table(
        index="depth",
        columns="env_step_bin",
        values="lp_mae",
        aggfunc="mean",
    ).sort_index()
    if pivot.empty:
        return False

    fig, ax = plt.subplots(figsize=(9, 5.5))
    im = ax.imshow(pivot.to_numpy(), aspect="auto", origin="lower", interpolation="nearest")
    fig.colorbar(im, ax=ax, label="LP estimation MAE")
    x_values = pivot.columns.to_numpy()
    y_values = pivot.index.to_numpy()
    x_tick_idx = np.linspace(0, len(x_values) - 1, min(8, len(x_values)), dtype=int)
    y_tick_idx = np.arange(len(y_values))
    ax.set_xticks(x_tick_idx)
    ax.set_xticklabels([f"{x_values[i]:.0f}" for i in x_tick_idx], rotation=30, ha="right")
    ax.set_yticks(y_tick_idx)
    ax.set_yticklabels([f"{y_values[i]:.0f}" for i in y_tick_idx])
    ax.set_title("Depth x env_step LP estimation MAE")
    ax.set_xlabel("env_step")
    ax.set_ylabel("depth")
    save_plot(fig, output_path, cfg.dpi)
    return True


def transition_metric_tables(logs: list[RunLogs], cfg: PlotConfig) -> list[pd.DataFrame]:
    tables = []
    required = ["env_step", "lpm_signed"]
    for run in logs:
        if run.transition is None or not require_columns(run.transition, required):
            continue
        df = run.transition.copy()
        df["env_step_bin"] = bin_env_step(df, cfg.bin_size)
        df["positive_rate"] = (df["lpm_signed"] > 0).astype(float)
        df["negative_rate"] = (df["lpm_signed"] < 0).astype(float)
        df["zero_rate"] = (df["lpm_signed"] == 0).astype(float)
        df["positive_abs_mean"] = np.where(df["lpm_signed"] > 0, df["lpm_signed"].abs(), np.nan)
        df["negative_abs_mean"] = np.where(df["lpm_signed"] < 0, df["lpm_signed"].abs(), np.nan)
        tables.append(
            df.groupby("env_step_bin", as_index=False)[
                [
                    "positive_rate",
                    "negative_rate",
                    "zero_rate",
                    "positive_abs_mean",
                    "negative_abs_mean",
                ]
            ].mean()
        )
    return tables


def plot_transition_signs(logs: list[RunLogs], output_path: Path, cfg: PlotConfig) -> bool:
    tables = transition_metric_tables(logs, cfg)
    if not tables:
        return False

    fig, axes = plt.subplots(2, 1, figsize=(8, 7), sharex=True)
    for metric in ["positive_rate", "negative_rate", "zero_rate"]:
        agg = aggregate_run_series(tables, x_col="env_step_bin", y_col=metric)
        if not agg.empty:
            line_mean_std(axes[0], agg, x_col="env_step_bin", label=metric, cfg=cfg)
    for metric in ["positive_abs_mean", "negative_abs_mean"]:
        agg = aggregate_run_series(tables, x_col="env_step_bin", y_col=metric)
        if not agg.empty:
            line_mean_std(axes[1], agg, x_col="env_step_bin", label=metric, cfg=cfg)

    axes[0].set_ylim(-0.02, 1.02)
    axes[0].set_title("Transition LP sign rates")
    axes[0].set_ylabel("rate")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(frameon=False)
    axes[1].set_xlabel("env_step")
    axes[1].set_ylabel("mean |LP|")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(frameon=False)
    save_plot(fig, output_path, cfg.dpi)
    return True


def clamp_saturation(df: pd.DataFrame, cfg: PlotConfig) -> pd.Series:
    if "lpm_raw_used" not in df.columns:
        return pd.Series(np.nan, index=df.index)
    raw = pd.to_numeric(df["lpm_raw_used"], errors="coerce")
    if cfg.clamp_max is not None:
        return (raw >= cfg.clamp_max - 1.0e-12).astype(float)
    finite_raw = raw[np.isfinite(raw)]
    if finite_raw.empty:
        return pd.Series(np.nan, index=df.index)
    inferred_max = finite_raw.max()
    if inferred_max <= 0:
        return pd.Series(0.0, index=df.index)
    return np.isclose(raw, inferred_max, rtol=1.0e-7, atol=1.0e-12).astype(float)


def clamp_metric_tables(logs: list[RunLogs], cfg: PlotConfig) -> list[pd.DataFrame]:
    tables = []
    for run in logs:
        if run.transition is None or "env_step" not in run.transition.columns:
            continue
        df = run.transition.copy()
        df["env_step_bin"] = bin_env_step(df, cfg.bin_size)
        if "warmup_active" in df.columns:
            df["warmup_rate"] = pd.to_numeric(df["warmup_active"], errors="coerce")
        else:
            df["warmup_rate"] = np.nan
        if "lpm_raw_used" in df.columns:
            raw = pd.to_numeric(df["lpm_raw_used"], errors="coerce")
            df["raw_zero_rate"] = np.isclose(raw, 0.0, atol=1.0e-12).astype(float)
            df["clamp_saturation_rate"] = clamp_saturation(df, cfg)
        else:
            df["raw_zero_rate"] = np.nan
            df["clamp_saturation_rate"] = np.nan
        tables.append(
            df.groupby("env_step_bin", as_index=False)[
                ["warmup_rate", "raw_zero_rate", "clamp_saturation_rate"]
            ].mean()
        )
    return tables


def plot_clamp_warmup(logs: list[RunLogs], output_path: Path, cfg: PlotConfig) -> bool:
    tables = clamp_metric_tables(logs, cfg)
    if not tables:
        return False

    fig, ax = plt.subplots(figsize=(8, 5))
    for metric in ["warmup_rate", "clamp_saturation_rate", "raw_zero_rate"]:
        agg = aggregate_run_series(tables, x_col="env_step_bin", y_col=metric)
        if not agg.empty:
            line_mean_std(ax, agg, x_col="env_step_bin", label=metric, cfg=cfg)
    ax.set_ylim(-0.02, 1.02)
    ax.set_title("Warm-up and clamp saturation rates")
    ax.set_xlabel("env_step")
    ax.set_ylabel("rate")
    ax.grid(True, alpha=0.3)
    ax.legend(frameon=False)
    save_plot(fig, output_path, cfg.dpi)
    return True


def plot_first_reward_ecdf(logs: list[RunLogs], output_path: Path, cfg: PlotConfig) -> bool:
    first_steps = []
    censored_steps = []
    for run in logs:
        if run.transition is None or not require_columns(
            run.transition, ["env_step", "extrinsic_reward"]
        ):
            continue
        df = run.transition[["env_step", "extrinsic_reward"]].dropna()
        if df.empty:
            continue
        found = df[df["extrinsic_reward"] > 0]
        if found.empty:
            censored_steps.append(float(df["env_step"].max()))
        else:
            first_steps.append(float(found["env_step"].min()))
    total = len(first_steps) + len(censored_steps)
    if total == 0:
        return False

    fig, ax = plt.subplots(figsize=(7, 5))
    if first_steps:
        xs = np.sort(np.asarray(first_steps, dtype=float))
        ys = np.arange(1, len(xs) + 1) / total
        ax.step(xs, ys, where="post", linewidth=2.2, label="first reward ECDF")
        ax.scatter(xs, ys, s=28)
    if censored_steps:
        ax.scatter(
            censored_steps,
            np.zeros(len(censored_steps)),
            marker="x",
            s=50,
            label="not found by final logged step",
        )
    ax.set_ylim(-0.02, 1.02)
    ax.set_title(f"First extrinsic reward discovery ECDF (n={total})")
    ax.set_xlabel("first env_step with extrinsic_reward > 0")
    ax.set_ylabel("fraction of runs found")
    ax.grid(True, alpha=0.3)
    ax.legend(frameon=False)
    save_plot(fig, output_path, cfg.dpi)
    return True


def write_summary(
    logs: list[RunLogs], output_dir: Path, created: list[str], skipped: list[str]
) -> None:
    rows = []
    for run in logs:
        rows.append(
            {
                "run_id": run.run_id,
                "run_dir": str(run.run_dir),
                "probe_rows": 0 if run.probe is None else len(run.probe),
                "transition_rows": 0 if run.transition is None else len(run.transition),
                "update_rows": 0 if run.update is None else len(run.update),
            }
        )
    pd.DataFrame(rows).to_csv(output_dir / "lpm_diagnostics_runs.csv", index=False)
    with (output_dir / "lpm_diagnostics_summary.txt").open("w", encoding="utf-8") as file:
        file.write("Created plots:\n")
        for name in created:
            file.write(f"- {name}\n")
        file.write("\nSkipped plots:\n")
        for name in skipped:
            file.write(f"- {name}\n")


def main() -> None:
    args = parse_args()
    group_dir = Path(args.group_dir)
    output_dir = Path(args.output_dir) if args.output_dir else group_dir / "plots_lpm_diagnostics"
    output_dir.mkdir(parents=True, exist_ok=True)
    cfg = PlotConfig(
        dpi=int(args.dpi),
        smoothing_window=int(args.smoothing_window),
        bin_size=int(args.bin_size),
        scatter_sample=int(args.scatter_sample),
        show_raw_runs=bool(args.show_raw_runs),
        show_std_band=not bool(args.no_std_band),
        clamp_max=args.clamp_max,
    )

    logs = find_run_logs(group_dir)
    if not logs:
        raise SystemExit(f"No LPM CSV logs found under {group_dir}")

    probe_all = concat_probe(logs)
    created: list[str] = []
    skipped: list[str] = []

    plot_jobs: list[tuple[str, Callable[[], bool]]] = [
        (
            "01_predicted_previous_error_vs_previous_error_log.png",
            lambda: plot_scatter(
                probe_all,
                x="previous_error_log",
                y="predicted_previous_error",
                title="Predicted previous error vs actual previous error",
                output_path=output_dir / "01_predicted_previous_error_vs_previous_error_log.png",
                cfg=cfg,
            ),
        ),
        (
            "02_estimated_lpm_vs_oracle_pointwise_progress.png",
            lambda: plot_scatter(
                probe_all,
                x="oracle_pointwise_progress",
                y="estimated_lpm_signed",
                title="Estimated LPM vs oracle pointwise progress",
                output_path=output_dir / "02_estimated_lpm_vs_oracle_pointwise_progress.png",
                cfg=cfg,
                zero_lines=True,
            ),
        ),
        (
            "03_probe_mae_bias_sign_agreement.png",
            lambda: plot_probe_accuracy(
                logs,
                output_dir / "03_probe_mae_bias_sign_agreement.png",
                cfg,
            ),
        ),
        (
            "04_oracle_lp_and_estimated_lp_over_time.png",
            lambda: plot_lp_time(
                logs,
                output_dir / "04_oracle_lp_and_estimated_lp_over_time.png",
                cfg,
            ),
        ),
        (
            "05_depth_env_step_lp_mae_heatmap.png",
            lambda: plot_depth_heatmap(
                probe_all,
                output_dir / "05_depth_env_step_lp_mae_heatmap.png",
                cfg,
            ),
        ),
        (
            "06_transition_positive_negative_lp_rates.png",
            lambda: plot_transition_signs(
                logs,
                output_dir / "06_transition_positive_negative_lp_rates.png",
                cfg,
            ),
        ),
        (
            "07_clamp_saturation_and_warmup.png",
            lambda: plot_clamp_warmup(
                logs,
                output_dir / "07_clamp_saturation_and_warmup.png",
                cfg,
            ),
        ),
        (
            "08_first_reward_discovery_ecdf.png",
            lambda: plot_first_reward_ecdf(
                logs,
                output_dir / "08_first_reward_discovery_ecdf.png",
                cfg,
            ),
        ),
    ]

    for filename, job in plot_jobs:
        if job():
            created.append(filename)
        else:
            skipped.append(filename)

    write_summary(logs, output_dir, created, skipped)
    print(f"Found {len(logs)} run(s).")
    print(f"Saved {len(created)} plot(s) to {output_dir}")
    if skipped:
        print("Skipped plots:")
        for name in skipped:
            print(f"  - {name}")


if __name__ == "__main__":
    main()
