from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, Optional

import matplotlib.pyplot as plt
import pandas as pd


"""
pglp_debug.csvの各種指標を時系列でプロットするスクリプト。

使い方:
    uv run python -m analysis.plot_pglp_debug --run-dir outputs/dqn_sp_20260418-171614 
"""


DEFAULT_COLUMNS = [
    "mean_gate",
    "mean_progress",
    "mean_raw_intrinsic",
    "mean_current_error",
    "mean_neighbor_count",
    "mean_prototype_count_for_action",
    "mean_prototype_long",
    "mean_prototype_short",
    "mean_prototype_index",
    "num_samples",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot pglp_debug.csv metrics.")
    parser.add_argument(
        "--run-dir",
        type=str,
        default=None,
        help="Run directory like outputs/<run_name>. If given, CSV is read from <run-dir>/logs/pglp_debug.csv and plots are saved to <run-dir>/plots_pglp.",
    )
    parser.add_argument(
        "--csv-path",
        type=str,
        default=None,
        help="Direct path to pglp_debug.csv. If omitted, --run-dir must be given.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directory to save plots. Default: <run-dir>/plots_pglp or <csv-dir>/plots_pglp",
    )
    parser.add_argument(
        "--smoothing-window",
        type=int,
        default=5,
        help="Rolling mean window for smoothed line.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=140,
        help="Figure DPI.",
    )
    return parser.parse_args()


def resolve_paths(args: argparse.Namespace) -> tuple[Path, Path]:
    if args.csv_path is not None:
        csv_path = Path(args.csv_path)
    elif args.run_dir is not None:
        csv_path = Path(args.run_dir) / "logs" / "pglp_debug.csv"
    else:
        raise ValueError("Either --run-dir or --csv-path must be provided.")

    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    if args.output_dir is not None:
        output_dir = Path(args.output_dir)
    elif args.run_dir is not None:
        output_dir = Path(args.run_dir) / "plots_pglp"
    else:
        output_dir = csv_path.parent / "plots_pglp"

    output_dir.mkdir(parents=True, exist_ok=True)
    return csv_path, output_dir


def load_debug_csv(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    if "timestep" not in df.columns:
        raise ValueError("pglp_debug.csv must contain a 'timestep' column.")

    for col in df.columns:
        if col != "timestep":
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.sort_values("timestep").reset_index(drop=True)
    return df


def rolling(series: pd.Series, window: int) -> pd.Series:
    window = max(int(window), 1)
    return series.rolling(window=window, min_periods=1).mean()


def save_single_plot(
    df: pd.DataFrame,
    *,
    x: str,
    y: str,
    output_path: Path,
    smoothing_window: int,
    dpi: int,
    ylabel: Optional[str] = None,
    title: Optional[str] = None,
) -> None:
    plt.figure(figsize=(7, 4))
    plt.plot(df[x], df[y], alpha=0.35, label="raw")
    plt.plot(df[x], rolling(df[y], smoothing_window), linewidth=2.0, label=f"smoothed({smoothing_window})")
    plt.xlabel(x)
    plt.ylabel(ylabel or y)
    plt.title(title or f"{y} vs {x}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=dpi)
    plt.close()


def save_dual_plot(
    df: pd.DataFrame,
    *,
    x: str,
    y1: str,
    y2: str,
    output_path: Path,
    smoothing_window: int,
    dpi: int,
    title: str,
) -> None:
    plt.figure(figsize=(7, 4))
    plt.plot(df[x], rolling(df[y1], smoothing_window), label=y1, linewidth=2.0)
    plt.plot(df[x], rolling(df[y2], smoothing_window), label=y2, linewidth=2.0)
    plt.xlabel(x)
    plt.ylabel("value")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=dpi)
    plt.close()


def save_core_combined_plot(
    df: pd.DataFrame,
    *,
    output_path: Path,
    smoothing_window: int,
    dpi: int,
) -> None:
    cols = [c for c in ["mean_gate", "mean_progress", "mean_raw_intrinsic", "mean_current_error"] if c in df.columns]
    if not cols:
        return

    plt.figure(figsize=(7, 4))
    for col in cols:
        plt.plot(df["timestep"], rolling(df[col], smoothing_window), label=col, linewidth=2.0)
    plt.xlabel("timestep")
    plt.ylabel("value")
    plt.title("PGLP core metrics vs timestep")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=dpi)
    plt.close()


def save_summary_text(df: pd.DataFrame, output_dir: Path, smoothing_window: int, dpi: int) -> None:
    txt_path = output_dir / "pglp_plot_summary.txt"
    with txt_path.open("w", encoding="utf-8") as f:
        f.write(f"rows: {len(df)}\n")
        f.write(f"timestep_min: {df['timestep'].min()}\n")
        f.write(f"timestep_max: {df['timestep'].max()}\n")
        f.write(f"smoothing_window: {smoothing_window}\n")
        f.write(f"dpi: {dpi}\n")
        f.write("columns:\n")
        for col in df.columns:
            f.write(f"- {col}\n")


def plot_available_columns(df: pd.DataFrame, output_dir: Path, smoothing_window: int, dpi: int) -> None:
    for col in DEFAULT_COLUMNS:
        if col not in df.columns:
            continue
        save_single_plot(
            df,
            x="timestep",
            y=col,
            output_path=output_dir / f"{col}_vs_timestep.png",
            smoothing_window=smoothing_window,
            dpi=dpi,
        )

    if "mean_prototype_long" in df.columns and "mean_prototype_short" in df.columns:
        save_dual_plot(
            df,
            x="timestep",
            y1="mean_prototype_long",
            y2="mean_prototype_short",
            output_path=output_dir / "prototype_long_short_vs_timestep.png",
            smoothing_window=smoothing_window,
            dpi=dpi,
            title="Prototype EMA long vs short",
        )

    save_core_combined_plot(
        df,
        output_path=output_dir / "pglp_core_metrics_vs_timestep.png",
        smoothing_window=smoothing_window,
        dpi=dpi,
    )


def main() -> None:
    args = parse_args()
    csv_path, output_dir = resolve_paths(args)
    df = load_debug_csv(csv_path)

    if len(df) == 0:
        raise ValueError("pglp_debug.csv is empty.")

    plot_available_columns(
        df,
        output_dir=output_dir,
        smoothing_window=args.smoothing_window,
        dpi=args.dpi,
    )
    save_summary_text(df, output_dir, args.smoothing_window, args.dpi)

    print(f"Loaded: {csv_path}")
    print(f"Saved plots to: {output_dir}")


if __name__ == "__main__":
    main()
