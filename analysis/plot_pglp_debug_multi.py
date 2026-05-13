from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import pandas as pd


""""
This script aggregates multiple pglp/pwlp debug CSVs under a common group directory and plots mean metrics with optional raw runs and std bands.

Example usage:
python analysis/plot_pglp_debug_multi.py \
  --group-dir outputs/multiseed_sp_base_dqn_pglp_local_10sim_20260423-012700

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
    "mean_signed_gap",
    "mean_positive_gap",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aggregate multiple pglp/pwlp debug CSVs and plot mean metrics.")
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
        help="Directory to save plots. Default: <group-dir>/plots_pglp_multi",
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
    return parser.parse_args()


def resolve_output_dir(group_dir: Path, output_dir_arg: Optional[str]) -> Path:
    output_dir = Path(output_dir_arg) if output_dir_arg else group_dir / "plots_pglp_multi"
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def find_csv_paths(group_dir: Path) -> List[Path]:
    csv_paths = []
    for p in sorted(group_dir.rglob("pglp_debug.csv")):
        if p.is_file() and p.parent.name == "logs":
            csv_paths.append(p)
    for p in sorted(group_dir.rglob("pwlp_debug.csv")):
        if p.is_file() and p.parent.name == "logs":
            csv_paths.append(p)
    unique = []
    seen = set()
    for p in csv_paths:
        rp = p.resolve()
        if rp not in seen:
            unique.append(p)
            seen.add(rp)
    return unique


def load_debug_csv(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    if "timestep" not in df.columns:
        raise ValueError(f"{csv_path} must contain a 'timestep' column.")

    for col in df.columns:
        if col != "timestep":
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.sort_values("timestep").reset_index(drop=True)
    return df


def rolling(series: pd.Series, window: int) -> pd.Series:
    window = max(int(window), 1)
    return series.rolling(window=window, min_periods=1).mean()


def aggregate_metric_tables(run_tables: List[pd.DataFrame], x_col: str = "timestep") -> Dict[str, pd.DataFrame]:
    metric_names = sorted(set().union(*[set(df.columns) for df in run_tables]))
    metric_names = [m for m in metric_names if m != x_col]

    aggregated: Dict[str, pd.DataFrame] = {}
    for metric in metric_names:
        series_list = []
        has_any = False
        for ridx, df in enumerate(run_tables):
            if metric not in df.columns:
                continue
            has_any = True
            s = df[[x_col, metric]].copy().rename(columns={metric: f"run_{ridx}"})
            s = s.groupby(x_col, as_index=False).mean(numeric_only=True)
            series_list.append(s)

        if not has_any or not series_list:
            continue

        merged = series_list[0]
        for s in series_list[1:]:
            merged = merged.merge(s, on=x_col, how="outer")
        merged = merged.sort_values(x_col).reset_index(drop=True)
        run_cols = [c for c in merged.columns if c != x_col]
        merged["mean"] = merged[run_cols].mean(axis=1, skipna=True)
        merged["std"] = merged[run_cols].std(axis=1, ddof=0, skipna=True)
        merged["count"] = merged[run_cols].notna().sum(axis=1)
        aggregated[metric] = merged

    return aggregated


def save_single_plot(
    agg_df: pd.DataFrame,
    *,
    raw_run_tables: Optional[List[pd.DataFrame]],
    x: str,
    y: str,
    output_path: Path,
    smoothing_window: int,
    dpi: int,
    ylabel: Optional[str] = None,
    title: Optional[str] = None,
    show_raw_runs: bool = False,
    show_std_band: bool = True,
) -> None:
    plt.figure(figsize=(7, 4))
    if show_raw_runs and raw_run_tables is not None:
        for df in raw_run_tables:
            if y in df.columns:
                plt.plot(df[x], df[y], alpha=0.10, linewidth=1.0)

    mean_s = rolling(agg_df["mean"], smoothing_window)
    std_s = rolling(agg_df["std"], smoothing_window)
    plt.plot(agg_df[x], mean_s, linewidth=2.2, label="mean")
    if show_std_band:
        plt.fill_between(agg_df[x], mean_s - std_s, mean_s + std_s, alpha=0.2, label="±1 std")
    plt.xlabel(x)
    plt.ylabel(ylabel or y)
    plt.title(title or f"{y} vs {x}")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=dpi)
    plt.close()


def save_dual_plot(
    agg1: pd.DataFrame,
    agg2: pd.DataFrame,
    *,
    output_path: Path,
    smoothing_window: int,
    dpi: int,
    title: str,
) -> None:
    plt.figure(figsize=(7, 4))
    plt.plot(agg1["timestep"], rolling(agg1["mean"], smoothing_window), label="mean_prototype_long", linewidth=2.0)
    plt.plot(agg2["timestep"], rolling(agg2["mean"], smoothing_window), label="mean_prototype_short", linewidth=2.0)
    plt.xlabel("timestep")
    plt.ylabel("value")
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=dpi)
    plt.close()


def save_core_combined_plot(
    agg_tables: Dict[str, pd.DataFrame],
    *,
    output_path: Path,
    smoothing_window: int,
    dpi: int,
) -> None:
    cols = [c for c in ["mean_gate", "mean_progress", "mean_raw_intrinsic", "mean_current_error"] if c in agg_tables]
    if not cols:
        return

    plt.figure(figsize=(7, 4))
    for col in cols:
        plt.plot(
            agg_tables[col]["timestep"],
            rolling(agg_tables[col]["mean"], smoothing_window),
            label=col,
            linewidth=2.0,
        )
    plt.xlabel("timestep")
    plt.ylabel("value")
    plt.title("Aggregated PGLP/PWLP core metrics vs timestep")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=dpi)
    plt.close()


def save_summary_text(
    csv_paths: List[Path],
    output_dir: Path,
    smoothing_window: int,
    dpi: int,
    show_raw_runs: bool,
    show_std_band: bool,
    agg_tables: Dict[str, pd.DataFrame],
) -> None:
    txt_path = output_dir / "pglp_plot_summary_multi.txt"
    with txt_path.open("w", encoding="utf-8") as f:
        f.write(f"num_runs: {len(csv_paths)}\n")
        f.write(f"smoothing_window: {smoothing_window}\n")
        f.write(f"dpi: {dpi}\n")
        f.write(f"show_raw_runs: {show_raw_runs}\n")
        f.write(f"show_std_band: {show_std_band}\n")
        f.write("csv_paths:\n")
        for path in csv_paths:
            f.write(f"- {path}\n")
        f.write("columns:\n")
        for col in sorted(agg_tables.keys()):
            f.write(f"- {col}\n")


def main() -> None:
    args = parse_args()
    group_dir = Path(args.group_dir)
    if not group_dir.exists():
        raise FileNotFoundError(f"group directory not found: {group_dir}")

    output_dir = resolve_output_dir(group_dir, args.output_dir)
    csv_paths = find_csv_paths(group_dir)
    if not csv_paths:
        raise FileNotFoundError(f"No pglp_debug.csv or pwlp_debug.csv found under: {group_dir}")

    run_tables = [load_debug_csv(p) for p in csv_paths]
    agg_tables = aggregate_metric_tables(run_tables, x_col="timestep")

    if not agg_tables:
        raise ValueError("No aggregatable columns found in debug CSVs.")

    for col, agg_df in agg_tables.items():
        agg_df.to_csv(output_dir / f"{col}_multi.csv", index=False)

    for col in DEFAULT_COLUMNS:
        if col not in agg_tables:
            continue
        save_single_plot(
            agg_tables[col],
            raw_run_tables=run_tables,
            x="timestep",
            y=col,
            output_path=output_dir / f"{col}_vs_timestep_multi.png",
            smoothing_window=args.smoothing_window,
            dpi=args.dpi,
            show_raw_runs=args.show_raw_runs,
            show_std_band=not args.no_std_band,
        )

    if "mean_prototype_long" in agg_tables and "mean_prototype_short" in agg_tables:
        save_dual_plot(
            agg_tables["mean_prototype_long"],
            agg_tables["mean_prototype_short"],
            output_path=output_dir / "prototype_long_short_vs_timestep_multi.png",
            smoothing_window=args.smoothing_window,
            dpi=args.dpi,
            title="Aggregated prototype EMA long vs short",
        )

    save_core_combined_plot(
        agg_tables,
        output_path=output_dir / "pglp_core_metrics_vs_timestep_multi.png",
        smoothing_window=args.smoothing_window,
        dpi=args.dpi,
    )
    save_summary_text(
        csv_paths,
        output_dir,
        args.smoothing_window,
        args.dpi,
        args.show_raw_runs,
        not args.no_std_band,
        agg_tables,
    )

    print(f"Loaded {len(csv_paths)} debug CSV files from: {group_dir}")
    print(f"Saved plots to: {output_dir}")


if __name__ == "__main__":
    main()
