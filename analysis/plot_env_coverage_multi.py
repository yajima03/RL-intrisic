from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import pandas as pd


"""
This script aggregates multiple simulation runs (with env_status_log CSVs) under a common group directory and plots mean coverage metrics with optional raw runs and std bands.

Example usage:
python analysis/plot_env_coverage_multi.py \
  --group-dir outputs/multiseed_sp_base_dqn_none_10sim_20260423-012700
"""

EVAL_PATTERNS = [
    re.compile(r"sp_img_train_step(?P<label>\d+)_"),
    re.compile(r"sp_img_train_(?P<label>\d+)_wid\d+_"),
    re.compile(r"sp_img_train_(?P<label>\d+)_"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Aggregate multiple run directories and plot averaged SP coverage metrics."
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
        help="Output directory. Default: <group-dir>/coverage_plots_multi",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=140,
        help="Figure DPI.",
    )
    parser.add_argument(
        "--smoothing-window",
        type=int,
        default=1,
        help="Rolling mean window for smoothed curve. 1 disables smoothing.",
    )
    parser.add_argument(
        "--show-raw-runs",
        action="store_true",
        help="Overlay each run as faint lines.",
    )
    parser.add_argument(
        "--no-std-band",
        action="store_true",
        help="Disable mean±std shaded band.",
    )
    return parser.parse_args()


def resolve_output_dir(group_dir: Path, output_dir_arg: Optional[str]) -> Path:
    output_dir = Path(output_dir_arg) if output_dir_arg else group_dir / "coverage_plots_multi"
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def extract_label(path: Path) -> int:
    name = path.name
    for pattern in EVAL_PATTERNS:
        m = pattern.search(name)
        if m:
            return int(m.group("label"))
    digits = re.findall(r"(\d+)", path.stem)
    if digits:
        return int(digits[0])
    return 0


def read_env_status_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, skiprows=2)
    df.columns = [str(c).strip() for c in df.columns]
    needed = {"index", "depth", "visit_count"}
    missing = needed - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns {missing} in {path}")
    df["index"] = pd.to_numeric(df["index"], errors="raise").astype(int)
    df["depth"] = pd.to_numeric(df["depth"], errors="raise").astype(int)
    df["visit_count"] = pd.to_numeric(df["visit_count"], errors="coerce").fillna(0).astype(int)
    return df


def rolling(series: pd.Series, window: int) -> pd.Series:
    window = max(int(window), 1)
    return series.rolling(window=window, min_periods=1).mean()


def find_run_dirs(group_dir: Path) -> List[Path]:
    run_dirs: List[Path] = []
    for p in sorted(group_dir.rglob("*")):
        if not p.is_dir():
            continue
        env_status_dir = p / "logs" / "env_status_log"
        if env_status_dir.is_dir() and any(env_status_dir.glob("sp_img_train*.csv")):
            run_dirs.append(p)
    unique = []
    seen = set()
    for p in run_dirs:
        rp = p.resolve()
        if rp not in seen:
            unique.append(p)
            seen.add(rp)
    return unique


def list_eval_csvs(env_status_dir: Path) -> List[Path]:
    files = [p for p in env_status_dir.glob("sp_img_train*.csv") if p.is_file()]
    files.sort(key=lambda p: (extract_label(p), p.name))
    return files


def compute_run_coverage(run_dir: Path) -> Tuple[pd.DataFrame, pd.DataFrame, int]:
    env_status_dir = run_dir / "logs" / "env_status_log"
    files = list_eval_csvs(env_status_dir)
    if not files:
        raise FileNotFoundError(f"No train CSVs found in {env_status_dir}")

    summary_rows = []
    cumulative_seen = set()
    total_states = None
    all_depths = set()
    layer_rows = []

    for path in files:
        label = extract_label(path)
        df = read_env_status_csv(path)
        if total_states is None:
            total_states = int(len(df))
        elif int(len(df)) != total_states:
            raise ValueError(
                f"State count changed across files in {run_dir}. "
                f"Expected {total_states}, got {len(df)} in {path.name}."
            )

        visited_mask = df["visit_count"] > 0
        visited_indices = set(df.loc[visited_mask, "index"].tolist())
        cumulative_seen |= visited_indices

        overall_coverage = float(len(visited_indices) / total_states)
        cumulative_unique_states = int(len(cumulative_seen))
        cumulative_unique_ratio = float(cumulative_unique_states / total_states)

        summary_rows.append(
            {
                "label": label,
                "overall_coverage": overall_coverage,
                "cumulative_unique_states": cumulative_unique_states,
                "cumulative_unique_ratio": cumulative_unique_ratio,
            }
        )

        depth_groups = df.groupby("depth", sort=True)
        layer_row = {"label": label}
        for depth, g in depth_groups:
            depth = int(depth)
            all_depths.add(depth)
            layer_row[f"depth_{depth}"] = float((g["visit_count"] > 0).mean())
        layer_rows.append(layer_row)

    summary_df = pd.DataFrame(summary_rows).sort_values("label").reset_index(drop=True)
    layer_df = pd.DataFrame(layer_rows).sort_values("label").reset_index(drop=True)

    for depth in sorted(all_depths):
        col = f"depth_{depth}"
        if col not in layer_df.columns:
            layer_df[col] = 0.0
    ordered_cols = ["label"] + [f"depth_{d}" for d in sorted(all_depths)]
    layer_df = layer_df[ordered_cols]

    return summary_df, layer_df, int(total_states)


def aggregate_metric_tables(run_tables: List[pd.DataFrame], x_col: str = "label") -> Dict[str, pd.DataFrame]:
    metric_names = [c for c in run_tables[0].columns if c != x_col]
    aggregated: Dict[str, pd.DataFrame] = {}
    for metric in metric_names:
        series_list = []
        for ridx, df in enumerate(run_tables):
            s = df[[x_col, metric]].copy().rename(columns={metric: f"run_{ridx}"})
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
        aggregated[metric] = merged
    return aggregated


def save_line_plot(
    agg_df: pd.DataFrame,
    *,
    raw_run_tables: Optional[List[pd.DataFrame]],
    x: str,
    y: str,
    output_path: Path,
    dpi: int,
    smoothing_window: int = 1,
    title: Optional[str] = None,
    ylabel: Optional[str] = None,
    show_raw_runs: bool = False,
    show_std_band: bool = True,
) -> None:
    plt.figure(figsize=(7, 4))
    if show_raw_runs and raw_run_tables is not None:
        for df in raw_run_tables:
            if y in df.columns:
                plt.plot(df[x], df[y], alpha=0.12, linewidth=1.0)

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


def save_layerwise_plot(
    layer_agg: Dict[str, pd.DataFrame],
    *,
    raw_layer_tables: Optional[List[pd.DataFrame]],
    output_path: Path,
    dpi: int,
    smoothing_window: int = 1,
    show_raw_runs: bool = False,
) -> None:
    plt.figure(figsize=(8, 5))
    metric_names = sorted(layer_agg.keys(), key=lambda s: int(s.split("_")[1]) if "_" in s else s)
    for metric in metric_names:
        agg_df = layer_agg[metric]
        x = agg_df["label"]
        y = rolling(agg_df["mean"], smoothing_window)
        plt.plot(x, y, label=metric)

        if show_raw_runs and raw_layer_tables is not None:
            for df in raw_layer_tables:
                if metric in df.columns:
                    plt.plot(df["label"], df[metric], alpha=0.06, linewidth=0.8)

    plt.xlabel("label")
    plt.ylabel("coverage ratio")
    plt.title("Aggregated layer-wise coverage vs label")
    plt.legend(ncol=2, fontsize=8)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=dpi)
    plt.close()


def main() -> None:
    args = parse_args()
    group_dir = Path(args.group_dir)
    if not group_dir.exists():
        raise FileNotFoundError(f"group directory not found: {group_dir}")

    output_dir = resolve_output_dir(group_dir, args.output_dir)
    run_dirs = find_run_dirs(group_dir)
    if not run_dirs:
        raise FileNotFoundError(f"No run directories with env_status_log found under: {group_dir}")

    summary_tables: List[pd.DataFrame] = []
    layer_tables: List[pd.DataFrame] = []
    total_states_list = []

    for run_dir in run_dirs:
        summary_df, layer_df, total_states = compute_run_coverage(run_dir)
        summary_tables.append(summary_df)
        layer_tables.append(layer_df)
        total_states_list.append(total_states)

    if len(set(total_states_list)) != 1:
        raise ValueError(f"total_states differs across runs: {sorted(set(total_states_list))}")

    summary_agg = aggregate_metric_tables(summary_tables, x_col="label")
    layer_agg = aggregate_metric_tables(layer_tables, x_col="label")

    for metric, df in summary_agg.items():
        df.to_csv(output_dir / f"{metric}_multi.csv", index=False)
    for metric, df in layer_agg.items():
        df.to_csv(output_dir / f"{metric}_layer_multi.csv", index=False)

    save_line_plot(
        summary_agg["overall_coverage"],
        raw_run_tables=summary_tables,
        x="label",
        y="overall_coverage",
        output_path=output_dir / "overall_coverage_vs_label_multi.png",
        dpi=args.dpi,
        smoothing_window=args.smoothing_window,
        title="Aggregated overall coverage vs label",
        ylabel="coverage ratio",
        show_raw_runs=args.show_raw_runs,
        show_std_band=not args.no_std_band,
    )
    save_line_plot(
        summary_agg["cumulative_unique_states"],
        raw_run_tables=summary_tables,
        x="label",
        y="cumulative_unique_states",
        output_path=output_dir / "cumulative_unique_states_vs_label_multi.png",
        dpi=args.dpi,
        smoothing_window=args.smoothing_window,
        title="Aggregated cumulative unique states vs label",
        ylabel="unique states",
        show_raw_runs=args.show_raw_runs,
        show_std_band=not args.no_std_band,
    )
    save_line_plot(
        summary_agg["cumulative_unique_ratio"],
        raw_run_tables=summary_tables,
        x="label",
        y="cumulative_unique_ratio",
        output_path=output_dir / "cumulative_unique_ratio_vs_label_multi.png",
        dpi=args.dpi,
        smoothing_window=args.smoothing_window,
        title="Aggregated cumulative unique ratio vs label",
        ylabel="coverage ratio",
        show_raw_runs=args.show_raw_runs,
        show_std_band=not args.no_std_band,
    )
    save_layerwise_plot(
        layer_agg,
        raw_layer_tables=layer_tables,
        output_path=output_dir / "layerwise_coverage_vs_label_multi.png",
        dpi=args.dpi,
        smoothing_window=args.smoothing_window,
        show_raw_runs=args.show_raw_runs,
    )

    with (output_dir / "coverage_plot_summary_multi.txt").open("w", encoding="utf-8") as f:
        f.write(f"group_dir: {group_dir}\n")
        f.write(f"num_runs: {len(run_dirs)}\n")
        f.write(f"total_states: {total_states_list[0]}\n")
        f.write(f"smoothing_window: {args.smoothing_window}\n")
        f.write(f"dpi: {args.dpi}\n")
        f.write(f"show_raw_runs: {args.show_raw_runs}\n")
        f.write(f"show_std_band: {not args.no_std_band}\n")
        f.write("run_dirs:\n")
        for rd in run_dirs:
            f.write(f"- {rd}\n")
        f.write("layer_columns:\n")
        for metric in sorted(layer_agg.keys(), key=lambda s: int(s.split('_')[1]) if "_" in s else s):
            f.write(f"- {metric}\n")

    print(f"Scanned {len(run_dirs)} run directories from: {group_dir}")
    print(f"Saved aggregated plots and CSV summaries to: {output_dir}")


if __name__ == "__main__":
    main()
