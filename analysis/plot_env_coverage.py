from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Iterable, Optional

import matplotlib.pyplot as plt
import pandas as pd

"""
sp タスクを実行した際の env_status_log 内の sp_img_train_*.csv をスキャンして、カバレッジ指標をプロットするスクリプト。

使い方:
  uv run python -m analysis.plot_env_coverage --env-status-dir outputs/dqn_sp_20260418-171614/logs/env_status_log
"""


EVAL_PATTERNS = [
    re.compile(r"sp_img_train_step(?P<label>\d+)_"),
    re.compile(r"sp_img_train_(?P<label>\d+)_wid\d+_"),
    re.compile(r"sp_img_train_(?P<label>\d+)_"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Scan env_status_log train CSVs and plot coverage metrics."
    )
    parser.add_argument(
        "--env-status-dir",
        type=str,
        default=None,
        help="Path to env_status_log directory.",
    )
    parser.add_argument(
        "--run-dir",
        type=str,
        default=None,
        help="Run directory like outputs/<run_name>. If given, env_status_log is read from <run-dir>/logs/env_status_log.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directory to save plots/csv. Default: <env-status-dir>/coverage_plots",
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
    return parser.parse_args()


def resolve_dirs(args: argparse.Namespace) -> tuple[Path, Path]:
    if args.env_status_dir is not None:
        env_status_dir = Path(args.env_status_dir)
    elif args.run_dir is not None:
        env_status_dir = Path(args.run_dir) / "logs" / "env_status_log"
    else:
        raise ValueError("Either --env-status-dir or --run-dir must be provided.")

    if not env_status_dir.exists():
        raise FileNotFoundError(f"env_status_log directory not found: {env_status_dir}")

    if args.output_dir is not None:
        output_dir = Path(args.output_dir)
    else:
        output_dir = env_status_dir / "coverage_plots"

    output_dir.mkdir(parents=True, exist_ok=True)
    return env_status_dir, output_dir


def extract_label(path: Path) -> int:
    name = path.name
    for pattern in EVAL_PATTERNS:
        m = pattern.search(name)
        if m:
            return int(m.group("label"))
    # fallback: digits in stem
    digits = re.findall(r"(\d+)", path.stem)
    if digits:
        return int(digits[0])
    return 0


def list_eval_csvs(env_status_dir: Path) -> list[Path]:
    files = [p for p in env_status_dir.glob("sp_img_train*.csv") if p.is_file()]
    if not files:
        raise FileNotFoundError(
            f"No train CSVs found in {env_status_dir}. Expected files like sp_img_train_*.csv"
        )
    files.sort(key=lambda p: (extract_label(p), p.name))
    return files


def read_env_status_csv(path: Path) -> pd.DataFrame:
    # First two lines are seed metadata, third line is actual header.
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


def save_line_plot(
    df: pd.DataFrame,
    *,
    x: str,
    y: str,
    output_path: Path,
    dpi: int,
    smoothing_window: int = 1,
    title: Optional[str] = None,
    ylabel: Optional[str] = None,
) -> None:
    plt.figure(figsize=(7, 4))
    plt.plot(df[x], df[y], alpha=0.35, label="raw")
    if smoothing_window > 1:
        plt.plot(df[x], rolling(df[y], smoothing_window), linewidth=2.0, label=f"smoothed({smoothing_window})")
        plt.legend()
    plt.xlabel(x)
    plt.ylabel(ylabel or y)
    plt.title(title or f"{y} vs {x}")
    plt.tight_layout()
    plt.savefig(output_path, dpi=dpi)
    plt.close()


def save_layerwise_plot(
    layer_df: pd.DataFrame,
    *,
    output_path: Path,
    dpi: int,
    smoothing_window: int = 1,
) -> None:
    plt.figure(figsize=(8, 5))
    x = layer_df["label"]
    for col in layer_df.columns:
        if col == "label":
            continue
        series = layer_df[col]
        if smoothing_window > 1:
            series = rolling(series, smoothing_window)
        plt.plot(x, series, label=col)
    plt.xlabel("label")
    plt.ylabel("coverage ratio")
    plt.title("Layer-wise coverage vs label")
    plt.legend(ncol=2, fontsize=8)
    plt.tight_layout()
    plt.savefig(output_path, dpi=dpi)
    plt.close()


def main() -> None:
    args = parse_args()
    env_status_dir, output_dir = resolve_dirs(args)
    files = list_eval_csvs(env_status_dir)

    summary_rows = []
    cumulative_seen: set[int] = set()
    total_states: Optional[int] = None
    all_depths: set[int] = set()
    layer_rows = []

    for path in files:
        label = extract_label(path)
        df = read_env_status_csv(path)
        if total_states is None:
            total_states = int(len(df))
        elif int(len(df)) != total_states:
            raise ValueError(
                f"State count changed across files. First file had {total_states}, but {path.name} has {len(df)}."
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
                "filename": path.name,
                "total_states": total_states,
                "visited_states": len(visited_indices),
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

    summary_df.to_csv(output_dir / "coverage_summary.csv", index=False)
    layer_df.to_csv(output_dir / "coverage_layerwise.csv", index=False)

    save_line_plot(
        summary_df,
        x="label",
        y="overall_coverage",
        output_path=output_dir / "overall_coverage_vs_label.png",
        dpi=args.dpi,
        smoothing_window=args.smoothing_window,
        title="Overall coverage vs label",
        ylabel="coverage ratio",
    )
    save_line_plot(
        summary_df,
        x="label",
        y="cumulative_unique_states",
        output_path=output_dir / "cumulative_unique_states_vs_label.png",
        dpi=args.dpi,
        smoothing_window=args.smoothing_window,
        title="Cumulative unique states vs label",
        ylabel="unique states",
    )
    save_line_plot(
        summary_df,
        x="label",
        y="cumulative_unique_ratio",
        output_path=output_dir / "cumulative_unique_ratio_vs_label.png",
        dpi=args.dpi,
        smoothing_window=args.smoothing_window,
        title="Cumulative unique state ratio vs label",
        ylabel="coverage ratio",
    )
    save_layerwise_plot(
        layer_df,
        output_path=output_dir / "layerwise_coverage_vs_label.png",
        dpi=args.dpi,
        smoothing_window=args.smoothing_window,
    )

    with (output_dir / "coverage_plot_summary.txt").open("w", encoding="utf-8") as f:
        f.write(f"env_status_dir: {env_status_dir}\n")
        f.write(f"num_files: {len(files)}\n")
        f.write(f"label_min: {summary_df['label'].min()}\n")
        f.write(f"label_max: {summary_df['label'].max()}\n")
        f.write(f"total_states: {total_states}\n")
        f.write(f"smoothing_window: {args.smoothing_window}\n")
        f.write(f"dpi: {args.dpi}\n")
        f.write("depth_columns:\n")
        for col in layer_df.columns:
            if col != "label":
                f.write(f"- {col}\n")

    print(f"Scanned {len(files)} train CSV files from: {env_status_dir}")
    print(f"Saved plots and CSV summaries to: {output_dir}")


if __name__ == "__main__":
    main()
