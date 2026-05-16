from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

"""
This script analyzes sp_img_train_step*.csv files containing node visit counts by layer and generates Lorenz/concentration curve plots and Gini coefficients.

Usage:
python analysis/plot_sp_visit_gini.py outputs/dqn_sp_20260422-161654/logs/env_status_log/sp_img_train_step70000_2026-04-22_17-01-22.csv
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot layer-wise Lorenz / concentration curves and Gini coefficients from sp_img_train_step*.csv"
    )
    parser.add_argument(
        "csv_path",
        type=str,
        help="Path to sp_img_train_step{step}_{date}.csv",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directory to save plots/csv. Default: <csv_dir>/<csv_stem>_gini_plots",
    )
    parser.add_argument(
        "--final-layer-only",
        action="store_true",
        help="If set, only plot the final layer.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=160,
        help="Figure DPI.",
    )
    return parser.parse_args()


def load_env_status_csv(path: Path) -> pd.DataFrame:
    # First line: metadata header, second line: seed values, third line: actual columns
    df = pd.read_csv(path, skiprows=2, skipinitialspace=True)
    df.columns = [str(c).strip() for c in df.columns]

    required = {"depth", "visit_count"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns {sorted(missing)} in {path}")

    df["depth"] = pd.to_numeric(df["depth"], errors="raise").astype(int)
    df["visit_count"] = pd.to_numeric(df["visit_count"], errors="coerce").fillna(0).astype(float)
    return df


def compute_gini(counts: np.ndarray) -> float:
    x = np.asarray(counts, dtype=np.float64)
    if x.size == 0:
        return 0.0
    if np.allclose(x.sum(), 0.0):
        return 0.0

    x = np.sort(x)  # ascending for standard Lorenz / Gini
    n = x.size
    cum_weighted = np.sum((np.arange(1, n + 1, dtype=np.float64)) * x)
    gini = (2.0 * cum_weighted) / (n * x.sum()) - (n + 1) / n
    return float(gini)


def lorenz_curve(counts: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    x = np.asarray(counts, dtype=np.float64)
    n = x.size
    if n == 0:
        return np.array([0.0, 1.0]), np.array([0.0, 1.0])

    x = np.sort(x)  # ascending (standard Lorenz)
    total = x.sum()
    if np.allclose(total, 0.0):
        return np.linspace(0.0, 1.0, n + 1), np.linspace(0.0, 1.0, n + 1)

    cum_counts = np.concatenate([[0.0], np.cumsum(x) / total])
    cum_nodes = np.linspace(0.0, 1.0, n + 1)
    return cum_nodes, cum_counts


def concentration_curve_desc(counts: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Descending cumulative share curve.

    x-axis: top-k node fraction (largest visit counts first)
    y-axis: cumulative visit share captured by top-k nodes
    """
    x = np.asarray(counts, dtype=np.float64)
    n = x.size
    if n == 0:
        return np.array([0.0, 1.0]), np.array([0.0, 1.0])

    x = np.sort(x)[::-1]  # descending
    total = x.sum()
    if np.allclose(total, 0.0):
        return np.linspace(0.0, 1.0, n + 1), np.zeros(n + 1, dtype=np.float64)

    cum_counts = np.concatenate([[0.0], np.cumsum(x) / total])
    cum_nodes = np.linspace(0.0, 1.0, n + 1)
    return cum_nodes, cum_counts


def top_share(counts: np.ndarray, top_fraction: float) -> float:
    x = np.asarray(counts, dtype=np.float64)
    if x.size == 0 or np.allclose(x.sum(), 0.0):
        return 0.0
    x = np.sort(x)[::-1]
    k = max(1, int(math.ceil(float(top_fraction) * x.size)))
    return float(x[:k].sum() / x.sum())


def save_lorenz_plot(depth: int, counts: np.ndarray, output_dir: Path, dpi: int) -> None:
    x, y = lorenz_curve(counts)
    gini = compute_gini(counts)

    plt.figure(figsize=(5.2, 4.2))
    plt.plot(x, y, marker="o", markersize=3, label="Lorenz curve")
    plt.plot([0, 1], [0, 1], linestyle="--", label="Perfect equality")
    plt.fill_between(x, y, x, alpha=0.25)
    plt.xlabel("Cumulative fraction of nodes (ascending by visits)")
    plt.ylabel("Cumulative fraction of visits")
    plt.title(f"Layer {depth} Lorenz curve (Gini={gini:.4f})")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / f"layer_{depth:02d}_lorenz.png", dpi=dpi)
    plt.close()


def save_concentration_plot(depth: int, counts: np.ndarray, output_dir: Path, dpi: int) -> None:
    x, y = concentration_curve_desc(counts)
    gini = compute_gini(counts)
    share60 = np.interp(0.60, x, y)

    plt.figure(figsize=(5.2, 4.2))
    plt.plot(x, y, marker="o", markersize=3, label="Concentration curve")
    plt.plot([0, 1], [0, 1], linestyle="--", label="Perfect equality")
    plt.fill_between(x, y, x, alpha=0.25)
    plt.xlabel("Top fraction of nodes (descending by visits)")
    plt.ylabel("Cumulative fraction of visits")
    plt.title(f"Layer {depth} concentration curve (Gini={gini:.4f})")
    plt.annotate(
        f"Top 60% nodes -> {share60:.3f} of visits",
        xy=(0.60, share60),
        xytext=(0.40, min(0.95, share60 + 0.12)),
        arrowprops={"arrowstyle": "->"},
        fontsize=9,
    )
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / f"layer_{depth:02d}_concentration_desc.png", dpi=dpi)
    plt.close()


def save_summary_plots(summary_df: pd.DataFrame, output_dir: Path, dpi: int) -> None:
    if summary_df.empty:
        return

    # Gini by depth
    plt.figure(figsize=(5.8, 4.2))
    plt.plot(summary_df["depth"], summary_df["gini"], marker="o")
    plt.xlabel("Layer depth")
    plt.ylabel("Gini coefficient")
    plt.title("Gini coefficient by layer")
    plt.tight_layout()
    plt.savefig(output_dir / "gini_by_layer.png", dpi=dpi)
    plt.close()

    # top-60% share by depth
    plt.figure(figsize=(5.8, 4.2))
    plt.plot(summary_df["depth"], summary_df["top_60pct_share"], marker="o")
    plt.xlabel("Layer depth")
    plt.ylabel("Visit share captured by top 60% nodes")
    plt.title("Top-60% concentration by layer")
    plt.tight_layout()
    plt.savefig(output_dir / "top60_share_by_layer.png", dpi=dpi)
    plt.close()


def main() -> None:
    args = parse_args()
    csv_path = Path(args.csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    output_dir = Path(args.output_dir) if args.output_dir is not None else csv_path.parent / f"{csv_path.stem}_gini_plots"
    output_dir.mkdir(parents=True, exist_ok=True)

    df = load_env_status_csv(csv_path)
    depths = sorted(df["depth"].unique().tolist())
    if args.final_layer_only:
        depths = [max(depths)]

    summary_rows: List[Dict[str, float]] = []

    for depth in depths:
        layer_df = df[df["depth"] == depth].copy()
        counts = layer_df["visit_count"].to_numpy(dtype=np.float64)
        total_visits = float(counts.sum())
        num_nodes = int(counts.size)
        visited_nodes = int((counts > 0).sum())
        coverage = float(visited_nodes / num_nodes) if num_nodes > 0 else 0.0
        gini = compute_gini(counts)

        summary_rows.append(
            {
                "depth": depth,
                "num_nodes": num_nodes,
                "visited_nodes": visited_nodes,
                "coverage": coverage,
                "total_visits": total_visits,
                "gini": gini,
                "top_10pct_share": top_share(counts, 0.10),
                "top_25pct_share": top_share(counts, 0.25),
                "top_50pct_share": top_share(counts, 0.50),
                "top_60pct_share": top_share(counts, 0.60),
                "top_75pct_share": top_share(counts, 0.75),
            }
        )

        save_lorenz_plot(depth, counts, output_dir, args.dpi)
        save_concentration_plot(depth, counts, output_dir, args.dpi)

    summary_df = pd.DataFrame(summary_rows).sort_values("depth").reset_index(drop=True)
    summary_df.to_csv(output_dir / "layer_gini_summary.csv", index=False)
    save_summary_plots(summary_df, output_dir, args.dpi)

    readme_path = output_dir / "README.txt"
    with readme_path.open("w", encoding="utf-8") as f:
        f.write("This directory contains layer-wise visit-distribution plots.\n\n")
        f.write("Files:\n")
        f.write("- layer_{depth}_lorenz.png: standard Lorenz curve (ascending by visits) used for Gini\n")
        f.write("- layer_{depth}_concentration_desc.png: descending concentration curve (top-k nodes first)\n")
        f.write("- layer_gini_summary.csv: numeric summary per layer\n")
        f.write("- gini_by_layer.png: Gini coefficient by depth\n")
        f.write("- top60_share_by_layer.png: share of visits captured by top 60% nodes\n")

    print(f"Loaded: {csv_path}")
    print(f"Saved outputs to: {output_dir}")


if __name__ == "__main__":
    main()
