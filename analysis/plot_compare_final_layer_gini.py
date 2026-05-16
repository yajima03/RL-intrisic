from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import List, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

"""
This script compares the final-layer Gini coefficient and concentration across multiple run directories.
Each run directory should contain logs/env_status_log/ with CSV files named like sp_img_train_step{step}_*.csv.

Usage example:
python analysis/plot_compare_final_layer_gini.py \
  --run-dirs \
    outputs/dqn_only_run \
    outputs/rnd_run \
    outputs/count_run \
    outputs/pglp_run \
  --labels \
    DQN \
    RND \
    Count \
    PGLP \
  --step 80000


"""

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare final-layer Gini / concentration across multiple run directories."
    )
    parser.add_argument(
        "--run-dirs",
        nargs="+",
        required=True,
        help="Run directories to compare. Each directory should contain logs/env_status_log/.",
    )
    parser.add_argument(
        "--step",
        type=int,
        required=True,
        help="Target step label. The script searches for sp_img_train_step{step}_*.csv in each run.",
    )
    parser.add_argument(
        "--labels",
        nargs="*",
        default=None,
        help="Optional labels for the runs. Must match the number of --run-dirs if provided.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directory to save outputs. Default: ./gini_compare_step{step}",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=160,
        help="Figure DPI.",
    )
    return parser.parse_args()


def load_env_status_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, skiprows=2, skipinitialspace=True)
    df.columns = [str(c).strip() for c in df.columns]
    required = {"depth", "visit_count"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns {sorted(missing)} in {path}")
    df["depth"] = pd.to_numeric(df["depth"], errors="raise").astype(int)
    df["visit_count"] = pd.to_numeric(df["visit_count"], errors="coerce").fillna(0).astype(float)
    return df


def find_step_csv(run_dir: Path, step: int) -> Path:
    env_status_dir = run_dir / "logs" / "env_status_log"
    if not env_status_dir.exists():
        raise FileNotFoundError(f"env_status_log directory not found: {env_status_dir}")

    matches = sorted(env_status_dir.glob(f"sp_img_train_step{step}_*.csv"))
    if not matches:
        raise FileNotFoundError(
            f"No file matched sp_img_train_step{step}_*.csv under {env_status_dir}"
        )
    return matches[-1]


def compute_gini(counts: np.ndarray) -> float:
    x = np.asarray(counts, dtype=np.float64)
    if x.size == 0 or np.allclose(x.sum(), 0.0):
        return 0.0
    x = np.sort(x)  # ascending
    n = x.size
    cum_weighted = np.sum((np.arange(1, n + 1, dtype=np.float64)) * x)
    gini = (2.0 * cum_weighted) / (n * x.sum()) - (n + 1) / n
    return float(gini)


def concentration_curve_desc(counts: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    x = np.asarray(counts, dtype=np.float64)
    n = x.size
    if n == 0:
        return np.array([0.0, 1.0]), np.array([0.0, 1.0])
    x = np.sort(x)[::-1]
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


def collect_run_stats(run_dir: Path, label: str, step: int) -> dict:
    csv_path = find_step_csv(run_dir, step)
    df = load_env_status_csv(csv_path)
    final_depth = int(df["depth"].max())
    final_df = df[df["depth"] == final_depth].copy()
    counts = final_df["visit_count"].to_numpy(dtype=np.float64)

    x_desc, y_desc = concentration_curve_desc(counts)

    return {
        "label": label,
        "run_dir": str(run_dir),
        "csv_path": str(csv_path),
        "final_depth": final_depth,
        "num_nodes": int(counts.size),
        "visited_nodes": int((counts > 0).sum()),
        "coverage": float((counts > 0).mean()) if counts.size > 0 else 0.0,
        "total_visits": float(counts.sum()),
        "gini": compute_gini(counts),
        "top_50pct_share": top_share(counts, 0.50),
        "top_60pct_share": top_share(counts, 0.60),
        "top_75pct_share": top_share(counts, 0.75),
        "curve_x": x_desc,
        "curve_y": y_desc,
    }


def save_bar_plot(df: pd.DataFrame, column: str, title: str, ylabel: str, output_path: Path, dpi: int) -> None:
    plt.figure(figsize=(max(6.5, 1.0 * len(df)), 4.2))
    plt.bar(df["label"], df[column])
    plt.ylabel(ylabel)
    plt.title(title)
    plt.xticks(rotation=15, ha="right")
    plt.tight_layout()
    plt.savefig(output_path, dpi=dpi)
    plt.close()


def save_concentration_overlay(stats: Sequence[dict], output_path: Path, dpi: int) -> None:
    plt.figure(figsize=(6.2, 4.8))
    for item in stats:
        plt.plot(item["curve_x"], item["curve_y"], label=f"{item['label']} (Gini={item['gini']:.3f})")
    plt.plot([0, 1], [0, 1], linestyle="--", color="black", label="Perfect equality")
    plt.xlabel("Top fraction of final-layer nodes (descending by visits)")
    plt.ylabel("Cumulative fraction of visits")
    plt.title("Final-layer concentration curve comparison")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(output_path, dpi=dpi)
    plt.close()


def main() -> None:
    args = parse_args()
    run_dirs = [Path(p) for p in args.run_dirs]

    if args.labels is not None and len(args.labels) > 0 and len(args.labels) != len(run_dirs):
        raise ValueError("If --labels is provided, it must have the same length as --run-dirs")

    labels = args.labels if args.labels else [p.name for p in run_dirs]
    output_dir = Path(args.output_dir) if args.output_dir else Path(f"gini_compare_step{args.step}")
    output_dir.mkdir(parents=True, exist_ok=True)

    stats: List[dict] = []
    for run_dir, label in zip(run_dirs, labels):
        stats.append(collect_run_stats(run_dir, label, args.step))

    summary_df = pd.DataFrame(
        [
            {
                "label": item["label"],
                "run_dir": item["run_dir"],
                "csv_path": item["csv_path"],
                "final_depth": item["final_depth"],
                "num_nodes": item["num_nodes"],
                "visited_nodes": item["visited_nodes"],
                "coverage": item["coverage"],
                "total_visits": item["total_visits"],
                "gini": item["gini"],
                "top_50pct_share": item["top_50pct_share"],
                "top_60pct_share": item["top_60pct_share"],
                "top_75pct_share": item["top_75pct_share"],
            }
            for item in stats
        ]
    )
    summary_df.to_csv(output_dir / f"final_layer_gini_summary_step{args.step}.csv", index=False)

    save_bar_plot(
        summary_df,
        column="gini",
        title=f"Final-layer Gini comparison at step {args.step}",
        ylabel="Gini coefficient",
        output_path=output_dir / f"final_layer_gini_bar_step{args.step}.png",
        dpi=args.dpi,
    )
    save_bar_plot(
        summary_df,
        column="top_60pct_share",
        title=f"Top-60% visit share comparison at step {args.step}",
        ylabel="Visit share captured by top 60% nodes",
        output_path=output_dir / f"final_layer_top60_share_bar_step{args.step}.png",
        dpi=args.dpi,
    )
    save_bar_plot(
        summary_df,
        column="coverage",
        title=f"Final-layer coverage comparison at step {args.step}",
        ylabel="Coverage ratio",
        output_path=output_dir / f"final_layer_coverage_bar_step{args.step}.png",
        dpi=args.dpi,
    )
    save_concentration_overlay(
        stats,
        output_path=output_dir / f"final_layer_concentration_overlay_step{args.step}.png",
        dpi=args.dpi,
    )

    with (output_dir / "README.txt").open("w", encoding="utf-8") as f:
        f.write(f"Compared final-layer visit distributions at step {args.step}.\n\n")
        f.write("Outputs:\n")
        f.write(f"- final_layer_gini_summary_step{args.step}.csv\n")
        f.write(f"- final_layer_gini_bar_step{args.step}.png\n")
        f.write(f"- final_layer_top60_share_bar_step{args.step}.png\n")
        f.write(f"- final_layer_coverage_bar_step{args.step}.png\n")
        f.write(f"- final_layer_concentration_overlay_step{args.step}.png\n")

    print(f"Saved outputs to: {output_dir}")


if __name__ == "__main__":
    main()
