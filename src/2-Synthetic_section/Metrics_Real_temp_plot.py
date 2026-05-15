"""
exp_54_plot.py
==============
Reads results.json produced by exp_54_run.py and regenerates all figures
and tables. Output goes to the same folder as the results file by default.

Run directly (no arguments) — auto-selects the most recent run.
To plot a specific run:
    python exp_54_plot.py --log path/to/results.json
"""

import argparse
import csv
import json
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from Generators import NEG_MODE_NAMES


METHOD_COLORS = {
    "naive":   "steelblue",
    "sax":     "darkorange",
    "persist": "seagreen",
}


# =============================================================================
# LOG LOADING
# =============================================================================

def load_log(log_path):
    with open(log_path) as f:
        return json.load(f)


def find_latest_log():
    base = ROOT / "Data" / "Graphs" / "exp_54"
    if not base.is_dir():
        return None
    candidates = []
    for entry in os.scandir(base):
        if entry.is_dir():
            log = Path(entry.path) / "results.json"
            if log.is_file():
                candidates.append((entry.name, str(log)))
    if not candidates:
        return None
    candidates.sort(key=lambda x: x[0])
    return candidates[-1][1]


# =============================================================================
# TABLES
# =============================================================================

def save_overall_table(results, out_dir):
    csv_path = out_dir / "table_overall.csv"
    headers = ["method", "params", "precision", "recall", "f1",
               "TP", "FP", "FN", "TN", "n_states", "n_edges"]
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        for r in results:
            ov = r["overall"]
            param_str = ", ".join(f"{k}={v}" for k, v in r["params"].items())
            writer.writerow([
                r["method"], param_str,
                f"{ov['precision']:.3f}", f"{ov['recall']:.3f}", f"{ov['f1']:.3f}",
                ov.get("TP", "-"), ov.get("FP", "-"),
                ov.get("FN", "-"), ov.get("TN", "-"),
                r["n_states"], r["n_edges"],
            ])
    print(f"  Saved: {csv_path}")


def save_per_mode_table(results, out_dir):
    mode_names = list(NEG_MODE_NAMES.values())
    csv_path = out_dir / "table_per_mode.csv"
    headers = ["method"] + [m.capitalize() for m in mode_names]
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        for r in results:
            row = [r["method"]]
            for mode in mode_names:
                pct = r["per_mode"].get(mode, {}).get("rejection", 0.0)
                row.append(f"{pct:.1f}%")
            writer.writerow(row)
    print(f"  Saved: {csv_path}")


# =============================================================================
# PLOTS
# =============================================================================

def plot_metric_bars(results, metric, ylabel, out_path):
    """One bar per method for the given metric."""
    methods = [r["method"] for r in results]
    values = [r["overall"][metric] for r in results]

    fig, ax = plt.subplots(figsize=(7, 5))
    bars = ax.bar(methods, values, color=[METHOD_COLORS.get(m, "gray") for m in methods],
                  alpha=0.85)
    for bar, v in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.02, f"{v:.2f}",
                ha="center", va="bottom", fontsize=10)
    ax.set_ylim(0, 1.2)
    ax.set_ylabel(ylabel)
    ax.set_title(f"{ylabel} — real temperature data")
    ax.grid(True, linestyle="--", alpha=0.4, axis="y")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


def plot_per_mode_heatmap(results, out_path):
    mode_names = list(NEG_MODE_NAMES.values())
    methods = [r["method"] for r in results]
    data = np.zeros((len(methods), len(mode_names)))

    for i, r in enumerate(results):
        for j, mode in enumerate(mode_names):
            data[i, j] = r["per_mode"].get(mode, {}).get("rejection", 0.0)

    fig, ax = plt.subplots(figsize=(8, 4))
    im = ax.imshow(data, vmin=0, vmax=100, cmap="RdYlGn", aspect="auto")
    ax.set_xticks(range(len(mode_names)))
    ax.set_xticklabels([m.capitalize() for m in mode_names])
    ax.set_yticks(range(len(methods)))
    ax.set_yticklabels(methods)
    ax.set_title("Rejection rate (%) per anomaly mode — real temperature data")

    for i in range(len(methods)):
        for j in range(len(mode_names)):
            ax.text(j, i, f"{data[i, j]:.0f}",
                    ha="center", va="center", fontsize=11,
                    color="black" if 20 < data[i, j] < 80 else "white")
    plt.colorbar(im, ax=ax, label="%")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


def plot_state_edge_counts(results, out_path):
    """Bar chart of TA state and edge counts per method."""
    methods = [r["method"] for r in results]
    states = [r["n_states"] for r in results]
    edges = [r["n_edges"] for r in results]

    x = np.arange(len(methods))
    width = 0.35

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(x - width / 2, states, width, label="States",
           color="steelblue", alpha=0.85)
    ax.bar(x + width / 2, edges, width, label="Edges",
           color="darkorange", alpha=0.85)

    for i, (s, e) in enumerate(zip(states, edges)):
        ax.text(i - width / 2, s + 1, str(s), ha="center", fontsize=9)
        ax.text(i + width / 2, e + 1, str(e), ha="center", fontsize=9)

    ax.set_xticks(x)
    ax.set_xticklabels(methods)
    ax.set_ylabel("Count")
    ax.set_title("TA size per method — real temperature data")
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.4, axis="y")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--log", default=None,
        help="Path to results.json. Defaults to most recent run.",
    )
    parser.add_argument(
        "--out", default=None,
        help="Output folder. Defaults to the same folder as the log file.",
    )
    args = parser.parse_args()

    if args.log is None:
        args.log = find_latest_log()
        if args.log is None:
            print("No results.json found. Run exp_54_run.py first.")
            return
        print(f"Auto-selected log: {args.log}")

    out_dir = Path(args.out) if args.out else Path(args.log).parent
    out_dir.mkdir(parents=True, exist_ok=True)

    log = load_log(args.log)
    results = log["results"]

    print(f"Plotting run from: {log.get('timestamp', 'unknown')}")
    print(f"Negative mode:     {log.get('negative_mode', 'unknown')}")
    print(f"Total methods:     {len(results)}")
    print(f"Output folder:     {out_dir}\n")

    # ----- Tables -----
    print("=== Tables ===")
    save_overall_table(results, out_dir)
    save_per_mode_table(results, out_dir)

    # ----- Plots -----
    print("\n=== Plots ===")
    plot_metric_bars(results, "precision", "Precision",
                     out_dir / "comparison_precision.png")
    plot_metric_bars(results, "recall", "Recall",
                     out_dir / "comparison_recall.png")
    plot_metric_bars(results, "f1", "F1",
                     out_dir / "comparison_f1.png")
    plot_per_mode_heatmap(results, out_dir / "per_mode_heatmap.png")
    plot_state_edge_counts(results, out_dir / "state_edge_counts.png")

    print(f"\nDone. Output: {out_dir}")


if __name__ == "__main__":
    main()