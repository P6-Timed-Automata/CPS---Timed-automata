"""
Metrics_bins_plot.py
====================
Reads results.json from Metrics_bins_run.py and renders:

  f1_vs_bins.png         — F1 vs bin count (median + min/max band)
  recall_vs_bins.png     — same for recall
  precision_vs_bins.png  — same for precision
  per_mode_heatmap_<method>.png — rows = bin levels, cols = anomaly modes
  n_states_vs_bins.png   — TA size vs bin count (diagnostic)

Auto-selects the latest log if --log isn't given.
"""

import argparse
import json
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from Generators import NEG_MODE_NAMES


METHOD_COLORS = {
    "naive":   "steelblue",
    "sax":     "darkorange",
    "persist": "seagreen",
}


def _is_plottable(cell):
    return cell.get("status", "ok") in ("ok", "partial")


# =============================================================================
# LOG LOADING
# =============================================================================

def load_log(log_path):
    with open(log_path) as f:
        return json.load(f)


def find_latest_log():
    base = ROOT / "Data" / "Graphs" / "Metrics_bins"
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
# DATA RESHAPING
# =============================================================================

def _extract_method_series(log, method_name, metric):
    """For one method, return (bin_levels, medians, mins, maxs) for a metric."""
    bins, medians, mins, maxs = [], [], [], []
    for entry in log["results"]:
        cell = entry["methods"].get(method_name)
        if cell is None or not _is_plottable(cell):
            continue
        overall = cell.get("overall", {})
        if metric not in overall:
            continue
        bins.append(entry["bins"])
        medians.append(overall[metric]["median"])
        mins.append(overall[metric]["min"])
        maxs.append(overall[metric]["max"])
    return bins, medians, mins, maxs


def _extract_n_states_series(log, method_name):
    """For one method, return (bin_levels, median_states, min, max)."""
    bins, medians, mins, maxs = [], [], [], []
    for entry in log["results"]:
        cell = entry["methods"].get(method_name)
        if cell is None or not _is_plottable(cell):
            continue
        bins.append(entry["bins"])
        medians.append(cell.get("n_states_median", 0))
        mins.append(cell.get("n_states_min", 0))
        maxs.append(cell.get("n_states_max", 0))
    return bins, medians, mins, maxs


def _all_method_names(log):
    seen, methods = set(), []
    for m in log["methods"]:
        if m["method"] not in seen:
            seen.add(m["method"])
            methods.append(m["method"])
    return methods


# =============================================================================
# PLOTS
# =============================================================================

def plot_metric_vs_bins(log, metric, ylabel, title, out_path):
    fig, ax = plt.subplots(figsize=(10, 5))
    methods = _all_method_names(log)
    for method in methods:
        bins, medians, mins, maxs = _extract_method_series(log, method, metric)
        if not bins:
            continue
        color = METHOD_COLORS.get(method, "gray")
        ax.fill_between(bins, mins, maxs, color=color, alpha=0.18,
                        label=f"{method} (min-max)")
        ax.plot(bins, medians, marker="o", linewidth=2, markersize=5,
                color=color, label=f"{method} (median)")

    ax.set_xlabel("Number of bins (alphabet size)")
    ax.set_ylabel(ylabel)
    n_seeds = log.get("n_seeds", "?")
    ax.set_title(f"{title}  (median across {n_seeds} seeds, shaded = min-max)")
    ax.legend(fontsize=8, ncol=2)
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.set_ylim(0, 1.05)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


def plot_n_states_vs_bins(log, out_path):
    """Diagnostic: how does TA size grow with bin count?"""
    fig, ax = plt.subplots(figsize=(10, 5))
    methods = _all_method_names(log)
    for method in methods:
        bins, medians, mins, maxs = _extract_n_states_series(log, method)
        if not bins:
            continue
        color = METHOD_COLORS.get(method, "gray")
        ax.fill_between(bins, mins, maxs, color=color, alpha=0.18)
        ax.plot(bins, medians, marker="o", linewidth=2, markersize=5,
                color=color, label=method)
    ax.set_xlabel("Number of bins (alphabet size)")
    ax.set_ylabel("TA states (median across seeds)")
    ax.set_title("TA size vs bin count")
    ax.legend(fontsize=9)
    ax.grid(True, linestyle="--", alpha=0.5)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


def plot_per_mode_heatmap_per_method(log, out_dir):
    mode_names = list(NEG_MODE_NAMES.values())
    methods    = _all_method_names(log)
    bin_levels = [e["bins"] for e in log["results"]]

    for method in methods:
        data     = np.full((len(bin_levels), len(mode_names)), np.nan)
        statuses = ["unknown"] * len(bin_levels)

        for i, entry in enumerate(log["results"]):
            cell = entry["methods"].get(method)
            if cell is None:
                continue
            statuses[i] = cell.get("status", "ok")
            if not _is_plottable(cell):
                continue
            per_mode = cell.get("per_mode", {})
            for j, mode in enumerate(mode_names):
                pm = per_mode.get(mode)
                if pm:
                    data[i, j] = pm["rejection_median"]

        fig, ax = plt.subplots(figsize=(8, 5))
        im = ax.imshow(np.ma.masked_invalid(data),
                       vmin=0, vmax=100, cmap="RdYlGn", aspect="auto")
        im.cmap.set_bad(color="#444444")

        ax.set_xticks(range(len(mode_names)))
        ax.set_xticklabels([m.capitalize() for m in mode_names],
                           rotation=20, ha="right")
        ax.set_yticks(range(len(bin_levels)))
        ax.set_yticklabels([str(b) for b in bin_levels])
        ax.set_ylabel("Number of bins")
        ax.set_title(f"Per-mode rejection rate (%) — {method}")

        for i in range(len(bin_levels)):
            for j in range(len(mode_names)):
                v = data[i, j]
                if np.isnan(v):
                    label = "FAILED" if statuses[i] == "failed" else "-"
                    ax.text(j, i, label, ha="center", va="center",
                            fontsize=8, color="white", fontweight="bold")
                else:
                    ax.text(j, i, f"{v:.0f}", ha="center", va="center",
                            fontsize=9,
                            color="black" if 20 < v < 80 else "white")
        plt.colorbar(im, ax=ax, label="%")
        fig.tight_layout()
        out_path = out_dir / f"per_mode_heatmap_{method}.png"
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved: {out_path}")


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--log", default=None,
                        help="Path to results.json. Defaults to most recent run.")
    parser.add_argument("--out", default=None,
                        help="Output folder. Defaults to same folder as log file.")
    args = parser.parse_args()

    if args.log is None:
        args.log = find_latest_log()
        if args.log is None:
            print("No results.json found. Run Metrics_bins_run.py first.")
            return
        print(f"Auto-selected log: {args.log}")

    out_dir = Path(args.out) if args.out else Path(args.log).parent
    out_dir.mkdir(parents=True, exist_ok=True)

    log = load_log(args.log)

    print(f"Plotting run from: {log.get('timestamp', 'unknown')}")
    print(f"Seeds per cell:    {log.get('n_seeds', '?')}")
    print(f"SAX w (fixed):     {log.get('sax_w', '?')}")
    print(f"Output folder:     {out_dir}\n")

    print("=== Metric plots ===")
    plot_metric_vs_bins(log, "f1",        "F1",        "F1 vs bin count",
                        out_dir / "f1_vs_bins.png")
    plot_metric_vs_bins(log, "recall",    "Recall",    "Recall vs bin count",
                        out_dir / "recall_vs_bins.png")
    plot_metric_vs_bins(log, "precision", "Precision", "Precision vs bin count",
                        out_dir / "precision_vs_bins.png")

    print("\n=== Diagnostic plot ===")
    plot_n_states_vs_bins(log, out_dir / "n_states_vs_bins.png")

    print("\n=== Per-mode heatmaps ===")
    plot_per_mode_heatmap_per_method(log, out_dir)

    print(f"\nDone. Output: {out_dir}")


if __name__ == "__main__":
    main()