"""
exp_53_plot.py
==============
Reads results.json produced by exp_53_run.py and regenerates all figures.

For each metric (F1, recall, precision), plots median across seeds with
shaded min/max range. Overlays the clean and noisy reference noise levels
and the F1 threshold.

Also produces:
  - per-mode rejection rate heatmap per method
  - threshold-crossing summary table

Run directly from PyCharm (no arguments) — auto-selects most recent run.
To plot a specific run:
    python exp_53_plot.py --log path/to/results.json
"""
import argparse
import csv
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
    "sax144":  "darkorange",
    "sax48":   "orange",
    "sax24":   "gold",
    "persist": "seagreen",
}


# =============================================================================
# STATUS FILTERING
# =============================================================================

def _is_plottable(cell):
    """
    A cell has plottable data if at least one seed succeeded.
    "ok" and "partial" both have aggregated stats; "failed" does not.
    Defaults to "ok" for backward compat with older logs.
    """
    return cell.get("status", "ok") in ("ok", "partial")


def _is_failed(cell):
    """True if all seeds failed (no aggregated data available)."""
    return cell.get("status") == "failed"



# =============================================================================
# LOG LOADING
# =============================================================================

def load_log(log_path):
    with open(log_path) as f:
        return json.load(f)


def find_latest_log():
    base = ROOT / "Data" / "Graphs" / "Metrics_noisy_sweep"

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
    """
    For one method, extract per-noise-level (median, min, max) for a metric.
    Skips cells where all seeds failed (no overall data available).

    Returns
    -------
    noise_levels : list
    medians      : list
    mins         : list
    maxs         : list
    """
    noise_levels = []
    medians = []
    mins = []
    maxs = []

    for entry in log["results"]:
        if method_name not in entry["methods"]:
            continue
        m = entry["methods"][method_name]
        if not _is_plottable(m):
            continue
        overall = m.get("overall", {})
        if metric not in overall:
            continue
        noise_levels.append(entry["noise_std"])
        medians.append(overall[metric]["median"])
        mins.append(overall[metric]["min"])
        maxs.append(overall[metric]["max"])

    return noise_levels, medians, mins, maxs


def _all_method_names(log):
    """
    Return unique method names in insertion order.
    """

    seen = set()
    methods = []

    for m in log["methods"]:
        name = m["method"]

        if name not in seen:
            seen.add(name)
            methods.append(name)

    return methods


# =============================================================================
# PLOTS
# =============================================================================

def plot_metric_vs_noise(log, metric, ylabel, title, out_path):
    """
    Line plot of metric (median) vs. noise level, with shaded min/max range
    per method. Overlays F1 threshold (if F1) and clean/noisy reference lines.
    """

    fig, ax = plt.subplots(figsize=(10, 5))

    methods = _all_method_names(log)

    for method in methods:

        noise_levels, medians, mins, maxs = _extract_method_series(
            log,
            method,
            metric,
        )

        if not noise_levels:
            continue

        color = METHOD_COLORS.get(method, "gray")

        ax.fill_between(
            noise_levels,
            mins,
            maxs,
            color=color,
            alpha=0.18,
            label=f"{method} (min-max)",
        )

        ax.plot(
            noise_levels,
            medians,
            marker="o",
            linewidth=2,
            markersize=5,
            color=color,
            label=f"{method} (median)",
        )

    # F1 threshold line
    threshold = log.get("f1_threshold")

    if metric == "f1" and threshold is not None:
        ax.axhline(
            threshold,
            color="red",
            linewidth=1.2,
            linestyle="--",
            label=f"Threshold = {threshold}",
        )

    # Clean / noisy reference lines
    for level_key, label_text in [
        ("clean_noise", "clean"),
        ("noisy_noise", "noisy"),
    ]:

        level = log.get(level_key)

        if level is not None:

            ax.axvline(
                level,
                color="gray",
                linewidth=1.0,
                linestyle=":",
                alpha=0.7,
            )

            ax.text(
                level,
                0.02,
                label_text,
                rotation=90,
                fontsize=7,
                color="gray",
                va="bottom",
                ha="right",
            )

    ax.set_xlabel("Training noise std (°C)")
    ax.set_ylabel(ylabel)

    n_seeds = log.get("n_seeds", "?")

    ax.set_title(
        f"{title}  "
        f"(median across {n_seeds} seeds, shaded = min-max)"
    )

    ax.legend(fontsize=8, ncol=2)
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.set_ylim(0, 1.05)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")

    plt.close(fig)

    print(f"  Saved: {out_path}")


def plot_per_mode_heatmap_per_method(log, out_dir):
    mode_names = list(NEG_MODE_NAMES.values())
    methods = _all_method_names(log)
    noise_levels = [e["noise_std"] for e in log["results"]]

    for method in methods:
        data = np.full((len(noise_levels), len(mode_names)), np.nan)   # NaN = no data
        statuses = ["unknown"] * len(noise_levels)

        for i, entry in enumerate(log["results"]):
            method_entry = entry["methods"].get(method)
            if not method_entry:
                continue
            statuses[i] = method_entry.get("status", "ok")
            if not _is_plottable(method_entry):
                continue
            per_mode = method_entry.get("per_mode", {})
            for j, mode in enumerate(mode_names):
                pm = per_mode.get(mode)
                if pm:
                    data[i, j] = pm["rejection_median"]

        fig, ax = plt.subplots(figsize=(8, 5))
        # masked_invalid: NaN cells render as the cmap's "bad" color
        im = ax.imshow(
            np.ma.masked_invalid(data),
            vmin=0, vmax=100, cmap="RdYlGn", aspect="auto",
        )
        # Failed cells (NaN) get a distinct background
        im.cmap.set_bad(color="#444444")

        ax.set_xticks(range(len(mode_names)))
        ax.set_xticklabels([m.capitalize() for m in mode_names], rotation=20, ha="right")
        ax.set_yticks(range(len(noise_levels)))
        ax.set_yticklabels([f"{n:.3f}" for n in noise_levels])
        ax.set_ylabel("Training noise std")
        ax.set_title(f"Per-mode rejection rate (%) — {method}")

        for i in range(len(noise_levels)):
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
# THRESHOLD ANALYSIS
# =============================================================================

def _find_threshold_crossing(noise_levels, medians, threshold):
    """
    Find first noise level where F1 median drops below threshold.
    Also returns last noise level above threshold.
    """

    crossing_at = None
    last_above = None

    for n, f in zip(noise_levels, medians):

        if f >= threshold:
            last_above = n

        elif crossing_at is None:
            crossing_at = n

    return crossing_at, last_above


def write_threshold_summary(log, out_dir):
    """
    Compute and save threshold-crossing analysis (CSV + console).
    """

    threshold = log.get("f1_threshold")

    if threshold is None:
        return

    methods = _all_method_names(log)
    rows = []

    print(f"\nNoise tolerance (median F1 drops below {threshold}):")

    for method in methods:

        noise_levels, medians, mins, maxs = _extract_method_series(
            log,
            method,
            "f1",
        )

        # Skip methods with no successful runs
        if not noise_levels:
            continue

        crossing, last_above = _find_threshold_crossing(
            noise_levels,
            medians,
            threshold,
        )

        if crossing is None:
            crossing_str = "stays above threshold"
            last_above_str = f"{noise_levels[-1]:.3f}"

        else:
            crossing_str = f"{crossing:.3f}"
            last_above_str = (
                f"{last_above:.3f}"
                if last_above is not None
                else "n/a"
            )

        f1_at_clean = next(
            (
                m for n, m in zip(noise_levels, medians)
                if abs(n - log.get("clean_noise", -1)) < 1e-9
            ),
            None,
        )

        f1_at_noisy = next(
            (
                m for n, m in zip(noise_levels, medians)
                if abs(n - log.get("noisy_noise", -1)) < 1e-9
            ),
            None,
        )

        rows.append({
            "method":          method,
            "f1_at_clean":     f"{f1_at_clean:.3f}" if f1_at_clean is not None else "n/a",
            "f1_at_noisy":     f"{f1_at_noisy:.3f}" if f1_at_noisy is not None else "n/a",
            "drops_below_at":  crossing_str,
            "last_above":      last_above_str,
        })

        print(
            f"  {method:8s} | "
            f"F1@clean={rows[-1]['f1_at_clean']} | "
            f"F1@noisy={rows[-1]['f1_at_noisy']} | "
            f"drops at {crossing_str} "
            f"(last above: {last_above_str})"
        )

    # Save CSV
    if rows:

        csv_path = out_dir / "threshold_summary.csv"

        with open(csv_path, "w", newline="") as f:

            writer = csv.DictWriter(
                f,
                fieldnames=rows[0].keys(),
            )

            writer.writeheader()
            writer.writerows(rows)

        print(f"  Saved: {csv_path}")


# =============================================================================
# MAIN
# =============================================================================

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--log",
        default=None,
        help="Path to results.json. Defaults to most recent run.",
    )

    parser.add_argument(
        "--out",
        default=None,
        help="Output folder. Defaults to same folder as log file.",
    )

    args = parser.parse_args()

    if args.log is None:

        args.log = find_latest_log()

        if args.log is None:
            print("No results.json found. Run exp_53_run.py first.")
            return

        print(f"Auto-selected log: {args.log}")

    out_dir = Path(args.out) if args.out else Path(args.log).parent

    out_dir.mkdir(parents=True, exist_ok=True)

    log = load_log(args.log)

    print(f"Plotting run from: {log.get('timestamp', 'unknown')}")
    print(f"Seeds per point:   {log.get('n_seeds', '?')}")
    print(f"Output folder:     {out_dir}\n")

    # -------------------------------------------------------------------------
    # Metric-vs-noise plots
    # -------------------------------------------------------------------------

    print("=== Plots ===")

    plot_metric_vs_noise(
        log,
        "f1",
        "F1",
        "F1 vs training noise",
        out_dir / "f1_vs_noise.png",
        )

    plot_metric_vs_noise(
        log,
        "recall",
        "Recall",
        "Recall vs training noise",
        out_dir / "recall_vs_noise.png",
        )

    plot_metric_vs_noise(
        log,
        "precision",
        "Precision",
        "Precision vs training noise",
        out_dir / "precision_vs_noise.png",
        )

    # -------------------------------------------------------------------------
    # Per-mode heatmaps
    # -------------------------------------------------------------------------

    plot_per_mode_heatmap_per_method(log, out_dir)

    # -------------------------------------------------------------------------
    # Threshold analysis
    # -------------------------------------------------------------------------

    write_threshold_summary(log, out_dir)

    print(f"\nDone. Output: {out_dir}")


if __name__ == "__main__":
    main()