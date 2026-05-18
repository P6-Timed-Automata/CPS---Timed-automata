"""
exp_test_noise_plot.py
======================
Reads results.json produced by exp_test_noise_run.py and renders comparison
figures. Threshold removed.

For each metric (F1, recall, precision): one two-panel figure, left subplot
is the clean-trained TAs, right subplot is the noisy-trained TAs. Each
subplot has one line per method with shaded min–max range across seeds.

Per anomaly mode: one figure per method, two-panel heatmap (clean / noisy).
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
    "sax144":  "darkorange",
    "sax48":   "orange",
    "sax24":   "gold",
    "persist": "seagreen",
}


# =============================================================================
# STATUS FILTERING (same schema as exp_53_plot)
# =============================================================================

def _is_plottable(cell):
    return cell.get("status", "ok") in ("ok", "partial")


# =============================================================================
# LOG LOADING
# =============================================================================

def load_log(log_path):
    with open(log_path) as f:
        return json.load(f)


def find_latest_log():
    base = ROOT / "Data" / "Graphs" / "TestNoise_clean_vs_noisy"
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

def _extract_method_series(log, training_condition, method, metric):
    """
    For one (training_condition, method, metric), extract per-noise-level
    (median, min, max). Skips cells where all seeds failed.
    """
    noise_levels, medians, mins, maxs = [], [], [], []
    for entry in log["results"]:
        if entry.get("training_condition") != training_condition:
            continue
        m = entry["methods"].get(method)
        if m is None or not _is_plottable(m):
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
    seen = set()
    methods = []
    for m in log["methods"]:
        if m["method"] not in seen:
            seen.add(m["method"])
            methods.append(m["method"])
    return methods


# =============================================================================
# PLOTS
# =============================================================================

def plot_metric_vs_test_noise(log, metric, ylabel, title, out_path):
    """
    Two-panel figure: left = clean-trained, right = noisy-trained.
    Each panel: per-method median line + shaded min/max band.
    """
    conditions = log["training_conditions"]
    fig, axes = plt.subplots(
        1, len(conditions),
        figsize=(7 * len(conditions), 5),
        sharey=True, sharex=True,
    )
    if len(conditions) == 1:
        axes = [axes]

    methods = _all_method_names(log)

    for ax, cond in zip(axes, conditions):
        for method in methods:
            noise_levels, medians, mins, maxs = _extract_method_series(
                log, cond, method, metric,
            )
            if not noise_levels:
                continue
            color = METHOD_COLORS.get(method, "gray")
            ax.fill_between(noise_levels, mins, maxs, color=color, alpha=0.18,
                            label=f"{method} (min-max)")
            ax.plot(noise_levels, medians, marker="o", linewidth=2, markersize=5,
                    color=color, label=f"{method} (median)")

        ax.set_title(f"{cond}-trained TAs")
        ax.set_xlabel("Test noise std (°C)")
        if ax is axes[0]:
            ax.set_ylabel(ylabel)
        ax.legend(fontsize=8, ncol=2)
        ax.grid(True, linestyle="--", alpha=0.5)
        ax.set_ylim(0, 1.05)

    n_seeds = log.get("n_seeds", "?")
    fig.suptitle(
        f"{title}  (median across {n_seeds} seeds, shaded = min–max)",
        fontsize=12,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


def plot_per_mode_heatmaps(log, out_dir):
    """
    One figure per method, two panels (clean-trained / noisy-trained).
    Rows = test noise levels, cols = anomaly modes, cells = median rejection %.
    """
    mode_names   = list(NEG_MODE_NAMES.values())
    methods      = _all_method_names(log)
    conditions   = log["training_conditions"]
    noise_levels = sorted({e["noise_std"] for e in log["results"]})

    for method in methods:
        fig, axes = plt.subplots(
            1, len(conditions),
            figsize=(6 * len(conditions), 5),
            sharey=True,
        )
        if len(conditions) == 1:
            axes = [axes]

        for ax, cond in zip(axes, conditions):
            data     = np.full((len(noise_levels), len(mode_names)), np.nan)
            statuses = ["unknown"] * len(noise_levels)

            # Index entries by noise level for this condition.
            cond_entries = {
                e["noise_std"]: e
                for e in log["results"]
                if e.get("training_condition") == cond
            }
            for i, noise in enumerate(noise_levels):
                entry = cond_entries.get(noise)
                if entry is None:
                    continue
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

            im = ax.imshow(
                np.ma.masked_invalid(data),
                vmin=0, vmax=100, cmap="RdYlGn", aspect="auto",
            )
            im.cmap.set_bad(color="#444444")

            ax.set_xticks(range(len(mode_names)))
            ax.set_xticklabels(
                [m.capitalize() for m in mode_names], rotation=20, ha="right",
            )
            ax.set_yticks(range(len(noise_levels)))
            ax.set_yticklabels([f"{n:.3f}" for n in noise_levels])
            if ax is axes[0]:
                ax.set_ylabel("Test noise std")
            ax.set_title(f"{cond}-trained")

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

        fig.colorbar(im, ax=axes, label="Rejection rate (%)",
                     fraction=0.04, pad=0.02)
        fig.suptitle(f"Per-mode rejection — {method}", fontsize=12)
        out_path = out_dir / f"per_mode_heatmap_{method}.png"
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
        help="Output folder. Defaults to same folder as log file.",
    )
    args = parser.parse_args()

    if args.log is None:
        args.log = find_latest_log()
        if args.log is None:
            print("No results.json found. Run exp_test_noise_run.py first.")
            return
        print(f"Auto-selected log: {args.log}")

    out_dir = Path(args.out) if args.out else Path(args.log).parent
    out_dir.mkdir(parents=True, exist_ok=True)

    log = load_log(args.log)

    print(f"Plotting run from: {log.get('timestamp', 'unknown')}")
    print(f"Seeds per point:   {log.get('n_seeds', '?')}")
    print(f"Output folder:     {out_dir}\n")

    print("=== Metric vs test-noise plots ===")
    plot_metric_vs_test_noise(
        log, "f1",        "F1",        "F1 vs test noise",
        out_dir / "f1_vs_test_noise.png",
        )
    plot_metric_vs_test_noise(
        log, "recall",    "Recall",    "Recall vs test noise",
        out_dir / "recall_vs_test_noise.png",
        )
    plot_metric_vs_test_noise(
        log, "precision", "Precision", "Precision vs test noise",
        out_dir / "precision_vs_test_noise.png",
        )

    print("\n=== Per-mode heatmaps ===")
    plot_per_mode_heatmaps(log, out_dir)

    print(f"\nDone. Output: {out_dir}")


if __name__ == "__main__":
    main()