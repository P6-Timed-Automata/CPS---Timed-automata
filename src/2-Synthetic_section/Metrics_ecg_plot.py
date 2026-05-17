"""
Metrics_ecg_plot.py
===================
Reads results.json produced by exp_55_ecg.py (or whatever you rename it to)
and regenerates figures + tables.

Handles both negative-trace modes:
  - "folder"    : single "anomaly" mode (real labeled ECG anomalies)
  - "synthetic" : four modes (spikes/shifted/stuck/offset) — note that these
                  are temperature-shaped negatives applied to ECG; the domain
                  mismatch is the issue flagged in your project notes.

Run directly (auto-selects most recent log):
    python Metrics_ecg_plot.py
Or pin to a specific run:
    python Metrics_ecg_plot.py --log path/to/results.json
"""

import argparse
import csv
import json
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# Project root is three levels up.
ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from Generators import NEG_MODE_NAMES


METHOD_COLORS = {
    "naive":   "steelblue",
    "sax":     "darkorange",
    "persist": "seagreen",
}


# =============================================================================
# STATUS FILTERING
# =============================================================================

def _is_ok(r):
    """Default to 'ok' for old logs without a status field."""
    return r.get("status", "ok") == "ok"


def _get_mode_names(ok_results):
    """
    Derive per-mode keys from the loaded results. Preserves NEG_MODE_NAMES
    order if those keys appear, then appends extras (e.g. 'anomaly') sorted.
    Lets the same plotter handle synthetic-4-mode and folder-1-mode runs.
    """
    modes = set()
    for r in ok_results:
        modes.update(r.get("per_mode", {}).keys())
    canonical = list(NEG_MODE_NAMES.values())
    ordered = [m for m in canonical if m in modes]
    extras  = sorted([m for m in modes if m not in canonical])
    return ordered + extras


# =============================================================================
# LOG LOADING
# =============================================================================

def load_log(log_path):
    with open(log_path) as f:
        return json.load(f)


def find_latest_log():
    base = ROOT / "Data" / "Graphs" / "exp_55"
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
    """All methods, FAILED rows included with error info."""
    csv_path = out_dir / "table_overall.csv"
    headers = ["method", "params", "status", "precision", "recall", "f1",
               "TP", "FP", "FN", "TN", "n_states", "n_edges", "error"]
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        for r in results:
            param_str = ", ".join(f"{k}={v}" for k, v in r["params"].items())
            if _is_ok(r):
                ov = r["overall"]
                writer.writerow([
                    r["method"], param_str, "ok",
                    f"{ov['precision']:.3f}", f"{ov['recall']:.3f}",
                    f"{ov['f1']:.3f}",
                    ov.get("TP", "-"), ov.get("FP", "-"),
                    ov.get("FN", "-"), ov.get("TN", "-"),
                    r["n_states"], r["n_edges"], "",
                ])
            else:
                writer.writerow([
                    r["method"], param_str, "failed",
                    "-", "-", "-", "-", "-", "-", "-", "-", "-",
                    f"{r.get('error_type', '?')}: {r.get('error_msg', '')[:200]}",
                ])
    print(f"  Saved: {csv_path}")


def save_per_mode_table(ok_results, mode_names, out_dir):
    csv_path = out_dir / "table_per_mode.csv"
    headers = ["method"] + [m.capitalize() for m in mode_names]
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        for r in ok_results:
            row = [r["method"]]
            for mode in mode_names:
                pct = r["per_mode"].get(mode, {}).get("rejection", 0.0)
                row.append(f"{pct:.1f}%")
            writer.writerow(row)
    print(f"  Saved: {csv_path}")


# =============================================================================
# PLOTS
# =============================================================================

def plot_metric_bars(ok_results, metric, ylabel, out_path):
    methods = [r["method"] for r in ok_results]
    values  = [r["overall"][metric] for r in ok_results]

    fig, ax = plt.subplots(figsize=(7, 5))
    bars = ax.bar(
        methods, values,
        color=[METHOD_COLORS.get(m, "gray") for m in methods],
        alpha=0.85,
    )
    for bar, v in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.02, f"{v:.2f}",
            ha="center", va="bottom", fontsize=10,
            )
    ax.set_ylim(0, 1.2)
    ax.set_ylabel(ylabel)
    ax.set_title(f"{ylabel} — ECG data")
    ax.grid(True, linestyle="--", alpha=0.4, axis="y")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


def plot_per_mode_heatmap(ok_results, mode_names, out_path):
    methods = [r["method"] for r in ok_results]
    data    = np.zeros((len(methods), len(mode_names)))

    for i, r in enumerate(ok_results):
        for j, mode in enumerate(mode_names):
            data[i, j] = r["per_mode"].get(mode, {}).get("rejection", 0.0)

    # Heatmap shrinks gracefully for 1-mode "anomaly" runs.
    fig_width = max(6, len(mode_names) * 1.5 + 2)
    fig, ax = plt.subplots(figsize=(fig_width, max(3, len(methods) * 0.8 + 2)))
    im = ax.imshow(data, vmin=0, vmax=100, cmap="RdYlGn", aspect="auto")
    ax.set_xticks(range(len(mode_names)))
    ax.set_xticklabels([m.capitalize() for m in mode_names], rotation=20, ha="right")
    ax.set_yticks(range(len(methods)))
    ax.set_yticklabels(methods)
    ax.set_title("Rejection rate (%) per anomaly mode — ECG")

    for i in range(len(methods)):
        for j in range(len(mode_names)):
            ax.text(
                j, i, f"{data[i, j]:.0f}",
                ha="center", va="center", fontsize=11,
                color="black" if 20 < data[i, j] < 80 else "white",
            )
    plt.colorbar(im, ax=ax, label="%")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


def plot_state_edge_counts(ok_results, out_path):
    methods = [r["method"] for r in ok_results]
    states  = [r["n_states"] for r in ok_results]
    edges   = [r["n_edges"]  for r in ok_results]

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
    ax.set_title("TA size per method — ECG")
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
        help="Output folder. Defaults to same folder as log file.",
    )
    args = parser.parse_args()

    if args.log is None:
        args.log = find_latest_log()
        if args.log is None:
            print("No results.json found. Run the ECG runner first.")
            return
        print(f"Auto-selected log: {args.log}")

    out_dir = Path(args.out) if args.out else Path(args.log).parent
    out_dir.mkdir(parents=True, exist_ok=True)

    log = load_log(args.log)
    results        = log["results"]
    ok_results     = [r for r in results if _is_ok(r)]
    failed_results = [r for r in results if not _is_ok(r)]
    mode_names     = _get_mode_names(ok_results)

    print(f"Plotting run from: {log.get('timestamp', 'unknown')}")
    print(f"Data folder:       {log.get('data_folder', 'unknown')}")
    print(f"Methods: {len(ok_results)} ok, {len(failed_results)} failed")
    print(f"Modes:   {mode_names}")
    print(f"Output folder:     {out_dir}\n")

    if failed_results:
        print("Failed methods (will appear in table_overall.csv only):")
        for r in failed_results:
            print(f"  {r['method']:8s} : {r.get('error_type', '?')} — "
                  f"{r.get('error_msg', '')[:100]}")
        print()

    # ---- Tables ----
    print("=== Tables ===")
    save_overall_table(results, out_dir)
    if ok_results and mode_names:
        save_per_mode_table(ok_results, mode_names, out_dir)
    elif not ok_results:
        print("  No successful methods — skipping per-mode table.")

    # ---- Plots ----
    if not ok_results:
        print("\nNo successful methods to plot.")
        print(f"\nDone. Output: {out_dir}")
        return

    print("\n=== Plots ===")
    plot_metric_bars(ok_results, "precision", "Precision",
                     out_dir / "comparison_precision.png")
    plot_metric_bars(ok_results, "recall",    "Recall",
                     out_dir / "comparison_recall.png")
    plot_metric_bars(ok_results, "f1",        "F1",
                     out_dir / "comparison_f1.png")
    if mode_names:
        plot_per_mode_heatmap(ok_results, mode_names,
                              out_dir / "per_mode_heatmap.png")
    plot_state_edge_counts(ok_results, out_dir / "state_edge_counts.png")

    print(f"\nDone. Output: {out_dir}")


if __name__ == "__main__":
    main()