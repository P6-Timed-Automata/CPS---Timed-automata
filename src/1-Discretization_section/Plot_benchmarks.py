"""
plot_benchmark.py
=================
Read benchmark_log.json produced by run_benchmark.py and regenerate all figures.
Output is written to the same timestamped folder as the log file by default,
so figures always end up alongside the data that produced them.

Usage:
    python plot_benchmark.py --log path/to/TA_Benchmark/2026-05-12_14-30-00/benchmark_log.json
    python plot_benchmark.py --log path/to/benchmark_log.json --out path/to/custom/output
"""

import argparse
import json
import os

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_log(log_path: str) -> dict:
    with open(log_path) as f:
        return json.load(f)


def _header_color(ax, tab, n_cols, color="#FFD700"):
    for j in range(n_cols):
        tab[(0, j)].set_facecolor(color)


# ---------------------------------------------------------------------------
# Plot 1: signal + residual
# ---------------------------------------------------------------------------

def plot_signal(method_name: str, t_raw, v_raw, results: list, output_folder: str):
    best   = results[0]
    t_d    = np.array(best["plot_t_d"])
    v_d    = np.array(best["plot_v_d"])
    resids = np.array(best["plot_resids"])

    fig = plt.figure(figsize=(12, 6))
    gs  = GridSpec(2, 1, height_ratios=[3, 1], hspace=0.25)

    ax_top = fig.add_subplot(gs[0])
    ax_bot = fig.add_subplot(gs[1], sharex=ax_top)

    ax_top.plot(np.array(t_raw) / 3600, v_raw, alpha=0.5, label="Raw")
    ax_top.step(t_d / 3600, v_d, where="post", label=f"Discretized (best: {best['label']})")
    ax_top.set_title(f"{method_name} — best variant: {best['label']}")
    ax_top.set_ylabel("Temperature")
    ax_top.legend()

    ax_bot.axhline(0, color="black", lw=1, ls="--")
    ax_bot.plot(t_d / 3600, resids)
    ax_bot.set_xlabel("Time (hours)")
    ax_bot.set_ylabel("Residual")

    out = os.path.join(output_folder, f"{method_name}_signal.png")
    plt.savefig(out, bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"  Saved: {out}")


# ---------------------------------------------------------------------------
# Plot 2: structure table
# ---------------------------------------------------------------------------

def plot_structure_table(method_name: str, results: list, output_folder: str):
    is_persist = (method_name == "Persist")

    rows = []
    for r in results:
        n_states = f"{r['n_states_mean']:.1f}±{r['n_states_std']:.1f}"
        n_edges  = f"{r['n_edges_mean']:.1f}±{r['n_edges_std']:.1f}"
        rmse     = f"{r['rmse_mean']:.2f}±{r['rmse_std']:.2f}"
        t        = f"{r['time_mean']:.2f}±{r['time_std']:.2f}s"

        if is_persist:
            bin_vals = [pt["actual_bins"] for pt in r["per_trace"]]
            dominant = max(set(bin_vals), key=bin_vals.count)
            bins_display = str(dominant) + ("*" if len(set(bin_vals)) > 1 else "")
            rows.append([r["label"], bins_display, n_states, n_edges, rmse, t])
        else:
            rows.append([r["label"], n_states, n_edges, rmse, t])

    cols = (
        ["Parameter", "Bins/trace", "States", "Edges", "RMSE", "Time"]
        if is_persist else
        ["Parameter", "States", "Edges", "RMSE", "Time"]
    )

    n_cols  = len(cols)
    fig_h   = max(3.0, 0.55 * len(rows) + 1.2)
    fig, ax = plt.subplots(figsize=(max(10, 2.2 * n_cols), fig_h))
    ax.axis("off")
    ax.set_title(f"{method_name} — TA structure & accuracy", fontsize=12, pad=10)

    tab = ax.table(cellText=rows, colLabels=cols, cellLoc="center", loc="center")
    tab.auto_set_font_size(False)
    tab.set_fontsize(9.5)
    tab.scale(1.2, 2.0)
    _header_color(ax, tab, n_cols)

    # Highlight best RMSE row
    for j in range(n_cols):
        tab[(1, j)].set_facecolor("#CCFFCC")

    out = os.path.join(output_folder, f"{method_name}_structure_table.png")
    plt.savefig(out, bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"  Saved: {out}")


# ---------------------------------------------------------------------------
# Plot 3: combined (original layout)
# ---------------------------------------------------------------------------

def plot_combined(method_name: str, t_raw, v_raw, results: list, output_folder: str):
    is_persist = (method_name == "Persist")
    best = results[0]

    t_d    = np.array(best["plot_t_d"])
    v_d    = np.array(best["plot_v_d"])
    resids = np.array(best["plot_resids"])

    fig = plt.figure(figsize=(18, 8))
    gs  = GridSpec(2, 2, width_ratios=[4.5, 2.8],
                   height_ratios=[3, 1], hspace=0.28, wspace=0.18)

    ax_top   = fig.add_subplot(gs[0, 0])
    ax_bot   = fig.add_subplot(gs[1, 0], sharex=ax_top)
    ax_table = fig.add_subplot(gs[:, 1])
    ax_table.axis("off")

    ax_top.plot(np.array(t_raw) / 3600, v_raw, alpha=0.5, label="Raw")
    ax_top.step(t_d / 3600, v_d, where="post", label="Discretized (best)")
    ax_top.set_title(f"{method_name} — best: {best['label']}")
    ax_top.legend()

    ax_bot.axhline(0, color="black", lw=1, ls="--")
    ax_bot.plot(t_d / 3600, resids)
    ax_bot.set_xlabel("Time (hours)")
    ax_bot.set_ylabel("Residual")

    if is_persist:
        def _bins_display(r):
            bin_vals = [pt["actual_bins"] for pt in r["per_trace"]]
            dominant = max(set(bin_vals), key=bin_vals.count)
            return str(dominant) + ("*" if len(set(bin_vals)) > 1 else "")

        table_data = [
            [
                r["label"],
                _bins_display(r),
                f"{r['rmse_mean']:.2f}±{r['rmse_std']:.2f}",
                f"{r['time_mean']:.2f}±{r['time_std']:.2f}s",
            ]
            for r in results
        ]
        cols = ["Parameter", "Bins/trace", "RMSE", "Time"]
    else:
        table_data = [
            [
                r["label"],
                f"{r['rmse_mean']:.2f}±{r['rmse_std']:.2f}",
                f"{r['time_mean']:.2f}±{r['time_std']:.2f}s",
            ]
            for r in results
        ]
        cols = ["Parameter", "RMSE", "Time"]

    tab = ax_table.table(
        cellText=table_data, colLabels=cols, cellLoc="center", loc="center"
    )
    tab.auto_set_font_size(False)
    tab.set_fontsize(9.5)
    tab.scale(1.2, 2.3)
    _header_color(ax_table, tab, len(cols))

    out = os.path.join(output_folder, f"{method_name}_TA_Benchmark.png")
    plt.savefig(out, bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"  Saved: {out}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--log",
        required=True,
        help="Path to benchmark_log.json produced by run_benchmark.py",
    )
    parser.add_argument(
        "--out",
        default=None,
        help="Output folder for figures. Defaults to the same folder as the log file.",
    )
    args = parser.parse_args()

    # Default output folder = same directory as the log file
    out_dir = args.out if args.out else os.path.dirname(os.path.abspath(args.log))
    os.makedirs(out_dir, exist_ok=True)

    log   = load_log(args.log)
    t_raw = log["t_raw"]
    v_raw = log["v_raw"]

    ts = log.get("timestamp", "unknown")
    print(f"Plotting run from: {ts}")
    print(f"Output folder:     {out_dir}")

    for method_name, results in log["methods"].items():
        print(f"\n=== {method_name} ===")
        plot_combined(method_name, t_raw, v_raw, results, out_dir)
        plot_signal(method_name, t_raw, v_raw, results, out_dir)
        plot_structure_table(method_name, results, out_dir)

    print("\nDone.")


if __name__ == "__main__":
    main()