"""
plot_benchmark.py
=================
Read benchmark_log.json produced by run_benchmark.py and regenerate the
two figures used in the thesis:

  - <method>_TA_Benchmark.png/.svg
      Combined signal-overlay + MAE table for the best variant of each
      discretization method. Used as figure benchmarksNaive (and the
      SAX/Persist counterparts in the appendix).

  - <method>_table_with_time.png/.svg
      TA-structure table (states, edges, training time) per parameter
      setting. Used as figure TA-structure-naive (and the SAX/Persist
      counterparts in the appendix).

Log layout: <TA_Benchmark>/<timestamp>/k<n>/benchmark_log.json

Usage:
    python plot_benchmark.py                       # latest run for k=2, k=4
    python plot_benchmark.py --k_list 4            # just k=4
    python plot_benchmark.py --log path/to/log.json
"""

import argparse
import json
import os

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec


ROOT = os.path.normpath(os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "..", "..",
))


# ---------------------------------------------------------------------------
# Path / label helpers
# ---------------------------------------------------------------------------

def _rename_label(label):
    """Display 'k=' as 'bins=' in tables. Leaves other tokens (w=, etc.) intact."""
    return label.replace("k=", "bins=")


def _translate_path(stored_path):
    """
    Stored absolute paths from a remote (SLURM/Linux) run won't resolve on
    other machines. Rebuild relative to the local project ROOT by anchoring
    at the 'Data/' segment.
    """
    s = str(stored_path).replace("\\", "/")
    idx = s.find("/Data/")
    if idx != -1:
        return os.path.normpath(os.path.join(ROOT, s[idx + 1:]))
    if s.startswith("Data/"):
        return os.path.normpath(os.path.join(ROOT, s))
    return stored_path


# ---------------------------------------------------------------------------
# Status filtering
# ---------------------------------------------------------------------------

def _ok(r):
    return r.get("status", "ok") == "ok"


def _ok_results(results):
    return [r for r in results if _ok(r)]


# ---------------------------------------------------------------------------
# Log loading
# ---------------------------------------------------------------------------

def load_log(log_path):
    with open(log_path) as f:
        return json.load(f)


def find_latest_log(tag_k=2):
    base = os.path.normpath(os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "..", "..", "Data", "Graphs", "TA_Benchmark"
    ))
    if not os.path.isdir(base):
        return None

    candidates = []
    for run_entry in os.scandir(base):
        if not run_entry.is_dir():
            continue
        log = os.path.join(run_entry.path, f"k{tag_k}", "benchmark_log.json")
        if os.path.isfile(log):
            candidates.append((run_entry.name, log))

    if not candidates:
        return None
    candidates.sort(key=lambda x: x[0])
    return candidates[-1][1]


# ---------------------------------------------------------------------------
# Cell formatters
# ---------------------------------------------------------------------------

def _fmt_states(r):
    if not _ok(r):
        return "FAILED"
    return f"{r['n_states_median']:.0f} ({r['n_states_min']}-{r['n_states_max']})"


def _fmt_edges(r):
    if not _ok(r):
        return "FAILED"
    return f"{r['n_edges_median']:.0f} ({r['n_edges_min']}-{r['n_edges_max']})"


def _fmt_mae(r):
    if not _ok(r):
        return "FAILED"
    return f"{r['mae_median']:.3f} ({r['mae_min']:.3f}-{r['mae_max']:.3f})"


def _fmt_time(r):
    if not _ok(r):
        return "-"
    return f"{r['time_median']:.2f}±{r['time_std']:.2f}s"


def _consistency_marker(r):
    """Inline marker appended to the variant label when failures/inconsistencies exist."""
    if not _ok(r):
        return f" ({r.get('error_type', 'failed')})"
    n_inc = r["n_total"] - r["n_consistent"]
    if n_inc == 0:
        return ""
    return f" ✗{n_inc}/{r['n_total']}"


# ---------------------------------------------------------------------------
# Table cell coloring
# ---------------------------------------------------------------------------

def _header_color(tab, n_cols, color="#FFD700"):
    for j in range(n_cols):
        tab[(0, j)].set_facecolor(color)


def _highlight_best(tab, n_cols, color="#CCFFCC"):
    """Highlight first data row (best variant by lowest median MAE)."""
    for j in range(n_cols):
        tab[(1, j)].set_facecolor(color)


def _color_failed_row(tab, row_idx, n_cols, color="#FFCCCC"):
    for j in range(n_cols):
        tab[(row_idx + 1, j)].set_facecolor(color)


# ---------------------------------------------------------------------------
# Raw-trace loaders
# ---------------------------------------------------------------------------

def _load_raw_for_variant(r):
    path = _translate_path(r["plot_trace_path"])
    data = np.genfromtxt(path, delimiter=";", skip_header=1)
    return data[:, 0], data[:, 1]


def _load_raw_for_path(path):
    path = _translate_path(path)
    data = np.genfromtxt(path, delimiter=";", skip_header=1)
    return data[:, 0], data[:, 1]


# ---------------------------------------------------------------------------
# Figure 1: combined signal + MAE table for best variant per method
# Used in the thesis as benchmarksNaive (and SAX/Persist appendix variants).
# ---------------------------------------------------------------------------

def plot_combined(method_name, results, output_folder):
    ok = _ok_results(results)
    if not ok:
        print(f"  Skipping {method_name}_TA_Benchmark — no successful variants")
        return

    best = ok[0]  # results assumed pre-sorted by median MAE in the log

    # Prefer the first shared reference trace so all methods show the same
    # raw trace. Fall back to per-variant median-MAE trace for older logs.
    ref_traces = best.get("reference_traces", [])
    if ref_traces:
        ref = ref_traces[0]
        t_raw, v_raw = _load_raw_for_path(ref["trace_path"])
        t_d = np.array(ref["t_d"])
        v_d = np.array(ref["v_d"])
        resids = np.array(ref["resids"])
    else:
        t_raw, v_raw = _load_raw_for_variant(best)
        t_d = np.array(best["plot_t_d"])
        v_d = np.array(best["plot_v_d"])
        resids = np.array(best["plot_resids"])

    fig = plt.figure(figsize=(18, 8))
    gs = GridSpec(2, 2, width_ratios=[4.2, 3.2],
                  height_ratios=[3, 1], hspace=0.28, wspace=0.18)

    ax_top = fig.add_subplot(gs[0, 0])
    ax_bot = fig.add_subplot(gs[1, 0], sharex=ax_top)
    ax_table = fig.add_subplot(gs[:, 1])
    ax_table.axis("off")

    # Signal overlay
    ax_top.plot(np.array(t_raw) / 3600, v_raw, alpha=0.5, label="Raw")
    ax_top.step(t_d / 3600, v_d, where="post", label="Discretized")
    ax_top.set_title(
        f"{method_name} — {_rename_label(best['label'])}"
        f"{_consistency_marker(best)}"
    )
    ax_top.set_ylabel("Temperature")
    ax_top.legend()

    # Residuals
    ax_bot.axhline(0, color="black", lw=1, ls="--")
    ax_bot.plot(t_d / 3600, resids)
    ax_bot.set_xlabel("Time (hours)")
    ax_bot.set_ylabel("Residual")

    # MAE table (all variants for this method, with failed rows highlighted)
    failed_row_indices = []
    table_data = []
    for i, r in enumerate(results):
        row = [_rename_label(r["label"]) + _consistency_marker(r), _fmt_mae(r)]
        table_data.append(row)
        if not _ok(r):
            failed_row_indices.append(i)

    cols = ["Parameter", "MAE\n(median, min-max)"]
    n_cols = len(cols)
    tab = ax_table.table(
        cellText=table_data, colLabels=cols, cellLoc="center", loc="center"
    )
    tab.auto_set_font_size(False)
    tab.set_fontsize(9.5)
    tab.scale(1.2, 2.3)
    _header_color(tab, n_cols)
    _highlight_best(tab, n_cols)
    for idx in failed_row_indices:
        _color_failed_row(tab, idx, n_cols)

    if failed_row_indices:
        ax_table.text(0.5, -0.03,
                      f"{len(failed_row_indices)} failed variants in red",
                      ha="center", fontsize=8, clip_on=False,
                      transform=ax_table.transAxes, color="gray")

    base_name = os.path.join(output_folder, f"{method_name}_TA_Benchmark")
    plt.savefig(f"{base_name}.png", bbox_inches="tight", dpi=300)
    plt.savefig(f"{base_name}.svg", bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {base_name}.png / .svg")


# ---------------------------------------------------------------------------
# Figure 2: TA-structure table (states, edges, training time)
# Used in the thesis as TA-structure-naive (and SAX/Persist appendix variants).
# ---------------------------------------------------------------------------

def plot_structure_table_with_time(method_name, results, output_folder):
    n_traces = next((r["n_total"] for r in results if _ok(r)), None)

    rows = []
    failed_row_indices = []
    for i, r in enumerate(results):
        rows.append([
            _rename_label(r["label"]),
            _fmt_states(r), _fmt_edges(r), _fmt_time(r),
        ])
        if not _ok(r):
            failed_row_indices.append(i)

    cols = ["Parameter",
            "States\n(median, min-max)",
            "Edges\n(median, min-max)",
            "TA training Time(s)"]
    n_cols = len(cols)
    n_data = len(rows)

    # Figure sizing keyed to row count so tables of different lengths
    # (across methods) don't get stretched or cramped vertically.
    row_h    = 0.38
    title_h  = 0.30
    footer_h = 0.30
    fig_h = (n_data + 1) * row_h + title_h + footer_h
    fig_w = max(7, 2.0 * n_cols)

    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.axis("off")

    footer_frac = footer_h / fig_h
    title_frac  = title_h  / fig_h
    bbox = [0, footer_frac, 1, 1.0 - footer_frac - title_frac]

    tab = ax.table(cellText=rows, colLabels=cols, cellLoc="center", bbox=bbox)
    tab.auto_set_font_size(False)
    tab.set_fontsize(9.5)
    _header_color(tab, n_cols)
    for idx in failed_row_indices:
        _color_failed_row(tab, idx, n_cols)

    # Title above the table
    ax.text(0.5, 1.0 - title_frac / 2,
            f"{method_name} — TA structure",
            ha="center", va="center", fontsize=12, fontweight="bold",
            transform=ax.transAxes)

    # Footer with replicate count and any failure note
    footer_parts = []
    if n_traces is not None:
        footer_parts.append(f"Median across {n_traces} per-trace TAs")
    if failed_row_indices:
        footer_parts.append(f"{len(failed_row_indices)} failed variants in red")
    if footer_parts:
        ax.text(0.5, footer_frac / 2, "  |  ".join(footer_parts),
                ha="center", va="center", fontsize=8, color="gray",
                transform=ax.transAxes)

    fig.subplots_adjust(top=1.0, bottom=0.0, left=0.01, right=0.99)

    base_name = os.path.join(output_folder, f"{method_name}_table_with_time")
    plt.savefig(f"{base_name}.png", bbox_inches="tight", dpi=300)
    plt.savefig(f"{base_name}.svg", bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {base_name}.png / .svg")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--log", default=None,
        help="Path to a specific benchmark_log.json. If provided, --k_list is ignored.",
    )
    parser.add_argument(
        "--k_list", type=int, nargs="+", default=[2, 4],
        help="Space-separated list of TAG k values to plot. Default: 2 4",
    )
    parser.add_argument(
        "--out", default=None,
        help="Base output folder. Results saved in k-specific subfolders.",
    )
    args = parser.parse_args()

    logs_to_process = []
    if args.log:
        logs_to_process.append(args.log)
    else:
        for k in args.k_list:
            log_path = find_latest_log(tag_k=k)
            if log_path:
                logs_to_process.append(log_path)
            else:
                print(f"Warning: No benchmark_log.json found for k={k}. Skipping.")

    if not logs_to_process:
        print("No logs found to process. Exiting.")
        return

    for log_path in logs_to_process:
        log = load_log(log_path)
        current_k = log.get("tag_k", "?")

        print("\n" + "=" * 40)
        print(f"Processing k={current_k}")
        print(f"Log: {log_path}")
        print("=" * 40)

        if args.out:
            k_out_dir = os.path.join(args.out, f"k{current_k}")
        else:
            k_out_dir = os.path.dirname(os.path.abspath(log_path))
        os.makedirs(k_out_dir, exist_ok=True)
        print(f"Output folder: {k_out_dir}")

        for method_name, results in log["methods"].items():
            print(f"  Plotting method: {method_name}")
            plot_combined(method_name, results, k_out_dir)
            plot_structure_table_with_time(method_name, results, k_out_dir)


if __name__ == "__main__":
    main()