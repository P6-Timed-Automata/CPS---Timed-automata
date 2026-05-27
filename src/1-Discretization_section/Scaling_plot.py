"""
plot_scaling.py
===============
Read scaling_log.json produced by run_scaling.py and regenerate the
per-dataset scaling figures used in the thesis:

    learn_time_<dataset>_raw.svg   — TAG learning time vs. trace count
    n_states_<dataset>_raw.svg     — TA state count   vs. trace count
    n_edges_<dataset>_raw.svg      — TA edge count    vs. trace count

One file per (metric × dataset) combination. Figures are saved next to
the input log (or to --out if given).

Usage:
    python plot_scaling.py                  # latest run, k=4
    python plot_scaling.py --k 2            # latest run, k=2
    python plot_scaling.py --log path/to/scaling_log.json
"""

import argparse
import json
import os

import matplotlib.pyplot as plt
import numpy as np


# =============================================================================
# DATA FILTERS
# =============================================================================

def _is_ok_cell(stat_dict):
    """A stat_summary dict is OK if it has a non-None median."""
    return stat_dict is not None and stat_dict.get("median") is not None


def _is_completed(result):
    """A result is completed iff it has trace_counts as a list.
    Placeholders from a crashed runner have status='in_progress' and no
    trace_counts."""
    return isinstance(result.get("trace_counts"), list)


def _display_params(result):
    """Format a result's params dict for display, with the Persist
    off-by-one bin-count cosmetic correction."""
    params = result.get("params")
    if not params:
        return ""
    if result.get("method", "").lower() == "persist" and "bins" in params:
        params = {**params, "bins": params["bins"] - 1}
    return str(params)


# =============================================================================
# LOG LOADING
# =============================================================================

def load_log(log_path):
    with open(log_path) as f:
        return json.load(f)


def find_latest_log(tag_k=4):
    """Find the most recent scaling_log.json for the given k under
    ScalingExperiments/<timestamp>/k<n>/."""
    base = os.path.normpath(os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "..", "..", "Data", "Graphs", "ScalingExperiments"
    ))
    if not os.path.isdir(base):
        return None

    candidates = []
    for run_entry in os.scandir(base):
        if not run_entry.is_dir():
            continue
        log = os.path.join(run_entry.path, f"k{tag_k}", "scaling_log.json")
        if os.path.isfile(log):
            candidates.append((run_entry.name, log))

    if not candidates:
        return None
    candidates.sort(key=lambda x: x[0])
    return candidates[-1][1]


# =============================================================================
# COLORS / METRICS
# =============================================================================

_COLORS = [
    "navy", "firebrick", "forestgreen", "darkorange",
    "purple", "teal", "brown", "darkmagenta",
]

# (metric_key, ylabel, integer_y)
METRICS = [
    ("learn_time", "TAG learning time (s)", False),
    ("n_states",   "TA state count",        True),
    ("n_edges",    "TA edge count",         True),
]


# =============================================================================
# SERIES EXTRACTION / PLOT PRIMITIVES
# =============================================================================

def _extract_series(result, metric_key):
    """Extract (x, median, lower_err, upper_err) for the given metric,
    skipping cells where every repeat failed."""
    stats = result.get(metric_key, [])
    trace_counts = result.get("trace_counts", [])
    valid = [(tc, s) for tc, s in zip(trace_counts, stats) if _is_ok_cell(s)]
    if not valid:
        return [], [], [], []

    x       = [tc for tc, _ in valid]
    y       = [s["median"] for _, s in valid]
    y_low   = [s["median"] - s["min"] for _, s in valid]
    y_high  = [s["max"]    - s["median"] for _, s in valid]
    return x, y, y_low, y_high


def _plot_series(ax, x, y, y_low, y_high, color, label):
    """One series with asymmetric (median - min, max - median) error bars."""
    yerr = np.array([y_low, y_high])
    ax.errorbar(
        x, y, yerr=yerr,
        fmt="o-", color=color, ecolor=color,
        linewidth=2, markersize=4, capsize=5,
        label=label,
    )


def _finalize_axes(ax, xlabel, ylabel, title, integer_y=False):
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, linestyle="--", alpha=0.5)
    if integer_y:
        ax.yaxis.get_major_locator().set_params(integer=True)
    ax.legend(fontsize=8)


def _save_fig(fig, out_path):
    fig.tight_layout()
    # dpi is technically ignored by SVG, but kept for safe raster fallback
    fig.savefig(out_path, bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"  Saved: {out_path}")


# =============================================================================
# PER-DATASET PLOT (the only plot type used in the thesis)
# =============================================================================

def plot_per_dataset(results, output_folder, repeats, tag_k,
                     metric_key, ylabel, integer_y):
    """One plot per dataset, showing all variants for that dataset.
    File naming: <metric_key>_<dataset>_raw.svg."""
    datasets = sorted(set(r.get("dataset", "?") for r in results))

    for dataset in datasets:
        subset = [r for r in results if r.get("dataset") == dataset]
        if not subset:
            continue

        fig, ax = plt.subplots(figsize=(10, 5))
        plotted_any = False

        for result, color in zip(subset, _COLORS):
            x, y, y_low, y_high = _extract_series(result, metric_key)
            if not x:
                continue
            plotted_any = True
            method_name = result.get("method", "?").upper()
            params_str  = _display_params(result)
            _plot_series(ax, x, y, y_low, y_high,
                         color, f"{method_name} ({params_str})")

        if not plotted_any:
            plt.close(fig)
            print(f"  Skipping {dataset}/{metric_key} — no data")
            continue

        _finalize_axes(
            ax,
            xlabel="Training trace count",
            ylabel=ylabel,
            title=f"{dataset} dataset — {ylabel}  "
                  f"(k={tag_k}, median of {repeats} runs)",
            integer_y=integer_y,
        )
        out_path = os.path.join(output_folder,
                                f"{metric_key}_{dataset}_raw.svg")
        _save_fig(fig, out_path)


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--log", default=None,
        help="Path to a specific scaling_log.json. Overrides --k.",
    )
    parser.add_argument(
        "--k", type=int, default=4,
        help="TAG k value to plot (selects <run>/k<n>/scaling_log.json). "
             "Default: 4.",
    )
    parser.add_argument(
        "--out", default=None,
        help="Output folder. Defaults to the log's own k<n>/ folder.",
    )
    args = parser.parse_args()

    if args.log:
        log_path = args.log
    else:
        log_path = find_latest_log(tag_k=args.k)
        if log_path is None:
            print(f"No scaling_log.json found for k={args.k}.")
            return
        print(f"Auto-selected log: {log_path}")

    out_dir = args.out or os.path.dirname(os.path.abspath(log_path))
    os.makedirs(out_dir, exist_ok=True)

    print(f"\nProcessing: {log_path}")
    log = load_log(log_path)

    raw_results = log.get("results", [])
    results = [r for r in raw_results if _is_completed(r)]
    n_skipped = len(raw_results) - len(results)

    repeats = log.get("repeats", "?")
    tag_k   = log.get("tag_k", "?")

    print(f"  Run timestamp: {log.get('timestamp', 'unknown')} (k={tag_k})")
    print(f"  Experiments:   {len(results)} completed")
    if n_skipped > 0:
        print(f"  Skipped:       {n_skipped} placeholder/in-progress entries")
    print(f"  Output:        {out_dir}\n")

    if not results:
        print("  No completed results. Nothing to plot.")
        return

    for metric_key, ylabel, integer_y in METRICS:
        print(f"  === Metric: {ylabel} ===")
        plot_per_dataset(results, out_dir, repeats, tag_k,
                         metric_key, ylabel, integer_y)


if __name__ == "__main__":
    main()