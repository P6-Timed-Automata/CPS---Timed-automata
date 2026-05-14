"""
plot_scaling.py
===============
Read scaling_log.json produced by run_scaling.py and regenerate all figures.

Now produces plots for three timings and two structure metrics:
  - disc_time   : just the discretization step
  - learn_time  : just the TAG learning step
  - total_time  : discretization + symbolic conversion + TAG learning
  - n_states    : TA state count
  - n_edges     : TA edge count

Each metric gets four figure sets (individual / combined / per-dataset /
per-method). Timing plots also get *_fit.png variants with linear and
exponential curve fits.

New layout: ScalingExperiments/<timestamp>/k<n>/scaling_log.json
Use --k to pick which TAG k value to plot (default 2).

Usage:
    python plot_scaling.py                    # latest run, k=2
    python plot_scaling.py --k 3              # latest run, k=3
    python plot_scaling.py --log path/to/scaling_log.json
"""

import argparse
import json
import os

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit


# =============================================================================
# COLORS
# =============================================================================

_COLORS = [
    "navy", "firebrick", "forestgreen", "darkorange",
    "purple", "teal", "brown", "darkmagenta",
]


# =============================================================================
# LOG LOADING
# =============================================================================

def load_log(log_path):
    with open(log_path) as f:
        return json.load(f)


def find_latest_log(tag_k=2):
    """
    Find the most recent scaling_log.json for the given TAG k value.

    New layout: <base>/<timestamp>/k<n>/scaling_log.json
    """
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
# STAT-SUMMARY EXTRACTION
# =============================================================================

def _extract_series(result, metric_key, central="median"):
    """
    Extract a (x, y_central, y_lower_err, y_upper_err) tuple for plotting.

    Parameters
    ----------
    result      : one entry from result["results"]
    metric_key  : "disc_time", "learn_time", "total_time", "n_states", "n_edges"
    central     : "median" or "mean" — which central tendency to plot

    Returns
    -------
    x           : list of trace counts
    y           : list of central values (median or mean)
    y_low       : asymmetric lower error (y - min) for median, (std) for mean
    y_high      : asymmetric upper error (max - y) for median, (std) for mean
    """
    x = result["trace_counts"]
    stats = result[metric_key]

    if central == "median":
        y      = [s["median"] for s in stats]
        y_low  = [s["median"] - s["min"] for s in stats]
        y_high = [s["max"] - s["median"] for s in stats]
    elif central == "mean":
        y      = [s["mean"] for s in stats]
        y_low  = [s["std"] for s in stats]
        y_high = [s["std"] for s in stats]
    else:
        raise ValueError(f"Unknown central='{central}'")

    return x, y, y_low, y_high


# =============================================================================
# CURVE FITTING (unchanged from original; operates on x, y arrays)
# =============================================================================

def _r2(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return float(1.0 - ss_res / ss_tot) if ss_tot > 0 else 0.0


def _fit_both(trace_counts, y_values):
    """Fit linear + exponential curves. Returns dict of fit info."""
    x = np.array(trace_counts, dtype=float)
    y = np.array(y_values, dtype=float)
    x_dense = np.linspace(x[0], x[-1], 200)

    lin_c = np.polyfit(x, y, 1)
    lin_pred = np.polyval(lin_c, x)
    lin_r2 = _r2(y, lin_pred)

    fits = {
        "linear": {
            "label":   f"linear fit (R²={lin_r2:.3f})",
            "x_dense": x_dense,
            "y_dense": np.polyval(lin_c, x_dense),
            "r2":      lin_r2,
        }
    }

    if np.all(y > 0):
        try:
            def _exp(x, a, b):
                return a * np.exp(b * x)

            popt, _ = curve_fit(_exp, x, y, p0=(y[0], 0.01), maxfev=10000)
            exp_pred = _exp(x, *popt)
            exp_r2 = _r2(y, exp_pred)
            a, b = popt

            fits["exponential"] = {
                "label":   f"exponential fit (R²={exp_r2:.3f})",
                "x_dense": x_dense,
                "y_dense": a * np.exp(b * x_dense),
                "r2":      exp_r2,
            }
        except Exception:
            pass

    return fits


def _fit_tag(trace_counts, y_values):
    """Short legend tag with R² values for combined plots."""
    fits = _fit_both(trace_counts, y_values)
    parts = [f"lin R²={fits['linear']['r2']:.3f}"]
    if "exponential" in fits:
        parts.append(f"exp R²={fits['exponential']['r2']:.3f}")
    return f"({', '.join(parts)})"


# =============================================================================
# METRIC METADATA
# =============================================================================

# (key, y_label, file_prefix, supports_fit, integer_y)
METRICS = [
    ("disc_time",  "Discretization time (s)",  "disc_time",  True,  False),
    ("learn_time", "TAG learning time (s)",    "learn_time", True,  False),
    ("total_time", "Total pipeline time (s)",  "total_time", True,  False),
    ("n_states",   "TA state count",           "n_states",   False, True),
    ("n_edges",    "TA edge count",            "n_edges",    False, True),
]


# =============================================================================
# PLOT-BUILDING PRIMITIVES
# =============================================================================

def _plot_series(ax, x, y, y_low, y_high, color, label, integer_y=False):
    """Plot one series with asymmetric error bars."""
    yerr = np.array([y_low, y_high])
    ax.errorbar(
        x, y, yerr=yerr,
        fmt="o-", color=color, ecolor=color,
        linewidth=2, markersize=4, capsize=5,
        label=label,
    )


def _overlay_fits(ax, trace_counts, y_values, color, alpha=0.6):
    """Overlay linear + exponential fits on the current axes."""
    fits = _fit_both(trace_counts, y_values)
    if "linear" in fits:
        ax.plot(fits["linear"]["x_dense"], fits["linear"]["y_dense"],
                linestyle="--", linewidth=1.1, color=color, alpha=alpha)
    if "exponential" in fits:
        ax.plot(fits["exponential"]["x_dense"], fits["exponential"]["y_dense"],
                linestyle="-.", linewidth=1.1, color=color, alpha=alpha)


def _finalize_axes(ax, xlabel, ylabel, title, integer_y=False, legend_kwargs=None):
    """Common axes setup."""
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, linestyle="--", alpha=0.5)
    if integer_y:
        ax.yaxis.get_major_locator().set_params(integer=True)
    if legend_kwargs is not None:
        ax.legend(**legend_kwargs)
    else:
        ax.legend(fontsize=8)


def _save_fig(fig, out_path):
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"  Saved: {out_path}")


# =============================================================================
# PLOT TYPES (parameterized by metric)
# =============================================================================

def plot_individual(results, output_folder, repeats, tag_k,
                    metric_key, ylabel, file_prefix,
                    supports_fit, integer_y, show_fit=False):
    if show_fit and not supports_fit:
        return   # don't generate fit variants for integer-valued metrics
    suffix = "_fit" if show_fit else "_raw"

    for result in results:
        x, y, y_low, y_high = _extract_series(result, metric_key)
        label = result["label"]

        fig, ax = plt.subplots(figsize=(10, 5))
        _plot_series(ax, x, y, y_low, y_high, color="navy", label="Measured")

        if show_fit:
            _overlay_fits(ax, x, y, color="navy")
            # Also draw the explicit fit legend lines
            fits = _fit_both(x, y)
            fit_styles = {
                "linear":      ("firebrick",   "--"),
                "exponential": ("forestgreen", "-."),
            }
            for fit_type, (fc, ls) in fit_styles.items():
                if fit_type in fits:
                    ax.plot(fits[fit_type]["x_dense"], fits[fit_type]["y_dense"],
                            linestyle=ls, linewidth=1.5, color=fc,
                            label=fits[fit_type]["label"])

        _finalize_axes(
            ax,
            xlabel="Training trace count",
            ylabel=ylabel,
            title=f"{label}  (k={tag_k}, median of {repeats} runs)",
            integer_y=integer_y,
        )

        safe_label = label.replace(" ", "_").replace("/", "-").replace("—", "-")
        out_path = os.path.join(
            output_folder, f"{file_prefix}_{safe_label}{suffix}.png"
        )
        _save_fig(fig, out_path)


def plot_combined(results, output_folder, repeats, tag_k,
                  metric_key, ylabel, file_prefix,
                  supports_fit, integer_y, show_fit=False):
    if show_fit and not supports_fit:
        return
    suffix = "_fit" if show_fit else "_raw"

    fig, ax = plt.subplots(figsize=(12, 6))
    for result, color in zip(results, _COLORS):
        x, y, y_low, y_high = _extract_series(result, metric_key)
        if show_fit:
            tag = _fit_tag(x, y)
            legend_label = f"{result['label']} {tag}"
            _overlay_fits(ax, x, y, color=color)
        else:
            legend_label = result["label"]
        _plot_series(ax, x, y, y_low, y_high, color, legend_label)

    _finalize_axes(
        ax,
        xlabel="Training trace count",
        ylabel=ylabel,
        title=f"Scaling — {ylabel}  (k={tag_k}, median of {repeats} runs)",
        integer_y=integer_y,
    )

    out_path = os.path.join(
        output_folder, f"{file_prefix}_combined{suffix}.png"
    )
    _save_fig(fig, out_path)


def plot_per_dataset(results, output_folder, repeats, tag_k,
                     metric_key, ylabel, file_prefix,
                     supports_fit, integer_y, show_fit=False):
    if show_fit and not supports_fit:
        return
    suffix = "_fit" if show_fit else "_raw"
    datasets = sorted(set(r["dataset"] for r in results))

    for dataset in datasets:
        subset = [r for r in results if r["dataset"] == dataset]
        fig, ax = plt.subplots(figsize=(10, 5))

        for result, color in zip(subset, _COLORS):
            x, y, y_low, y_high = _extract_series(result, metric_key)
            if show_fit:
                tag = _fit_tag(x, y)
                legend_label = f"{result['method'].upper()} ({result['params']}) {tag}"
                _overlay_fits(ax, x, y, color=color)
            else:
                legend_label = f"{result['method'].upper()} ({result['params']})"
            _plot_series(ax, x, y, y_low, y_high, color, legend_label)

        _finalize_axes(
            ax,
            xlabel="Training trace count",
            ylabel=ylabel,
            title=f"{dataset} dataset — {ylabel}  (k={tag_k}, median of {repeats} runs)",
            integer_y=integer_y,
        )

        out_path = os.path.join(
            output_folder, f"{file_prefix}_{dataset}{suffix}.png"
        )
        _save_fig(fig, out_path)


def plot_per_method(results, output_folder, repeats, tag_k,
                    metric_key, ylabel, file_prefix,
                    supports_fit, integer_y, show_fit=False):
    if show_fit and not supports_fit:
        return
    suffix = "_fit" if show_fit else "_raw"
    methods = sorted(set(r["method"] for r in results))

    for method in methods:
        subset = [r for r in results if r["method"] == method]
        fig, ax = plt.subplots(figsize=(10, 5))

        for result, color in zip(subset, _COLORS):
            x, y, y_low, y_high = _extract_series(result, metric_key)
            if show_fit:
                tag = _fit_tag(x, y)
                legend_label = f"{result['dataset']} {tag}"
                _overlay_fits(ax, x, y, color=color)
            else:
                legend_label = result["dataset"]
            _plot_series(ax, x, y, y_low, y_high, color, legend_label)

        _finalize_axes(
            ax,
            xlabel="Training trace count",
            ylabel=ylabel,
            title=f"{method.upper()} — {ylabel}  (k={tag_k}, median of {repeats} runs)",
            integer_y=integer_y,
            legend_kwargs={"title": "Dataset", "fontsize": 8},
        )

        out_path = os.path.join(
            output_folder, f"{file_prefix}_{method}{suffix}.png"
        )
        _save_fig(fig, out_path)


# =============================================================================
# CONSISTENCY OVERVIEW (bonus — one plot summarizing inconsistency across runs)
# =============================================================================

def plot_consistency_summary(results, output_folder, repeats, tag_k):
    """
    A small overview plot showing how many of the per-n repeats were consistent
    for each experiment. Useful to spot configurations that produce broken TAs.
    """
    fig, ax = plt.subplots(figsize=(12, 6))

    for result, color in zip(results, _COLORS):
        x = result["trace_counts"]
        consistency_ratio = [c / repeats for c in result["n_consistent"]]
        ax.plot(x, consistency_ratio, "o-", color=color,
                label=result["label"], linewidth=2, markersize=4)

    ax.set_xlabel("Training trace count")
    ax.set_ylabel(f"Consistent TAs / {repeats}")
    ax.set_title(f"TA consistency rate per configuration (k={tag_k})")
    ax.set_ylim(-0.05, 1.05)
    ax.axhline(1.0, color="black", linewidth=0.6, linestyle=":", alpha=0.5)
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(fontsize=8)

    out_path = os.path.join(output_folder, "consistency_summary.png")
    _save_fig(fig, out_path)


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--log", default=None,
        help="Path to scaling_log.json. Defaults to most recent run for the given --k.",
    )
    parser.add_argument(
        "--k", type=int, default=2,
        help="TAG k value to plot (selects <run>/k<n>/scaling_log.json). Default: 2.",
    )
    parser.add_argument(
        "--out", default=None,
        help="Output folder for figures. Defaults to the same folder as the log file.",
    )
    args = parser.parse_args()

    if args.log is None:
        args.log = find_latest_log(tag_k=args.k)
        if args.log is None:
            print(f"No scaling_log.json found for k={args.k}. Run run_scaling.py first.")
            return
        print(f"Auto-selected log: {args.log}")

    out_dir = args.out if args.out else os.path.dirname(os.path.abspath(args.log))
    os.makedirs(out_dir, exist_ok=True)

    log = load_log(args.log)
    results = log["results"]
    repeats = log.get("repeats", "?")
    tag_k = log.get("tag_k", args.k)

    print(f"Plotting run from: {log.get('timestamp', 'unknown')} (k={tag_k})")
    print(f"Experiments:       {len(results)}")
    print(f"Output folder:     {out_dir}\n")

    for metric_key, ylabel, file_prefix, supports_fit, integer_y in METRICS:
        print(f"=== Metric: {ylabel} ===")

        for show_fit in [False, True]:
            if show_fit and not supports_fit:
                continue
            label = "with fit" if show_fit else "raw"
            print(f"  --- {label.upper()} ---")

            print(f"    Individual...")
            plot_individual(results, out_dir, repeats, tag_k,
                            metric_key, ylabel, file_prefix,
                            supports_fit, integer_y, show_fit=show_fit)

            print(f"    Combined...")
            plot_combined(results, out_dir, repeats, tag_k,
                          metric_key, ylabel, file_prefix,
                          supports_fit, integer_y, show_fit=show_fit)

            print(f"    Per dataset...")
            plot_per_dataset(results, out_dir, repeats, tag_k,
                             metric_key, ylabel, file_prefix,
                             supports_fit, integer_y, show_fit=show_fit)

            print(f"    Per method...")
            plot_per_method(results, out_dir, repeats, tag_k,
                            metric_key, ylabel, file_prefix,
                            supports_fit, integer_y, show_fit=show_fit)

        print()

    print("=== Consistency summary ===")
    plot_consistency_summary(results, out_dir, repeats, tag_k)
    print()

    print("Done.")


if __name__ == "__main__":
    main()