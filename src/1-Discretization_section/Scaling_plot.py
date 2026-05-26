"""
plot_scaling.py
===============
Read scaling_log.json produced by run_scaling.py and regenerate all figures.

Defensive against crashed runs:
  - Filters out placeholders / in-progress results (entries from a SLURM
    SIGKILL between placeholder-append and result-replace in the runner).
  - All plot functions skip cells where _stat_summary returned None
    (all repeats failed at that n).
  - All label lookups use .get() with fallbacks via _safe_label.

Usage:
    python plot_scaling.py                    # latest run, k=2, outputs SVG
    python plot_scaling.py --k 4              # latest run, k=4
    python plot_scaling.py --format png       # output as PNG instead
    python plot_scaling.py --log path/to/scaling_log.json
"""

import argparse
import json
import os

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit

def find_all_logs():
    """
    Find every scaling_log.json under ScalingExperiments/, across all
    timestamped runs and all k values.

    Returns
    -------
    list of str
        Sorted by (timestamp, k). Older timestamps first; within a timestamp,
        lower k values first.
    """
    base = os.path.normpath(os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "..", "..", "Data", "Graphs", "ScalingExperiments"
    ))
    if not os.path.isdir(base):
        return []

    logs = []
    for run_entry in sorted(os.scandir(base), key=lambda e: e.name):
        if not run_entry.is_dir():
            continue
        # Within each timestamp folder, look for any k<n>/scaling_log.json
        for k_entry in sorted(os.scandir(run_entry.path), key=lambda e: e.name):
            if not k_entry.is_dir():
                continue
            if not k_entry.name.startswith("k"):
                continue
            log_path = os.path.join(k_entry.path, "scaling_log.json")
            if os.path.isfile(log_path):
                logs.append(log_path)
    return logs

def _is_ok_cell(stat_dict):
    """A stat_summary dict is OK if it has a non-None median (>=1 successful repeat)."""
    return stat_dict is not None and stat_dict.get("median") is not None


def _ok_indices(result, metric):
    """Return indices into trace_counts where the given metric has data."""
    return [
        i for i, s in enumerate(result.get(metric, []))
        if _is_ok_cell(s)
    ]


def _is_completed(result):
    """
    A result is considered completed if it has trace_counts as a list.
    Placeholders inserted by run_scaling.py (when SIGKILLed between
    append-placeholder and replace-with-result) only have:
        {dataset, method, params, tag_k, status="in_progress"}
    and lack trace_counts and label.
    """
    return isinstance(result.get("trace_counts"), list)

def _display_params(result):
    """
    Format a result's params dict for display, applying cosmetic corrections.

    Persist's stored bin count is off-by-one relative to the effective
    alphabet size; decrement on display so figures show the matched
    alphabet size across methods.
    """
    params = result.get("params")
    if not params:
        return ""
    if result.get("method", "").lower() == "persist" and "bins" in params:
        params = {**params, "bins": params["bins"] - 1}
    return str(params)


def _safe_label(result):
    """Best-effort label for log/print messages. Routes through _display_params
    so cosmetic corrections apply consistently."""
    method = (result.get("method") or "").lower()
    # Use the stored label only when no cosmetic correction is needed.
    if "label" in result and method != "persist":
        return result["label"]

    parts = []
    if result.get("dataset"):
        parts.append(result["dataset"])
    if result.get("method"):
        parts.append(result["method"].upper())
    params_str = _display_params(result)
    if params_str:
        parts.append(params_str)
    return " ".join(parts) if parts else "<unknown>"

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
    Extract (x, y_central, y_lower_err, y_upper_err) for plotting,
    skipping cells where all repeats failed.
    """
    stats = result.get(metric_key, [])
    trace_counts = result.get("trace_counts", [])

    valid_indices = [i for i, s in enumerate(stats) if _is_ok_cell(s)]

    if not valid_indices:
        return [], [], [], []

    x = [trace_counts[i] for i in valid_indices]
    stats_valid = [stats[i] for i in valid_indices]

    if central == "median":
        y      = [s["median"] for s in stats_valid]
        y_low  = [s["median"] - s["min"] for s in stats_valid]
        y_high = [s["max"] - s["median"] for s in stats_valid]
    elif central == "mean":
        y      = [s["mean"] for s in stats_valid]
        y_low  = [s["std"] for s in stats_valid]
        y_high = [s["std"] for s in stats_valid]
    else:
        raise ValueError(f"Unknown central='{central}'")

    return x, y, y_low, y_high


# =============================================================================
# CURVE FITTING
# =============================================================================

def _r2(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return float(1.0 - ss_res / ss_tot) if ss_tot > 0 else 0.0


def _fit_both(trace_counts, y_values):
    """Fit linear + exponential curves. Returns dict of fit info."""
    x = np.array(trace_counts, dtype=float)
    y = np.array(y_values, dtype=float)

    if len(x) < 2:
        return {}

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
    if not fits:
        return ""
    parts = []
    if "linear" in fits:
        parts.append(f"lin R²={fits['linear']['r2']:.3f}")
    if "exponential" in fits:
        parts.append(f"exp R²={fits['exponential']['r2']:.3f}")
    return f"({', '.join(parts)})" if parts else ""


# =============================================================================
# METRIC METADATA
# =============================================================================

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
    # dpi is technically ignored by SVG, but kept for safe raster fallbacks
    fig.savefig(out_path, bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"  Saved: {out_path}")


# =============================================================================
# PLOT TYPES (parameterized by metric)
# =============================================================================

def plot_individual(results, output_folder, repeats, tag_k,
                    metric_key, ylabel, file_prefix,
                    supports_fit, integer_y, show_fit=False, fmt="svg"):
    if show_fit and not supports_fit:
        return
    suffix = "_fit" if show_fit else "_raw"

    for result in results:
        x, y, y_low, y_high = _extract_series(result, metric_key)
        if not x:
            print(f"  Skipping {_safe_label(result)} for {metric_key} — no successful cells")
            continue
        label = _safe_label(result)

        fig, ax = plt.subplots(figsize=(10, 5))
        _plot_series(ax, x, y, y_low, y_high, color="navy", label="Measured")

        if show_fit:
            _overlay_fits(ax, x, y, color="navy")
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
            output_folder, f"{file_prefix}_{safe_label}{suffix}.{fmt}"
        )
        _save_fig(fig, out_path)


def plot_combined(results, output_folder, repeats, tag_k,
                  metric_key, ylabel, file_prefix,
                  supports_fit, integer_y, show_fit=False, fmt="svg"):
    if show_fit and not supports_fit:
        return
    suffix = "_fit" if show_fit else "_raw"

    fig, ax = plt.subplots(figsize=(12, 6))
    plotted_any = False

    for result, color in zip(results, _COLORS):
        x, y, y_low, y_high = _extract_series(result, metric_key)
        if not x:
            continue
        plotted_any = True
        if show_fit:
            tag = _fit_tag(x, y)
            legend_label = f"{_safe_label(result)} {tag}"
            _overlay_fits(ax, x, y, color=color)
        else:
            legend_label = _safe_label(result)
        _plot_series(ax, x, y, y_low, y_high, color, legend_label)

    if not plotted_any:
        plt.close(fig)
        print(f"  Skipping combined plot for {metric_key} — no successful data in any variant")
        return

    _finalize_axes(
        ax,
        xlabel="Training trace count",
        ylabel=ylabel,
        title=f"Scaling — {ylabel}  (k={tag_k}, median of {repeats} runs)",
        integer_y=integer_y,
    )

    out_path = os.path.join(
        output_folder, f"{file_prefix}_combined{suffix}.{fmt}"
    )
    _save_fig(fig, out_path)


def plot_per_dataset(results, output_folder, repeats, tag_k,
                     metric_key, ylabel, file_prefix,
                     supports_fit, integer_y, show_fit=False, fmt="svg"):
    if show_fit and not supports_fit:
        return
    suffix = "_fit" if show_fit else "_raw"
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
            params_str = _display_params(result)
            if show_fit:
                tag = _fit_tag(x, y)
                legend_label = f"{method_name} ({params_str}) {tag}"
                _overlay_fits(ax, x, y, color=color)
            else:
                legend_label = f"{method_name} ({params_str})"
            _plot_series(ax, x, y, y_low, y_high, color, legend_label)

        if not plotted_any:
            plt.close(fig)
            print(f"  Skipping per-dataset plot for {dataset}/{metric_key} — no data")
            continue

        _finalize_axes(
            ax,
            xlabel="Training trace count",
            ylabel=ylabel,
            title=f"{dataset} dataset — {ylabel}  (k={tag_k}, median of {repeats} runs)",
            integer_y=integer_y,
        )

        out_path = os.path.join(
            output_folder, f"{file_prefix}_{dataset}{suffix}.{fmt}"
        )
        _save_fig(fig, out_path)


def plot_per_method(results, output_folder, repeats, tag_k,
                    metric_key, ylabel, file_prefix,
                    supports_fit, integer_y, show_fit=False, fmt="svg"):
    if show_fit and not supports_fit:
        return
    suffix = "_fit" if show_fit else "_raw"
    methods = sorted(set(r.get("method", "?") for r in results))

    for method in methods:
        subset = [r for r in results if r.get("method") == method]
        if not subset:
            continue

        fig, ax = plt.subplots(figsize=(10, 5))
        plotted_any = False

        for result, color in zip(subset, _COLORS):
            x, y, y_low, y_high = _extract_series(result, metric_key)
            if not x:
                continue
            plotted_any = True
            if show_fit:
                tag = _fit_tag(x, y)
                legend_label = f"{result.get('dataset', '?')} {tag}"
                _overlay_fits(ax, x, y, color=color)
            else:
                legend_label = result.get("dataset", "?")
            _plot_series(ax, x, y, y_low, y_high, color, legend_label)

        if not plotted_any:
            plt.close(fig)
            print(f"  Skipping per-method plot for {method}/{metric_key} — no data")
            continue

        _finalize_axes(
            ax,
            xlabel="Training trace count",
            ylabel=ylabel,
            title=f"{method.upper()} — {ylabel}  (k={tag_k}, median of {repeats} runs)",
            integer_y=integer_y,
            legend_kwargs={"title": "Dataset", "fontsize": 8},
        )

        out_path = os.path.join(
            output_folder, f"{file_prefix}_{method}{suffix}.{fmt}"
        )
        _save_fig(fig, out_path)


# =============================================================================
# CONSISTENCY OVERVIEW
# =============================================================================

def plot_consistency_summary(results, output_folder, repeats, tag_k, fmt="svg"):
    """
    Overview plot of consistency rate across configurations.
    """
    fig, ax = plt.subplots(figsize=(12, 6))
    plotted_any = False

    for result, color in zip(results, _COLORS):
        x = result.get("trace_counts", [])
        n_consistent = result.get("n_consistent", [])
        if not x or not n_consistent:
            continue
        plotted_any = True

        try:
            denom = float(repeats) if repeats else 1
        except (TypeError, ValueError):
            denom = 1
        consistency_ratio = [c / denom for c in n_consistent]

        ax.plot(x, consistency_ratio, "o-", color=color,
                label=_safe_label(result), linewidth=2, markersize=4)

    if not plotted_any:
        plt.close(fig)
        print("  Skipping consistency summary — no data")
        return

    ax.set_xlabel("Training trace count")
    ax.set_ylabel(f"Consistent TAs / {repeats}")
    ax.set_title(f"TA consistency rate per configuration (k={tag_k})")
    ax.set_ylim(-0.05, 1.05)
    ax.axhline(1.0, color="black", linewidth=0.6, linestyle=":", alpha=0.5)
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(fontsize=8)

    out_path = os.path.join(output_folder, f"consistency_summary.{fmt}")
    _save_fig(fig, out_path)


# =============================================================================
# FAILURE SUMMARY
# =============================================================================

def plot_failure_summary(log, output_folder, fmt="svg"):
    """
    Render a table of all failed cells in the scaling sweep.
    Skipped if everything succeeded.
    """
    all_failed = []
    for r in log.get("results", []):
        if not _is_completed(r):
            continue
        reasons_per_n = r.get("failure_reasons_per_n", [])
        trace_counts = r.get("trace_counts", [])
        if not reasons_per_n:
            continue
        for n, reasons in zip(trace_counts, reasons_per_n):
            for reason in reasons:
                all_failed.append({
                    "dataset":    r.get("dataset", "?"),
                    "method":     r.get("method", "?"),
                    "params":     str(r.get("params", "?")),
                    "n":          n,
                    "repeat":     reason.get("repeat", "?"),
                    "error_type": reason.get("error_type", "?"),
                    "error_msg":  reason.get("error_msg", "")[:120],
                })

    if not all_failed:
        return

    n_rows = len(all_failed) + 1
    fig_h = max(2, 0.35 * n_rows + 1)
    fig, ax = plt.subplots(figsize=(16, fig_h))
    ax.axis("off")

    headers = ["Dataset", "Method", "Params", "n", "Repeat",
               "Error type", "Error message"]
    cell_text = [
        [f["dataset"], f["method"], f["params"], str(f["n"]),
         str(f["repeat"]), f["error_type"], f["error_msg"]]
        for f in all_failed
    ]

    tbl = ax.table(cellText=cell_text, colLabels=headers,
                   cellLoc="center", loc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8)
    tbl.scale(1, 1.4)
    for j in range(len(headers)):
        tbl[(0, j)].set_facecolor("#2c3e50")
        tbl[(0, j)].set_text_props(color="white", fontweight="bold")

    tag_k = log.get("tag_k", "?")
    fig.suptitle(f"Failed scaling cells (k={tag_k})", fontsize=12,
                 fontweight="bold", y=0.97)

    out_path = os.path.join(output_folder, f"scaling_failures.{fmt}")
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


# =============================================================================
# MAIN
# =============================================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--log", default=None,
        help="Path to a specific scaling_log.json. Overrides --k and --all.",
    )
    parser.add_argument(
        "--k", type=int, default=4,
        help="TAG k value to plot (selects <run>/k<n>/scaling_log.json). "
             "Used only with single-log mode. Default: 2.",
    )
    parser.add_argument(
        "--all", action="store_true",
        help="Process every scaling_log.json under ScalingExperiments/. "
             "Plots are saved in each log's own k<n>/ folder.",
    )
    parser.add_argument(
        "--out", default=None,
        help="Output folder. Only used in single-log mode. "
             "In --all mode, plots go to each log's own folder.",
    )
    parser.add_argument(
        "--format", type=str, default="svg", choices=["png", "svg", "pdf"],
        help="Output format for the plots. Default: svg.",
    )
    args = parser.parse_args()
    fmt = args.format

    # ----- Build list of logs to process -----
    if args.log:
        logs_to_process = [args.log]
    elif args.all:
        logs_to_process = find_all_logs()
        if not logs_to_process:
            print("No scaling_log.json files found under ScalingExperiments/.")
            return
        print(f"Found {len(logs_to_process)} log(s) under ScalingExperiments/.")
    else:
        latest = find_latest_log(tag_k=args.k)
        if latest is None:
            print(f"No scaling_log.json found for k={args.k}. Run run_scaling.py first.")
            return
        logs_to_process = [latest]
        print(f"Auto-selected log: {latest}")

    # ----- Process each log -----
    summary = []   # (log_path, n_completed, n_skipped, status)

    for log_path in logs_to_process:
        print(f"\n{'=' * 70}")
        print(f"Processing: {log_path}")
        print(f"{'=' * 70}")

        # In --all mode, always save next to the log file
        if args.all or args.out is None:
            out_dir = os.path.dirname(os.path.abspath(log_path))
        else:
            out_dir = args.out
        os.makedirs(out_dir, exist_ok=True)

        try:
            log = load_log(log_path)
        except Exception as e:
            print(f"  FAILED to load log: {e}")
            summary.append((log_path, 0, 0, f"load error: {e}"))
            continue

        raw_results = log.get("results", [])
        results = [r for r in raw_results if _is_completed(r)]
        n_skipped = len(raw_results) - len(results)

        repeats = log.get("repeats", "?")
        tag_k = log.get("tag_k", "?")

        print(f"  Run timestamp: {log.get('timestamp', 'unknown')} (k={tag_k})")
        print(f"  Experiments:   {len(results)} completed")
        if n_skipped > 0:
            print(f"  Skipped:       {n_skipped} placeholder/in-progress entries")
        print(f"  Output:        {out_dir}\n")

        if not results:
            print("  No completed results. Skipping this log.\n")
            summary.append((log_path, 0, n_skipped, "no completed results"))
            continue

        n_total_failed_cells = sum(
            sum(1 for s in r.get("status_per_n", []) if s != "ok")
            for r in results
        )
        if n_total_failed_cells > 0:
            print(f"  Note: {n_total_failed_cells} (variant, n) cells had failures; "
                  f"see scaling_failures.{fmt}\n")

        try:
            for metric_key, ylabel, file_prefix, supports_fit, integer_y in METRICS:
                print(f"  === Metric: {ylabel} ===")

                for show_fit in [False, True]:
                    if show_fit and not supports_fit:
                        continue

                    plot_individual(results, out_dir, repeats, tag_k,
                                    metric_key, ylabel, file_prefix,
                                    supports_fit, integer_y, show_fit=show_fit, fmt=fmt)
                    plot_combined(results, out_dir, repeats, tag_k,
                                  metric_key, ylabel, file_prefix,
                                  supports_fit, integer_y, show_fit=show_fit, fmt=fmt)
                    plot_per_dataset(results, out_dir, repeats, tag_k,
                                     metric_key, ylabel, file_prefix,
                                     supports_fit, integer_y, show_fit=show_fit, fmt=fmt)
                    plot_per_method(results, out_dir, repeats, tag_k,
                                    metric_key, ylabel, file_prefix,
                                    supports_fit, integer_y, show_fit=show_fit, fmt=fmt)

            plot_consistency_summary(results, out_dir, repeats, tag_k, fmt=fmt)
            plot_failure_summary(log, out_dir, fmt=fmt)

            summary.append((log_path, len(results), n_skipped, "ok"))

        except Exception as e:
            print(f"  ERROR during plotting: {type(e).__name__}: {e}")
            summary.append((log_path, len(results), n_skipped,
                            f"plot error: {type(e).__name__}"))

    # ----- Final summary -----
    print(f"\n{'=' * 70}")
    print(f"Final summary")
    print(f"{'=' * 70}")
    for log_path, n_done, n_skip, status in summary:
        run_name = os.path.basename(
            os.path.dirname(os.path.dirname(os.path.abspath(log_path)))
        )
        k_name = os.path.basename(os.path.dirname(os.path.abspath(log_path)))
        print(f"  {status:25s} {run_name}/{k_name:6s} "
              f"({n_done} variants, {n_skip} skipped)")
    print(f"\nProcessed {len(summary)} log(s). Done.")


if __name__ == "__main__":
    main()