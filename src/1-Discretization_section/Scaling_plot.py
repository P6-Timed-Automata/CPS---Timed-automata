"""
plot_scaling.py
===============
Read scaling_log.json produced by run_scaling.py and regenerate all figures.

Produces two sets of every plot:
  *_raw.png  — measured data only (error bars)
  *_fit.png  — same data with linear/exponential fit overlay

Run directly from PyCharm (no arguments) — auto-selects the most recent run.
To plot a specific run:
    python plot_scaling.py --log path/to/ScalingExperiments/2026-05-12_14-30-00/scaling_log.json
"""

import argparse
import json
import os

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit


# =============================================================================
# HELPERS
# =============================================================================

_COLORS = [
    "navy",
    "firebrick",
    "forestgreen",
    "darkorange",
    "purple",
    "teal",
    "brown",
]


def load_log(log_path: str) -> dict:
    with open(log_path) as f:
        return json.load(f)


def find_latest_log() -> str:
    base = os.path.normpath(os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "..", "..", "Data", "Graphs", "ScalingExperiments"
    ))
    if not os.path.isdir(base):
        return None
    candidates = []
    for entry in os.scandir(base):
        if entry.is_dir():
            log = os.path.join(entry.path, "scaling_log.json")
            if os.path.isfile(log):
                candidates.append((entry.name, log))
    if not candidates:
        return None
    candidates.sort(key=lambda x: x[0])
    return candidates[-1][1]


# =============================================================================
# CURVE FITTING
# =============================================================================

def _r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return float(1.0 - ss_res / ss_tot) if ss_tot > 0 else 0.0


def _fit_both(trace_counts: list, means: list) -> dict:
    """
    Fit both linear and exponential curves in original space.

    Returns a dict with keys 'linear' and 'exponential', each containing:
        label   : legend string with R²
        x_dense : 200-point x array for smooth line
        y_dense : fitted y values
        r2      : R²
    """
    x = np.array(trace_counts, dtype=float)
    y = np.array(means,        dtype=float)
    x_dense = np.linspace(x[0], x[-1], 200)

    # --- Linear ---
    lin_c    = np.polyfit(x, y, 1)
    lin_pred = np.polyval(lin_c, x)
    lin_r2   = _r2(y, lin_pred)

    fits = {
        "linear": {
            "label":   f"linear fit (R²={lin_r2:.3f})",
            "x_dense": x_dense,
            "y_dense": np.polyval(lin_c, x_dense),
            "r2":      lin_r2,
        }
    }

    # --- Exponential: y = a * exp(b*x) fitted in original space ---
    if np.all(y > 0):
        try:
            def _exp(x, a, b):
                return a * np.exp(b * x)

            popt, _ = curve_fit(_exp, x, y, p0=(y[0], 0.01), maxfev=10000)
            exp_pred = _exp(x, *popt)
            exp_r2   = _r2(y, exp_pred)
            a, b     = popt

            fits["exponential"] = {
                "label":   f"exponential fit (R²={exp_r2:.3f})",
                "x_dense": x_dense,
                "y_dense": a * np.exp(b * x_dense),
                "r2":      exp_r2,
            }
        except Exception:
            pass

    return fits


def _fit_tag(trace_counts: list, means: list) -> str:
    """Short tag for combined legend showing both R² values."""
    fits = _fit_both(trace_counts, means)
    lin_r2 = fits["linear"]["r2"]
    parts  = [f"lin R²={lin_r2:.3f}"]
    if "exponential" in fits:
        parts.append(f"exp R²={fits['exponential']['r2']:.3f}")
    return f"({', '.join(parts)})"


# =============================================================================
# INDIVIDUAL PLOTS
# =============================================================================

def plot_individual(results: list, output_folder: str, repeats: int,
                    show_fit: bool = False) -> None:
    suffix = "_fit" if show_fit else "_raw"

    for result in results:
        trace_counts = result["trace_counts"]
        means        = result["means"]
        stds         = result["stds"]
        label        = result["label"]

        fig, ax = plt.subplots(figsize=(10, 5))

        ax.errorbar(
            trace_counts, means, yerr=stds,
            fmt="o-", linewidth=2, markersize=5, capsize=5,
            label="Measured", color="navy",
        )

        if show_fit:
            fits = _fit_both(trace_counts, means)
            fit_styles = {
                "linear":      ("firebrick",   "--"),
                "exponential": ("forestgreen", "-."),
            }
            for fit_type, (fc, ls) in fit_styles.items():
                if fit_type in fits:
                    ax.plot(fits[fit_type]["x_dense"], fits[fit_type]["y_dense"],
                            linestyle=ls, linewidth=1.5, color=fc,
                            label=fits[fit_type]["label"])

        ax.set_xlabel("Training Trace Count")
        ax.set_ylabel("Learning Time (seconds)")
        ax.set_title(f"{label}\n(Mean of {repeats} runs)")
        ax.legend()
        ax.grid(True, linestyle="--", alpha=0.5)
        fig.tight_layout()

        safe_label = label.replace(" ", "_").replace("/", "-").replace("—", "-")
        out_path   = os.path.join(output_folder, f"{safe_label}{suffix}.png")
        fig.savefig(out_path, bbox_inches="tight", dpi=300)
        plt.close(fig)
        print(f"  Saved: {out_path}")


# =============================================================================
# COMBINED PLOT — all experiments
# =============================================================================

def plot_combined(results: list, output_folder: str, repeats: int,
                  show_fit: bool = False) -> None:
    suffix = "_fit" if show_fit else "_raw"
    fig, ax = plt.subplots(figsize=(12, 6))

    for result, color in zip(results, _COLORS):
        if show_fit:
            tag          = _fit_tag(result["trace_counts"], result["means"])
            legend_label = f"{result['label']} {tag}"
            fits = _fit_both(result["trace_counts"], result["means"])
            if "linear" in fits:
                ax.plot(fits["linear"]["x_dense"], fits["linear"]["y_dense"],
                        linestyle="--", linewidth=1.1, color=color, alpha=0.6)
            if "exponential" in fits:
                ax.plot(fits["exponential"]["x_dense"], fits["exponential"]["y_dense"],
                        linestyle="-.", linewidth=1.1, color=color, alpha=0.6)
        else:
            legend_label = result["label"]

        ax.errorbar(
            result["trace_counts"], result["means"], yerr=result["stds"],
            fmt="o-", color=color, ecolor=color,
            linewidth=2, markersize=4, capsize=5,
            label=legend_label,
        )

    ax.set_xlabel("Training Trace Count")
    ax.set_ylabel("Learning Time (seconds)")
    ax.set_title(f"TA Learning Scaling (Mean of {repeats} runs)")
    ax.legend(fontsize=8)
    ax.grid(True, linestyle="--", alpha=0.5)
    fig.tight_layout()

    out_path = os.path.join(output_folder, f"combined_scaling{suffix}.png")
    fig.savefig(out_path, bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"  Saved: {out_path}")


# =============================================================================
# PER-DATASET PLOTS
# =============================================================================

def plot_per_dataset(results: list, output_folder: str, repeats: int,
                     show_fit: bool = False) -> None:
    suffix   = "_fit" if show_fit else "_raw"
    datasets = sorted(set(r["dataset"] for r in results))

    for dataset in datasets:
        subset = [r for r in results if r["dataset"] == dataset]
        fig, ax = plt.subplots(figsize=(10, 5))

        for result, color in zip(subset, _COLORS):
            if show_fit:
                tag          = _fit_tag(result["trace_counts"], result["means"])
                legend_label = f"{result['method'].upper()} ({result['params']}) {tag}"
                fits = _fit_both(result["trace_counts"], result["means"])
                if "linear" in fits:
                    ax.plot(fits["linear"]["x_dense"], fits["linear"]["y_dense"],
                            linestyle="--", linewidth=1.1, color=color, alpha=0.6)
                if "exponential" in fits:
                    ax.plot(fits["exponential"]["x_dense"], fits["exponential"]["y_dense"],
                            linestyle="-.", linewidth=1.1, color=color, alpha=0.6)
            else:
                legend_label = f"{result['method'].upper()} ({result['params']})"

            ax.errorbar(
                result["trace_counts"], result["means"], yerr=result["stds"],
                fmt="o-", color=color, ecolor=color,
                linewidth=2, markersize=4, capsize=5,
                label=legend_label,
            )

        ax.set_xlabel("Training Trace Count")
        ax.set_ylabel("Learning Time (seconds)")
        ax.set_title(f"Scaling — {dataset} dataset (Mean of {repeats} runs)")
        ax.legend(fontsize=8)
        ax.grid(True, linestyle="--", alpha=0.5)
        fig.tight_layout()

        out_path = os.path.join(output_folder, f"scaling_{dataset}{suffix}.png")
        fig.savefig(out_path, bbox_inches="tight", dpi=300)
        plt.close(fig)
        print(f"  Saved: {out_path}")


# =============================================================================
# PER-METHOD PLOTS
# =============================================================================

def plot_per_method(results: list, output_folder: str, repeats: int,
                    show_fit: bool = False) -> None:
    suffix  = "_fit" if show_fit else "_raw"
    methods = sorted(set(r["method"] for r in results))

    for method in methods:
        subset = [r for r in results if r["method"] == method]
        fig, ax = plt.subplots(figsize=(10, 5))

        for result, color in zip(subset, _COLORS):
            if show_fit:
                tag          = _fit_tag(result["trace_counts"], result["means"])
                legend_label = f"{result['dataset']} {tag}"
                fits = _fit_both(result["trace_counts"], result["means"])
                if "linear" in fits:
                    ax.plot(fits["linear"]["x_dense"], fits["linear"]["y_dense"],
                            linestyle="--", linewidth=1.1, color=color, alpha=0.6)
                if "exponential" in fits:
                    ax.plot(fits["exponential"]["x_dense"], fits["exponential"]["y_dense"],
                            linestyle="-.", linewidth=1.1, color=color, alpha=0.6)
            else:
                legend_label = result["dataset"]

            ax.errorbar(
                result["trace_counts"], result["means"], yerr=result["stds"],
                fmt="o-", color=color, ecolor=color,
                linewidth=2, markersize=4, capsize=5,
                label=legend_label,
            )

        ax.set_xlabel("Training Trace Count")
        ax.set_ylabel("Learning Time (seconds)")
        ax.set_title(f"Scaling — {method.upper()} (Mean of {repeats} runs)")
        ax.legend(title="Dataset", fontsize=8)
        ax.grid(True, linestyle="--", alpha=0.5)
        fig.tight_layout()

        out_path = os.path.join(output_folder, f"scaling_{method}{suffix}.png")
        fig.savefig(out_path, bbox_inches="tight", dpi=300)
        plt.close(fig)
        print(f"  Saved: {out_path}")


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--log", default=None,
        help="Path to scaling_log.json. Defaults to the most recent run.",
    )
    parser.add_argument(
        "--out", default=None,
        help="Output folder for figures. Defaults to the same folder as the log.",
    )
    args = parser.parse_args()

    if args.log is None:
        args.log = find_latest_log()
        if args.log is None:
            print("No scaling_log.json found. Run run_scaling.py first.")
            return
        print(f"Auto-selected log: {args.log}")

    out_dir = args.out if args.out else os.path.dirname(os.path.abspath(args.log))
    os.makedirs(out_dir, exist_ok=True)

    log     = load_log(args.log)
    results = log["results"]
    repeats = log.get("repeats", "?")

    print(f"Plotting run from: {log.get('timestamp', 'unknown')}")
    print(f"Experiments:       {len(results)}")
    print(f"Output folder:     {out_dir}\n")

    for show_fit in [False, True]:
        label = "with fit" if show_fit else "raw"
        print(f"--- {label.upper()} ---")

        print(f"  Individual...")
        plot_individual(results, out_dir, repeats, show_fit=show_fit)

        print(f"  Combined...")
        plot_combined(results, out_dir, repeats, show_fit=show_fit)

        print(f"  Per dataset...")
        plot_per_dataset(results, out_dir, repeats, show_fit=show_fit)

        print(f"  Per method...")
        plot_per_method(results, out_dir, repeats, show_fit=show_fit)

        print()

    print("Done.")


if __name__ == "__main__":
    main()