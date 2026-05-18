"""
exp_seq_characterization.py
===========================
Characterize symbolic sequence properties produced by each discretization
method.

Outputs both:
  - Single-trace plots: based on one specific trace (TRACE_INDEX) for visual
    inspection of the discretization process.
  - Multi-trace symbol frequency plots: aggregate alphabet usage across the
    full training set, showing how each method distributes signal across bins
    at scale.

Output (timestamped folder under Graphs/SeqCharacterization/):
  config.txt
  results.json
  table_summary.csv
  symbol_frequency_<method>.png             — single trace
  symbol_frequency_<method>_multitrace.png  — aggregated across all traces
  run_length_<method>.png                   — single trace
  discretization_<method>.png               — single trace
"""

import csv
import json
import string
import subprocess
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from Generate_data import load_all_data
from Discretization.naive import equal_width_discretization
from Discretization.sax import (
    sax_discretization_multi,
    sax_bins_in_original_space,
)
from Discretization.persist import (
    Persist,
    get_best_bins,
    discretize_traces_with_bins,
    flatten_traces_to_ts,
)


# =============================================================================
# CONFIG
# =============================================================================

METHODS = [
    ("naive",   {"bins": 15}),
    ("sax",     {"w": 48, "bins": 15}),
    ("persist", {"bins": 16}),
]

TRAINING_CONDITION = "clean"

# Trace index used for single-trace plots (discretization, run-length,
# single-trace symbol frequency)
TRACE_INDEX = 0


# =============================================================================
# DATA FORMAT CONVERSION
# =============================================================================

def to_list_format(traces):
    return [
        [(float(v), int(t)) for t, v in zip(times, temps)]
        for times, temps in traces
    ]


# =============================================================================
# HELPERS
# =============================================================================

def _format_params(params):
    """Format params dict as a short '(key=val, key=val)' string for titles."""
    if not params:
        return ""
    return "(" + ", ".join(f"{k}={v}" for k, v in params.items()) + ")"


# =============================================================================
# DISCRETIZATION
# =============================================================================

def _discretize(method, params, traces_list):
    """
    Returns
    -------
    symbolic_traces
    n_symbols
    traces_disc
    bins
    """
    if method == "naive":
        traces_disc, bins = equal_width_discretization(
            traces_list,
            k=params["bins"]
        )

    elif method == "sax":
        traces_disc, bins_z, mean_, std_ = sax_discretization_multi(
            traces_list,
            w=params["w"],
            k=params["bins"]
        )
        bins = sax_bins_in_original_space(bins_z, mean_, std_)

    elif method == "persist":
        ts = flatten_traces_to_ts(traces_list)

        persist_obj = Persist(
            ts,
            break_min=params["bins"],
            break_max=params["bins"],
            skip=np.array([4, 4]),
        )

        bins = get_best_bins(persist_obj, ts)

        traces_disc = discretize_traces_with_bins(
            traces_list,
            bins
        )

    else:
        raise ValueError(f"Unknown method: {method}")

    n_symbols = len(bins) - 1

    alphabet = list(string.ascii_lowercase)

    symbolic_traces = []
    for trace in traces_disc:
        letters = [alphabet[int(l)] for l, _ in trace]
        symbolic_traces.append(letters)

    return symbolic_traces, n_symbols, traces_disc, bins


# =============================================================================
# METRICS
# =============================================================================

def alphabet_usage(symbolic_traces, n_symbols):
    all_symbols = [s for trace in symbolic_traces for s in trace]

    counter = Counter(all_symbols)

    used = len(counter)

    alphabet_letters = list(string.ascii_lowercase)[:n_symbols]

    full_freq = {
        letter: counter.get(letter, 0)
        for letter in alphabet_letters
    }

    total = sum(full_freq.values())

    full_freq_normalized = {
        letter: (count / total if total > 0 else 0.0)
        for letter, count in full_freq.items()
    }

    return {
        "n_symbols_defined": n_symbols,
        "n_symbols_used": used,
        "usage_rate": used / n_symbols if n_symbols > 0 else 0.0,
        "freq_raw": full_freq,
        "freq_normalized": full_freq_normalized,
    }


def run_length_distribution(symbolic_traces):
    run_lengths = []

    for trace in symbolic_traces:
        if not trace:
            continue

        current_symbol = trace[0]
        current_length = 1

        for sym in trace[1:]:
            if sym == current_symbol:
                current_length += 1
            else:
                run_lengths.append(current_length)
                current_symbol = sym
                current_length = 1

        run_lengths.append(current_length)

    return run_lengths


# =============================================================================
# PLOTS
# =============================================================================

def plot_symbol_frequency(method_name, params, freq_dict, out_path,
                          subtitle=None):
    """
    Plot symbol frequency as a bar chart.

    Parameters
    ----------
    method_name : str
        For the title.
    params : dict
        Method parameters, appended to the title.
    freq_dict : dict[str, int]
        Symbol -> count.
    out_path : Path or str
    subtitle : str, optional
        Extra context line under the main title (e.g. "1 trace" or "20 traces").
    """
    symbols = sorted(freq_dict.keys())
    counts = [freq_dict[s] for s in symbols]

    fig, ax = plt.subplots(figsize=(8, 4))

    bars = ax.bar(symbols, counts, alpha=0.85)

    for bar, c in zip(bars, counts):
        if c > 0:
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.5,
                str(c),
                ha="center",
                va="bottom",
                fontsize=8,
                )

    ax.set_xlabel("Symbol")
    ax.set_ylabel("Frequency")

    title = f"Symbol frequency — {method_name} {_format_params(params)}"
    if subtitle:
        title += f"\n{subtitle}"
    ax.set_title(title)

    ax.grid(True, axis="y", linestyle="--", alpha=0.5)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")

    plt.close(fig)

    print(f"  Saved: {out_path}")


def plot_run_length_distribution(method_name, params, run_lengths, out_path,
                                 subtitle=None):
    fig, ax = plt.subplots(figsize=(8, 4))

    ax.hist(
        run_lengths,
        bins=30,
        alpha=0.85,
        edgecolor="black",
    )

    ax.set_yscale("log")

    median = float(np.median(run_lengths))
    mean = float(np.mean(run_lengths))

    ax.axvline(
        median,
        linewidth=1.5,
        linestyle="--",
        label=f"median = {median:.1f}",
    )

    ax.axvline(
        mean,
        linewidth=1.5,
        linestyle=":",
        label=f"mean = {mean:.1f}",
    )

    ax.set_xlabel("Run length")
    ax.set_ylabel("Frequency (log scale)")

    title = f"Run-length distribution — {method_name} {_format_params(params)}"
    if subtitle:
        title += f"\n{subtitle}"
    ax.set_title(title)

    ax.legend()
    ax.grid(True, axis="y", linestyle="--", alpha=0.4)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")

    plt.close(fig)

    print(f"  Saved: {out_path}")


def plot_discretization(
        method_name, params,
        original_trace,
        discretized_trace,
        bins,
        out_path,
        subtitle=None,
):
    """
    Plot raw trace together with discretized levels.

    original_trace : list of (value, time) tuples — value first, time second.
    discretized_trace : list of (label, time) tuples.
    """
    times = np.array([t for _, t in original_trace])
    values = np.array([v for v, _ in original_trace])

    disc_values = []
    disc_times = []

    for label, t in discretized_trace:
        label = int(label)

        low = bins[label]
        high = bins[label + 1]

        midpoint = (low + high) / 2

        disc_values.append(midpoint)
        disc_times.append(t)

    fig, ax = plt.subplots(figsize=(12, 4))

    ax.plot(
        times,
        values,
        linewidth=1.0,
        alpha=0.6,
        label="Raw trace",
    )

    ax.step(
        disc_times,
        disc_values,
        where="post",
        linewidth=2,
        label="Discretized",
    )

    for b in bins[1:-1]:
        ax.axhline(
            b,
            linestyle="--",
            linewidth=0.8,
            alpha=0.5,
        )

    ax.set_xlabel("Time")
    ax.set_ylabel("Temperature")

    title = f"Discretization comparison — {method_name} {_format_params(params)}"
    if subtitle:
        title += f"\n{subtitle}"
    ax.set_title(title)

    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.4)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")

    plt.close(fig)

    print(f"  Saved: {out_path}")


# =============================================================================
# SUMMARY TABLE
# =============================================================================

def save_summary_table(per_method_results, out_dir):
    headers = [
        "method",
        "params",
        "alphabet_defined",
        "alphabet_used",
        "usage_rate",
        "run_length_median",
        "run_length_mean",
        "run_length_max",
        "alphabet_used_multitrace",
        "usage_rate_multitrace",
    ]

    csv_path = out_dir / "table_summary.csv"

    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)

        writer.writerow(headers)

        for r in per_method_results:
            mt = r.get("alphabet_multitrace", {})
            writer.writerow([
                r["method"],
                ", ".join(f"{k}={v}" for k, v in r["params"].items()),
                r["alphabet"]["n_symbols_defined"],
                r["alphabet"]["n_symbols_used"],
                f"{r['alphabet']['usage_rate']:.2f}",
                f"{r['run_length_median']:.1f}",
                f"{r['run_length_mean']:.1f}",
                r["run_length_max"],
                mt.get("n_symbols_used", "-"),
                f"{mt['usage_rate']:.2f}" if mt else "-",
            ])

    print(f"  Saved: {csv_path}")


# =============================================================================
# CONFIG / GIT
# =============================================================================

def _git_hash():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=ROOT
        ).decode().strip()

    except Exception:
        return "unknown"


def _save_config(out_dir, training_condition, n_multitrace):
    lines = [
        "=" * 55,
        "Symbolic sequence characterization",
        "=" * 55,
        "",
        f"Timestamp          : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"Git hash           : {_git_hash()}",
        f"Training condition : {training_condition}",
        f"Single-trace index : {TRACE_INDEX}",
        f"Multi-trace count  : {n_multitrace}",
        "",
        "--- Methods ---",
        ]

    for method, params in METHODS:
        param_str = ", ".join(f"{k}={v}" for k, v in params.items())
        lines.append(f"  {method:8s}: {param_str}")

    lines += [
        "",
        "--- Output folder ---",
        f"  {out_dir}",
        "",
        "=" * 55,
        ]

    (out_dir / "config.txt").write_text("\n".join(lines))

    print(f"  Saved: {out_dir / 'config.txt'}")


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    out_dir = (
            ROOT
            / "Data"
            / "Graphs"
            / "SeqCharacterization"
            / timestamp
    )

    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Output folder: {out_dir}\n")

    data = load_all_data()

    all_train_traces = data[f"{TRAINING_CONDITION}_train"]

    # Single-trace data (for discretization comparison + single-trace freq plot)
    single_traces = [all_train_traces[TRACE_INDEX]]
    single_list = to_list_format(single_traces)

    # Multi-trace data (for aggregated symbol frequency)
    multi_list = to_list_format(all_train_traces)

    n_multitrace = len(multi_list)

    print(
        f"Single-trace plots: {TRAINING_CONDITION} trace index {TRACE_INDEX}"
    )
    print(
        f"Multi-trace plots:  all {n_multitrace} {TRAINING_CONDITION} "
        f"training traces\n"
    )

    _save_config(out_dir, TRAINING_CONDITION, n_multitrace)

    log = {
        "timestamp": timestamp,
        "git_hash": _git_hash(),
        "training_condition": TRAINING_CONDITION,
        "single_trace_index": TRACE_INDEX,
        "n_multitrace": n_multitrace,
        "methods": [],
    }

    per_method_results = []

    original_trace = single_list[0]

    for method, params in METHODS:

        print(f"--- {method} {params} ---")

        # -----------------------------------------------------------------
        # SINGLE-TRACE PROCESSING
        # -----------------------------------------------------------------
        (
            symbolic_traces_single,
            n_symbols_single,
            traces_disc_single,
            bins_single,
        ) = _discretize(method, params, single_list)

        alphabet_info_single = alphabet_usage(
            symbolic_traces_single,
            n_symbols_single,
        )

        run_lengths = run_length_distribution(symbolic_traces_single)
        rl_arr = np.array(run_lengths)

        # -----------------------------------------------------------------
        # MULTI-TRACE PROCESSING (for aggregated symbol frequency)
        # -----------------------------------------------------------------
        (
            symbolic_traces_multi,
            n_symbols_multi,
            _traces_disc_multi,
            _bins_multi,
        ) = _discretize(method, params, multi_list)

        alphabet_info_multi = alphabet_usage(
            symbolic_traces_multi,
            n_symbols_multi,
        )

        result = {
            "method": method,
            "params": params,
            "alphabet": alphabet_info_single,
            "alphabet_multitrace": alphabet_info_multi,
            "run_length_median": float(np.median(rl_arr)),
            "run_length_mean": float(np.mean(rl_arr)),
            "run_length_max": int(np.max(rl_arr)),
            "run_length_min": int(np.min(rl_arr)),
            "run_length_std": float(np.std(rl_arr)),
        }

        per_method_results.append(result)

        print(
            f"  Single-trace alphabet : "
            f"{alphabet_info_single['n_symbols_used']}/{n_symbols_single} used"
        )

        print(
            f"  Multi-trace alphabet  : "
            f"{alphabet_info_multi['n_symbols_used']}/{n_symbols_multi} used "
            f"(across {n_multitrace} traces)"
        )

        print(
            f"  Run length median={result['run_length_median']:.1f}, "
            f"mean={result['run_length_mean']:.1f}"
        )

        # -----------------------------------------------------------------
        # PLOTS
        # -----------------------------------------------------------------

        # Single-trace symbol frequency
        plot_symbol_frequency(
            method,
            params,
            alphabet_info_single["freq_raw"],
            out_dir / f"symbol_frequency_{method}.png",
            subtitle=f"single trace (index {TRACE_INDEX})",
            )

        # Multi-trace symbol frequency
        plot_symbol_frequency(
            method,
            params,
            alphabet_info_multi["freq_raw"],
            out_dir / f"symbol_frequency_{method}_multitrace.png",
            subtitle=f"aggregated across {n_multitrace} {TRAINING_CONDITION} training traces",
            )

        # Single-trace run-length distribution
        plot_run_length_distribution(
            method,
            params,
            run_lengths,
            out_dir / f"run_length_{method}.png",
            subtitle=f"single trace (index {TRACE_INDEX})",
            )

        # Single-trace discretization comparison
        plot_discretization(
            method,
            params,
            original_trace,
            traces_disc_single[0],
            bins_single,
            out_dir / f"discretization_{method}.png",
            subtitle=f"single trace (index {TRACE_INDEX})",
            )

        print()

    log["methods"] = per_method_results

    with open(out_dir / "results.json", "w") as f:
        json.dump(log, f, indent=2)

    print(f"  Saved: {out_dir / 'results.json'}")

    save_summary_table(per_method_results, out_dir)

    print(f"\nDone. Results -> {out_dir}")