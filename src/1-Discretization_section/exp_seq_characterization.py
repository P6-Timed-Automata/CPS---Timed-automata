"""
exp_seq_characterization.py
===========================
Characterize symbolic sequence properties produced by each discretization
method on a SINGLE training trace.

Compared to the original version:
  - Uses only ONE trace (TRACE_INDEX)
  - Removes transition matrix plots
  - Adds discretization comparison plots:
        raw signal + discretized signal
  - Keeps:
        * symbol frequency plots
        * run-length plots
        * summary table

Output (timestamped folder under Graphs/SeqCharacterization/):
  config.txt
  results.json
  table_summary.csv
  table_summary.png
  symbol_frequency_<method>.png
  run_length_<method>.png
  discretization_<method>.png
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
    ("naive",   {"bins": 10}),
    ("sax",     {"w": 48, "bins": 8}),
    ("persist", {"bins": 8}),
]

TRAINING_CONDITION = "clean"

# Use ONE trace only
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

def plot_symbol_frequency(method_name, freq_dict, out_path):
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
    ax.set_title(f"Symbol frequency — {method_name}")

    ax.grid(True, axis="y", linestyle="--", alpha=0.5)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")

    plt.close(fig)

    print(f"  Saved: {out_path}")


def plot_run_length_distribution(method_name, run_lengths, out_path):
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
    ax.set_title(f"Run-length distribution — {method_name}")

    ax.legend()

    ax.grid(True, axis="y", linestyle="--", alpha=0.4)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")

    plt.close(fig)

    print(f"  Saved: {out_path}")


def plot_discretization(
        method_name,
        original_trace,
        discretized_trace,
        bins,
        out_path,
):
    """
    Plot raw trace together with discretized levels.
    """

    times = np.array([t for t, _ in original_trace])
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

    ax.step(
        times,
        values,
        where="post",
        linewidth=1.8,
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
    ax.set_title(f"Discretization comparison — {method_name}")

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
    ]

    csv_path = out_dir / "table_summary.csv"

    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)

        writer.writerow(headers)

        for r in per_method_results:
            writer.writerow([
                r["method"],
                ", ".join(f"{k}={v}" for k, v in r["params"].items()),
                r["alphabet"]["n_symbols_defined"],
                r["alphabet"]["n_symbols_used"],
                f"{r['alphabet']['usage_rate']:.2f}",
                f"{r['run_length_median']:.1f}",
                f"{r['run_length_mean']:.1f}",
                r["run_length_max"],
            ])

    print(f"  Saved: {csv_path}")


# =============================================================================
# HELPERS
# =============================================================================

def _git_hash():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=ROOT
        ).decode().strip()

    except Exception:
        return "unknown"


def _save_config(out_dir, training_condition):
    lines = [
        "=" * 55,
        "Single-trace symbolic sequence characterization",
        "=" * 55,
        "",
        f"Timestamp          : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"Git hash           : {_git_hash()}",
        f"Training condition : {training_condition}",
        f"Trace index        : {TRACE_INDEX}",
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

    train_traces = [all_train_traces[TRACE_INDEX]]

    train_list = to_list_format(train_traces)

    print(
        f"Characterizing SINGLE {TRAINING_CONDITION} trace "
        f"(index={TRACE_INDEX}).\n"
    )

    _save_config(out_dir, TRAINING_CONDITION)

    log = {
        "timestamp": timestamp,
        "git_hash": _git_hash(),
        "training_condition": TRAINING_CONDITION,
        "trace_index": TRACE_INDEX,
        "methods": [],
    }

    per_method_results = []

    original_trace = train_list[0]

    for method, params in METHODS:

        print(f"--- {method} {params} ---")

        symbolic_traces, n_symbols, traces_disc, bins = _discretize(
            method,
            params,
            train_list,
        )

        alphabet_info = alphabet_usage(
            symbolic_traces,
            n_symbols,
        )

        run_lengths = run_length_distribution(symbolic_traces)

        rl_arr = np.array(run_lengths)

        result = {
            "method": method,
            "params": params,
            "alphabet": alphabet_info,
            "run_length_median": float(np.median(rl_arr)),
            "run_length_mean": float(np.mean(rl_arr)),
            "run_length_max": int(np.max(rl_arr)),
            "run_length_min": int(np.min(rl_arr)),
            "run_length_std": float(np.std(rl_arr)),
        }

        per_method_results.append(result)

        print(
            f"  Alphabet: "
            f"{alphabet_info['n_symbols_used']}/{n_symbols} used"
        )

        print(
            f"  Run length median={result['run_length_median']:.1f}, "
            f"mean={result['run_length_mean']:.1f}"
        )

        # --------------------------------------------------------------
        # PLOTS
        # --------------------------------------------------------------

        plot_symbol_frequency(
            method,
            alphabet_info["freq_raw"],
            out_dir / f"symbol_frequency_{method}.png",
            )

        plot_run_length_distribution(
            method,
            run_lengths,
            out_dir / f"run_length_{method}.png",
            )

        plot_discretization(
            method,
            original_trace,
            traces_disc[0],
            bins,
            out_dir / f"discretization_{method}.png",
            )

        print()

    log["methods"] = per_method_results

    with open(out_dir / "results.json", "w") as f:
        json.dump(log, f, indent=2)

    print(f"  Saved: {out_dir / 'results.json'}")

    save_summary_table(per_method_results, out_dir)

    print(f"\nDone. Results -> {out_dir}")