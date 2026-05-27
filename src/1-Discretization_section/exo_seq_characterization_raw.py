"""
exp_seq_characterization_folder.py
==================================
Folder-input variant of exp_seq_characterization. Loads raw CSV traces from
a hardcoded directory (used for the ECG dataset) and produces the same
symbol-frequency and discretization-overlay plots used in the real-data
section, plus a D'Agostino K² normality test on the raw values with an
accompanying Q-Q plot (used to justify the SAX normality discussion).

Output: Data/Graphs/SeqCharacterization/<timestamp>/
  config.txt
  results.json
  symbol_frequency_<method>.png/.svg             (single trace)
  symbol_frequency_<method>_multitrace.png/.svg  (all traces)
  discretization_<method>.png/.svg               (single trace)
  normality_qq_plot.png/.svg                     (raw values vs. normal)
"""

import csv
import json
import string
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from Discretization.naive import equal_width_discretization
from Discretization.sax import sax_discretization_multi
from Discretization.persist import (
    Persist,
    get_best_bins,
    discretize_traces_with_bins,
    flatten_traces_to_ts,
)


# =============================================================================
# CONFIG
# =============================================================================

DATA_DIR = Path(
    r"C:\Users\Jacob\Documents\GitHub\CPS---Timed-automata"
    r"\Data\3-ExtractInterval\ecg\1beat"
)

METHODS = [
    ("naive",   {"bins": 10}),
    ("sax",     {"w": 48, "bins": 10}),
    ("persist", {"bins": 10}),
]

# Index of the trace used for single-trace plots.
TRACE_INDEX = 0


# =============================================================================
# DATA LOADING & CONVERSION
# =============================================================================

def load_traces_from_folder(folder_path):
    """Read all *.csv files in folder_path. Files use ';' as delimiter with
    a `time_seconds;temperature` header line. Returns list of (times, temps).
    Non-numeric rows (the header) are skipped silently."""
    traces = []
    for file_path in folder_path.glob("*.csv"):
        times, temps = [], []
        with open(file_path, "r") as f:
            reader = csv.reader(f, delimiter=";")
            for row in reader:
                if not row:
                    continue
                try:
                    times.append(float(row[0]))
                    temps.append(float(row[1]))
                except ValueError:
                    continue  # header row
        if times and temps:
            traces.append((times, temps))
    return traces


def to_list_format(traces):
    """Convert (times, temps) arrays into [(value, time), ...] format that
    the Discretization functions expect."""
    return [
        [(float(v), int(t)) for t, v in zip(times, temps)]
        for times, temps in traces
    ]


# =============================================================================
# HELPERS
# =============================================================================

def _format_params(params):
    if not params:
        return ""
    return "(" + ", ".join(f"{k}={v}" for k, v in params.items()) + ")"


# =============================================================================
# DISCRETIZATION ROUTING
# =============================================================================

def _discretize(method, params, traces_list):
    if method == "naive":
        traces_disc, bins = equal_width_discretization(traces_list, k=params["bins"])

    elif method == "sax":
        # sax_discretization_multi returns (traces, bins_orig_space, breakpoints_z, mean_, std_)
        traces_disc, bins, *_ = sax_discretization_multi(
            traces_list, w=params["w"], k=params["bins"]
        )

    elif method == "persist":
        ts = flatten_traces_to_ts(traces_list)
        persist_obj = Persist(
            ts,
            break_min=params["bins"], break_max=params["bins"],
            skip=np.array([4, 4]),
        )
        bins = get_best_bins(persist_obj, ts)
        traces_disc = discretize_traces_with_bins(traces_list, bins)

    else:
        raise ValueError(f"Unknown method: {method}")

    n_symbols = len(bins) - 1
    alphabet = list(string.ascii_lowercase)
    symbolic_traces = [
        [alphabet[int(label)] for label, _ in trace]
        for trace in traces_disc
    ]
    return symbolic_traces, n_symbols, traces_disc, bins


# =============================================================================
# ALPHABET USAGE
# =============================================================================

def alphabet_usage(symbolic_traces, n_symbols):
    all_symbols = [s for trace in symbolic_traces for s in trace]
    counter = Counter(all_symbols)
    alphabet_letters = list(string.ascii_lowercase)[:n_symbols]
    full_freq = {letter: counter.get(letter, 0) for letter in alphabet_letters}
    return {
        "n_symbols_defined": n_symbols,
        "n_symbols_used":    len(counter),
        "usage_rate":        len(counter) / n_symbols if n_symbols > 0 else 0.0,
        "freq_raw":          full_freq,
    }


# =============================================================================
# PLOTS
# =============================================================================

def plot_symbol_frequency(method_name, params, freq_dict, out_path_base,
                          subtitle=None):
    """Bar chart of symbol frequency. Saves both .png and .svg."""
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
                ha="center", va="bottom", fontsize=8,
            )
    ax.set_xlabel("Symbol")
    ax.set_ylabel("Frequency")

    title = f"Symbol frequency — {method_name} {_format_params(params)}"
    if subtitle:
        title += f"\n{subtitle}"
    ax.set_title(title)
    ax.grid(True, axis="y", linestyle="--", alpha=0.5)

    fig.tight_layout()
    fig.savefig(out_path_base.with_suffix(".png"), dpi=150, bbox_inches="tight")
    fig.savefig(out_path_base.with_suffix(".svg"), bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path_base.name}.png/.svg")


def plot_discretization(method_name, params,
                        original_trace, discretized_trace, bins,
                        out_path_base, subtitle=None):
    """Raw trace overlaid with discretized step function and bin edges."""
    times = np.array([t for _, t in original_trace])
    values = np.array([v for v, _ in original_trace])

    disc_times = [t for _, t in discretized_trace]
    disc_values = [
        (bins[int(label)] + bins[int(label) + 1]) / 2
        for label, _ in discretized_trace
    ]

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(times, values, linewidth=1.0, alpha=0.6, label="Raw trace")
    ax.step(disc_times, disc_values, where="post", linewidth=2, label="Discretized")
    for b in bins[1:-1]:
        ax.axhline(b, linestyle="--", linewidth=0.8, alpha=0.5)

    ax.set_xlabel("Time")
    ax.set_ylabel("Temperature")

    title = f"Discretization comparison — {method_name} {_format_params(params)}"
    if subtitle:
        title += f"\n{subtitle}"
    ax.set_title(title)
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.4)

    fig.tight_layout()
    fig.savefig(out_path_base.with_suffix(".png"), dpi=150, bbox_inches="tight")
    fig.savefig(out_path_base.with_suffix(".svg"), bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path_base.name}.png/.svg")


def plot_qq_normality(raw_values, p_value, out_path_base):
    """Q-Q plot of raw values against a normal distribution, with the
    D'Agostino K² result rendered in a text box."""
    fig, ax = plt.subplots(figsize=(6, 6))
    stats.probplot(raw_values, dist="norm", plot=ax)
    ax.set_title("Q-Q plot: raw values vs. normal distribution")
    ax.grid(True, linestyle="--", alpha=0.5)

    result_label = "NOT normal" if p_value < 0.05 else "normal"
    textstr = (f"D'Agostino K² test\n"
               f"p-value: {p_value:.2e}\n"
               f"Result: {result_label}")
    ax.text(
        0.05, 0.95, textstr,
        transform=ax.transAxes, fontsize=10, verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
    )

    fig.tight_layout()
    fig.savefig(out_path_base.with_suffix(".png"), dpi=150, bbox_inches="tight")
    fig.savefig(out_path_base.with_suffix(".svg"), bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path_base.name}.png/.svg")


# =============================================================================
# CONFIG FILE
# =============================================================================

def _save_config(out_dir, n_multitrace, trace_index):
    lines = [
        "=" * 55,
        "Symbolic sequence characterization (folder input)",
        "=" * 55,
        "",
        f"Timestamp          : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"Data source        : {DATA_DIR}",
        f"Single-trace index : {trace_index}",
        f"Multi-trace count  : {n_multitrace}",
        "",
        "--- Methods ---",
    ]
    for method, params in METHODS:
        param_str = ", ".join(f"{k}={v}" for k, v in params.items())
        lines.append(f"  {method:8s}: {param_str}")
    lines += ["", "--- Output folder ---", f"  {out_dir}", "", "=" * 55]
    (out_dir / "config.txt").write_text("\n".join(lines))
    print(f"  Saved: config.txt")


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    if not DATA_DIR.exists():
        print(f"Error: Directory does not exist: {DATA_DIR}")
        sys.exit(1)

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    out_dir = ROOT / "Data" / "Graphs" / "SeqCharacterization" / timestamp
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Loading data from: {DATA_DIR}")
    print(f"Output folder: {out_dir}\n")

    raw_traces = load_traces_from_folder(DATA_DIR)
    if not raw_traces:
        print("No valid CSV data found in the directory.")
        sys.exit(1)

    multi_list = to_list_format(raw_traces)
    n_multitrace = len(multi_list)
    trace_index = min(TRACE_INDEX, n_multitrace - 1)   # guard against OOB
    single_list = [multi_list[trace_index]]

    print(f"Single-trace plots: trace index {trace_index}")
    print(f"Multi-trace plots:  all {n_multitrace} traces\n")

    _save_config(out_dir, n_multitrace, trace_index)

    log = {
        "timestamp":          timestamp,
        "data_directory":     str(DATA_DIR),
        "single_trace_index": trace_index,
        "n_multitrace":       n_multitrace,
        "methods":            [],
    }

    # -------------------------------------------------------------------------
    # Per-method analysis and plots
    # -------------------------------------------------------------------------
    original_trace = single_list[0]

    for method, params in METHODS:
        print(f"--- {method} {params} ---")

        symbolic_single, n_symbols_single, traces_disc_single, bins_single = \
            _discretize(method, params, single_list)
        alphabet_single = alphabet_usage(symbolic_single, n_symbols_single)

        symbolic_multi, n_symbols_multi, _, _ = \
            _discretize(method, params, multi_list)
        alphabet_multi = alphabet_usage(symbolic_multi, n_symbols_multi)

        log["methods"].append({
            "method":              method,
            "params":              params,
            "alphabet":            alphabet_single,
            "alphabet_multitrace": alphabet_multi,
        })

        print(f"  Single-trace alphabet: "
              f"{alphabet_single['n_symbols_used']}/{n_symbols_single} used")
        print(f"  Multi-trace alphabet : "
              f"{alphabet_multi['n_symbols_used']}/{n_symbols_multi} used "
              f"(across {n_multitrace} traces)")

        plot_symbol_frequency(
            method, params, alphabet_single["freq_raw"],
            out_dir / f"symbol_frequency_{method}",
            subtitle=f"single trace (index {trace_index})",
        )
        plot_symbol_frequency(
            method, params, alphabet_multi["freq_raw"],
            out_dir / f"symbol_frequency_{method}_multitrace",
            subtitle=f"aggregated across {n_multitrace} traces",
        )
        plot_discretization(
            method, params,
            original_trace, traces_disc_single[0], bins_single,
            out_dir / f"discretization_{method}",
            subtitle=f"single trace (index {trace_index})",
        )
        print()

    # -------------------------------------------------------------------------
    # Normality test on raw values (used to justify SAX normality discussion)
    # -------------------------------------------------------------------------
    print("--- Normality test (raw data) ---")
    all_raw_values = np.array([v for trace in multi_list for v, _ in trace])
    k2, p_val = stats.normaltest(all_raw_values)
    is_normal = bool(p_val >= 0.05)
    print(f"  D'Agostino K² p-value: {p_val:.2e} "
          f"-> {'normal' if is_normal else 'NOT normal'}\n")

    plot_qq_normality(all_raw_values, p_val, out_dir / "normality_qq_plot")

    log["raw_data_normality"] = {
        "p_value":   float(p_val),
        "k2":        float(k2),
        "is_normal": is_normal,
    }

    # -------------------------------------------------------------------------
    # Save log and finish
    # -------------------------------------------------------------------------
    with open(out_dir / "results.json", "w") as f:
        json.dump(log, f, indent=2)
    print(f"  Saved: results.json")
    print(f"\nDone. Results -> {out_dir}")