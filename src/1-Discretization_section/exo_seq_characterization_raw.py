"""
exp_seq_characterization_folder.py
==================================
Characterize symbolic sequence properties produced by each discretization
method, loading directly from a specified directory of raw CSV traces.

Output (timestamped folder under Graphs/SeqCharacterization/):
  config.txt
  results.json
  table_summary.csv
  symbol_frequency_<method>.png/.svg
  symbol_frequency_<method>_multitrace.png/.svg
  run_length_<method>.png/.svg
  discretization_<method>.png/.svg
"""
import scipy.stats as stats # <--- ADD THIS LINE
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

# Import your custom modules (ensure these are accessible in your environment)
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

# Hardcoded absolute path provided by user
DATA_DIR = Path(r"C:\Users\Jacob\Documents\GitHub\CPS---Timed-automata\Data\3-ExtractInterval\ecg\1beat")

METHODS = [
    ("naive",   {"bins": 10}),
    ("sax",     {"w": 48, "bins": 10}),
    ("persist", {"bins": 10}),
]

# Trace index used for single-trace plots (discretization, run-length,
# single-trace symbol frequency)
TRACE_INDEX = 0


# =============================================================================
# DATA LOADING & CONVERSION
# =============================================================================

def load_traces_from_folder(folder_path):
    """
    Reads all .csv files in the folder.
    Returns a list of tuples: (times_list, temps_list)
    """
    traces = []

    for file_path in folder_path.glob("*.csv"):
        times = []
        temps = []
        with open(file_path, 'r') as f:
            # Added delimiter=';' here to correctly parse your files
            reader = csv.reader(f, delimiter=';')
            header_skipped = False
            for row in reader:
                if not row:
                    continue
                try:
                    t = float(row[0])
                    v = float(row[1])
                    times.append(t)
                    temps.append(v)
                except ValueError:
                    # This will cleanly skip your header: "time_seconds;temperature"
                    if not header_skipped:
                        header_skipped = True
                    continue

        if times and temps:
            traces.append((times, temps))

    return traces


def to_list_format(traces):
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
# DISCRETIZATION
# =============================================================================

def _discretize(method, params, traces_list):
    if method == "naive":
        traces_disc, bins = equal_width_discretization(
            traces_list,
            k=params["bins"]
        )

    elif method == "sax":
        traces_disc, bins, breakpoints_z, mean_, std_ = sax_discretization_multi(
            traces_list,
            w=params["w"],
            k=params["bins"]
        )

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
def plot_qq_normality(raw_values, p_value, out_path_base):
    """
    Plots a Q-Q plot to visually assess normality against a straight line.
    """
    fig, ax = plt.subplots(figsize=(6, 6))

    # Generates the Q-Q plot
    stats.probplot(raw_values, dist="norm", plot=ax)

    ax.set_title("Q-Q Plot: Testing for Normal Distribution")
    ax.grid(True, linestyle="--", alpha=0.5)

    # Add a text box with the statistical test results
    textstr = f"D'Agostino K-squared Test\np-value: {p_value:.2e}\n"
    if p_value < 0.05:
        textstr += "Result: NOT Normal"
    else:
        textstr += "Result: Normal"

    props = dict(boxstyle='round', facecolor='white', alpha=0.8)
    ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', bbox=props)

    fig.tight_layout()

    png_path = out_path_base.with_suffix(".png")
    svg_path = out_path_base.with_suffix(".svg")

    fig.savefig(png_path, dpi=150, bbox_inches="tight")
    fig.savefig(svg_path, bbox_inches="tight")
    plt.close(fig)

    print(f"  Saved: {png_path.name} and {svg_path.name}")

def plot_symbol_frequency(method_name, params, freq_dict, out_path_base, subtitle=None):
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

    png_path = out_path_base.with_suffix(".png")
    svg_path = out_path_base.with_suffix(".svg")

    fig.savefig(png_path, dpi=150, bbox_inches="tight")
    fig.savefig(svg_path, bbox_inches="tight")
    plt.close(fig)

    print(f"  Saved: {png_path.name} and {svg_path.name}")


def plot_run_length_distribution(method_name, params, run_lengths, out_path_base, subtitle=None):
    fig, ax = plt.subplots(figsize=(8, 4))

    ax.hist(run_lengths, bins=30, alpha=0.85, edgecolor="black")
    ax.set_yscale("log")

    median = float(np.median(run_lengths))
    mean = float(np.mean(run_lengths))

    ax.axvline(median, linewidth=1.5, linestyle="--", label=f"median = {median:.1f}")
    ax.axvline(mean, linewidth=1.5, linestyle=":", label=f"mean = {mean:.1f}")

    ax.set_xlabel("Run length")
    ax.set_ylabel("Frequency (log scale)")

    title = f"Run-length distribution — {method_name} {_format_params(params)}"
    if subtitle:
        title += f"\n{subtitle}"
    ax.set_title(title)

    ax.legend()
    ax.grid(True, axis="y", linestyle="--", alpha=0.4)
    fig.tight_layout()

    png_path = out_path_base.with_suffix(".png")
    svg_path = out_path_base.with_suffix(".svg")

    fig.savefig(png_path, dpi=150, bbox_inches="tight")
    fig.savefig(svg_path, bbox_inches="tight")
    plt.close(fig)

    print(f"  Saved: {png_path.name} and {svg_path.name}")


def plot_discretization(method_name, params, original_trace, discretized_trace, bins, out_path_base, subtitle=None):
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

    png_path = out_path_base.with_suffix(".png")
    svg_path = out_path_base.with_suffix(".svg")

    fig.savefig(png_path, dpi=150, bbox_inches="tight")
    fig.savefig(svg_path, bbox_inches="tight")
    plt.close(fig)

    print(f"  Saved: {png_path.name} and {svg_path.name}")


# =============================================================================
# SUMMARY TABLE
# =============================================================================

def save_summary_table(per_method_results, out_dir):
    headers = [
        "method", "params", "alphabet_defined", "alphabet_used",
        "usage_rate", "run_length_median", "run_length_mean",
        "run_length_max", "alphabet_used_multitrace", "usage_rate_multitrace",
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

    print(f"  Saved: {csv_path.name}")


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


def _save_config(out_dir, n_multitrace):
    lines = [
        "=" * 55,
        "Symbolic sequence characterization",
        "=" * 55,
        "",
        f"Timestamp          : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"Git hash           : {_git_hash()}",
        f"Data Source        : {DATA_DIR}",
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
    print(f"  Saved: config.txt")


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    if not DATA_DIR.exists():
        print(f"Error: Directory does not exist: {DATA_DIR}")
        sys.exit(1)

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    # Outputs to Graphs/SeqCharacterization relative to your ROOT folder
    out_dir = (ROOT / "Data" / "Graphs" / "SeqCharacterization" / timestamp)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading data from: {DATA_DIR}")
    print(f"Output folder: {out_dir}\n")

    # Load from the designated folder directly
    raw_traces = load_traces_from_folder(DATA_DIR)

    if not raw_traces:
        print("No valid CSV data found in the directory.")
        sys.exit(1)

    # Multi-trace data (for aggregated symbol frequency)
    multi_list = to_list_format(raw_traces)
    n_multitrace = len(multi_list)

    # Single-trace data
    TRACE_INDEX = min(TRACE_INDEX, n_multitrace - 1) # Guard against index out of bounds
    single_list = [multi_list[TRACE_INDEX]]

    print(f"Single-trace plots: trace index {TRACE_INDEX}")
    print(f"Multi-trace plots:  all {n_multitrace} traces\n")

    _save_config(out_dir, n_multitrace)

    log = {
        "timestamp": timestamp,
        "git_hash": _git_hash(),
        "data_directory": str(DATA_DIR),
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
            symbolic_traces_single, n_symbols_single,
        )

        run_lengths = run_length_distribution(symbolic_traces_single)
        rl_arr = np.array(run_lengths)

        # -----------------------------------------------------------------
        # MULTI-TRACE PROCESSING
        # -----------------------------------------------------------------
        (
            symbolic_traces_multi,
            n_symbols_multi,
            _traces_disc_multi,
            _bins_multi,
        ) = _discretize(method, params, multi_list)

        alphabet_info_multi = alphabet_usage(
            symbolic_traces_multi, n_symbols_multi,
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
        plot_symbol_frequency(
            method, params, alphabet_info_single["freq_raw"],
            out_dir / f"symbol_frequency_{method}",
            subtitle=f"single trace (index {TRACE_INDEX})",
            )

        plot_symbol_frequency(
            method, params, alphabet_info_multi["freq_raw"],
            out_dir / f"symbol_frequency_{method}_multitrace",
            subtitle=f"aggregated across {n_multitrace} traces",
            )

        plot_run_length_distribution(
            method, params, run_lengths,
            out_dir / f"run_length_{method}",
            subtitle=f"single trace (index {TRACE_INDEX})",
            )

        plot_discretization(
            method, params, original_trace, traces_disc_single[0], bins_single,
            out_dir / f"discretization_{method}",
            subtitle=f"single trace (index {TRACE_INDEX})",
            )
        print()

    log["methods"] = per_method_results

    with open(out_dir / "results.json", "w") as f:
        json.dump(log, f, indent=2)
    print(f"  Saved: results.json")

    save_summary_table(per_method_results, out_dir)
    print(f"\nDone. Results -> {out_dir}")

    _save_config(out_dir, n_multitrace) # This line already exists

    # -----------------------------------------------------------------
    # NORMALITY TEST (RAW DATA)
    # -----------------------------------------------------------------
    print("--- Normality Test (Raw Data) ---")
    # Extract all raw temperature values into a single flat array
    all_raw_values = np.array([v for trace in multi_list for v, _ in trace])

    # Calculate D'Agostino's K-squared test
    # If the dataset is perfectly flat/uniform, normaltest can throw a runtime warning,
    # but for sensor data, it will calculate smoothly.
    k2, p_val = stats.normaltest(all_raw_values)
    is_normal = bool(p_val >= 0.05)

    print(f"  p-value: {p_val:.2e} -> {'Normal' if is_normal else 'NOT Normal'}\n")

    # Generate the Q-Q plot
    plot_qq_normality(all_raw_values, p_val, out_dir / "normality_qq_plot")

    log = {
        "timestamp": timestamp,
        "git_hash": _git_hash(),
        "data_directory": str(DATA_DIR),
        "single_trace_index": TRACE_INDEX,
        "n_multitrace": n_multitrace,
        "raw_data_normality": {                 # <--- Added to your JSON log
            "p_value": float(p_val),
            "is_normal": is_normal
        },
        "methods": [],
    }