"""
exp_seq_characterization.py
===========================
Characterize symbolic sequence properties produced by each discretization
method on the training data.

This is the mechanism-level analysis that links RQ1 (discretization
fidelity) and RQ2 (TA size and cost):

  - How many distinct symbols actually appear in training data?
  - How is symbol frequency distributed?
  - How long do symbols persist before changing (run-length distribution)?
  - How many distinct symbol→symbol transitions exist?

These properties mediate the discretization→TAG relationship. Reporting
them lets you explain *why* method X produces larger or smaller TAs than
method Y, beyond just reporting that it does.

Run on the training data the experiments use (clean or noisy) with the
final chosen parameter for each method.

Output (timestamped folder under Graphs/SeqCharacterization/):
  config.txt
  results.json
  table_summary.csv
  table_summary.png
  symbol_frequency_<method>.png
  run_length_<method>.png
  transition_matrix_<method>.png
"""

import csv
import json
import string
import subprocess
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from Generate_data import load_all_data
from Discretization.naive import equal_width_discretization
from Discretization.sax import sax_discretization_multi, sax_bins_in_original_space
from Discretization.persist import (
    Persist, get_best_bins, discretize_traces_with_bins, flatten_traces_to_ts,
)


# =============================================================================
# CONFIG — match exp_51_52 final choices
# =============================================================================

METHODS = [
    ("naive",   {"bins": 10}),
    ("sax",     {"w": 48,  "bins": 8}),
    ("persist", {"bins": 8}),
]

# Which training condition to characterize. Run twice if you want both.
TRAINING_CONDITION = "clean"   # or "noisy"


# =============================================================================
# DATA FORMAT CONVERSION
# =============================================================================

def to_list_format(traces):
    """Convert [(times, temps), ...] → [[(temp, time), ...], ...]."""
    return [
        [(float(v), int(t)) for t, v in zip(times, temps)]
        for times, temps in traces
    ]


# =============================================================================
# DISCRETIZATION ROUTING
# =============================================================================

def _discretize(method, params, traces_list):
    """Return symbolic traces as lists of letters per trace."""
    if method == "naive":
        traces_disc, bins = equal_width_discretization(traces_list, k=params["bins"])
    elif method == "sax":
        traces_disc, bins_z, mean_, std_ = sax_discretization_multi(
            traces_list, w=params["w"], k=params["bins"]
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
        traces_disc = discretize_traces_with_bins(traces_list, bins)
    else:
        raise ValueError(f"Unknown method: {method}")

    n_symbols = len(bins) - 1
    if method == "persist":
        # get_best_bins adds 2 outer edges; subtract to recover chosen k
        # But the label indices in traces_disc are already in [0, n_symbols-1].
        pass

    # Map int labels → letters
    alphabet = list(string.ascii_lowercase)
    symbolic_traces = []
    for trace in traces_disc:
        letters = [alphabet[int(l)] for l, _ in trace]
        symbolic_traces.append(letters)

    return symbolic_traces, n_symbols


# =============================================================================
# CHARACTERIZATION METRICS
# =============================================================================

def alphabet_usage(symbolic_traces, n_symbols):
    """Count distinct symbols actually appearing, and per-symbol frequency."""
    all_symbols = [s for trace in symbolic_traces for s in trace]
    counter = Counter(all_symbols)
    used = len(counter)
    alphabet_letters = list(string.ascii_lowercase)[:n_symbols]
    full_freq = {letter: counter.get(letter, 0) for letter in alphabet_letters}
    total = sum(full_freq.values())
    full_freq_normalized = {
        letter: (count / total if total > 0 else 0.0)
        for letter, count in full_freq.items()
    }
    return {
        "n_symbols_defined": n_symbols,
        "n_symbols_used":    used,
        "usage_rate":        used / n_symbols if n_symbols > 0 else 0.0,
        "freq_raw":          full_freq,
        "freq_normalized":   full_freq_normalized,
    }


def run_length_distribution(symbolic_traces):
    """
    For each trace, compute lengths of consecutive same-symbol runs.
    Returns one big list of run lengths across all traces.
    """
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


def transition_matrix(symbolic_traces, n_symbols):
    """
    Count symbol→symbol transitions (different-symbol only — same-symbol
    'transitions' are intra-run dwell).
    Returns a (n_symbols x n_symbols) matrix of counts.
    """
    alphabet = list(string.ascii_lowercase)[:n_symbols]
    sym_to_idx = {s: i for i, s in enumerate(alphabet)}
    mat = np.zeros((n_symbols, n_symbols), dtype=int)

    for trace in symbolic_traces:
        prev = None
        for sym in trace:
            if prev is not None and prev != sym:
                if prev in sym_to_idx and sym in sym_to_idx:
                    mat[sym_to_idx[prev], sym_to_idx[sym]] += 1
            prev = sym

    return mat, alphabet


def transition_entropy(matrix):
    """
    Shannon entropy of the transition distribution. Higher = more
    transition diversity (TAG sees more distinct k-grams).
    """
    total = matrix.sum()
    if total == 0:
        return 0.0
    probs = matrix.flatten() / total
    probs = probs[probs > 0]
    return float(-np.sum(probs * np.log2(probs)))


# =============================================================================
# PLOTS
# =============================================================================

def plot_symbol_frequency(method_name, freq_dict, out_path):
    """Bar chart of symbol frequency. Sorted by symbol letter."""
    symbols = sorted(freq_dict.keys())
    counts = [freq_dict[s] for s in symbols]

    fig, ax = plt.subplots(figsize=(8, 4))
    bars = ax.bar(symbols, counts, color="steelblue", alpha=0.85)
    for bar, c in zip(bars, counts):
        if c > 0:
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.5,
                    str(c), ha="center", va="bottom", fontsize=8)
    ax.set_xlabel("Symbol")
    ax.set_ylabel("Frequency")
    ax.set_title(f"Symbol frequency — {method_name}")
    ax.grid(True, axis="y", linestyle="--", alpha=0.5)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


def plot_run_length_distribution(method_name, run_lengths, out_path):
    """Histogram of run lengths on log y-axis (typically heavy-tailed)."""
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(run_lengths, bins=30, color="steelblue", alpha=0.85, edgecolor="black")
    ax.set_yscale("log")
    ax.set_xlabel("Run length (consecutive samples of same symbol)")
    ax.set_ylabel("Frequency (log scale)")
    median = float(np.median(run_lengths))
    mean = float(np.mean(run_lengths))
    ax.axvline(median, color="firebrick", linewidth=1.5, linestyle="--",
               label=f"median = {median:.1f}")
    ax.axvline(mean, color="darkgreen", linewidth=1.5, linestyle=":",
               label=f"mean = {mean:.1f}")
    ax.legend()
    ax.set_title(f"Run-length distribution — {method_name}")
    ax.grid(True, axis="y", linestyle="--", alpha=0.4)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


def plot_transition_matrix(method_name, matrix, alphabet, out_path):
    """Heatmap of symbol→symbol transition counts."""
    fig, ax = plt.subplots(figsize=(8, 7))
    # Mask diagonal (zero by construction) so the colorscale isn't dominated
    masked = matrix.astype(float).copy()
    np.fill_diagonal(masked, np.nan)
    im = ax.imshow(masked, cmap="viridis", aspect="auto")
    ax.set_xticks(range(len(alphabet)))
    ax.set_xticklabels(alphabet)
    ax.set_yticks(range(len(alphabet)))
    ax.set_yticklabels(alphabet)
    ax.set_xlabel("To symbol")
    ax.set_ylabel("From symbol")
    ax.set_title(f"Transitions — {method_name}\n(diagonal omitted)")

    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            if i != j and matrix[i, j] > 0:
                ax.text(j, i, str(matrix[i, j]),
                        ha="center", va="center", fontsize=8,
                        color="white" if matrix[i, j] > matrix.max() / 2 else "black")

    plt.colorbar(im, ax=ax, label="Transition count")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


def save_summary_table(per_method_results, out_dir):
    headers = ["method", "params", "alphabet_defined", "alphabet_used",
               "usage_rate", "run_length_median", "run_length_mean",
               "run_length_max", "n_distinct_transitions",
               "transition_entropy"]
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
                r["n_distinct_transitions"],
                f"{r['transition_entropy']:.2f}",
            ])
    print(f"  Saved: {csv_path}")

    # Render as image for thesis
    n_rows = len(per_method_results) + 1
    fig_w = 14
    fig_h = max(2, n_rows * 0.45 + 0.8)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.axis("off")

    cell_text = [headers]
    for r in per_method_results:
        cell_text.append([
            r["method"],
            ", ".join(f"{k}={v}" for k, v in r["params"].items()),
            r["alphabet"]["n_symbols_defined"],
            r["alphabet"]["n_symbols_used"],
            f"{r['alphabet']['usage_rate']:.2f}",
            f"{r['run_length_median']:.1f}",
            f"{r['run_length_mean']:.1f}",
            r["run_length_max"],
            r["n_distinct_transitions"],
            f"{r['transition_entropy']:.2f}",
        ])

    tbl = ax.table(cellText=cell_text, cellLoc="center", loc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    tbl.scale(1, 1.4)
    for j in range(len(headers)):
        tbl[(0, j)].set_facecolor("#2c3e50")
        tbl[(0, j)].set_text_props(color="white", fontweight="bold")
    fig.suptitle("Symbolic sequence characterization", fontsize=11,
                 fontweight="bold", y=0.95, color="#2c3e50")
    out_path = out_dir / "table_summary.png"
    fig.savefig(out_path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  Saved: {out_path}")


# =============================================================================
# MAIN
# =============================================================================

def _git_hash():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT
        ).decode().strip()
    except Exception:
        return "unknown"


def _save_config(out_dir, training_condition):
    lines = [
        "=" * 55,
        "Symbolic sequence characterization",
        "=" * 55,
        "",
        f"Timestamp          : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"Git hash           : {_git_hash()}",
        f"Training condition : {training_condition}",
        "",
        "--- Methods ---",
        ]
    for method, params in METHODS:
        param_str = ", ".join(f"{k}={v}" for k, v in params.items())
        lines.append(f"  {method:8s}: {param_str}")
    lines += ["", "--- Output folder ---", f"  {out_dir}", "", "=" * 55]
    (out_dir / "config.txt").write_text("\n".join(lines))
    print(f"  Saved: {out_dir / 'config.txt'}")


if __name__ == "__main__":
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    out_dir = ROOT / "Data" / "Graphs" / "SeqCharacterization" / timestamp
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output folder: {out_dir}\n")

    data = load_all_data()
    train_traces = data[f"{TRAINING_CONDITION}_train"]
    train_list = to_list_format(train_traces)
    print(f"Characterizing {len(train_list)} {TRAINING_CONDITION} training traces.\n")

    _save_config(out_dir, TRAINING_CONDITION)

    log = {
        "timestamp":          timestamp,
        "git_hash":           _git_hash(),
        "training_condition": TRAINING_CONDITION,
        "n_traces":           len(train_list),
        "methods":            [],
    }
    per_method_results = []

    for method, params in METHODS:
        print(f"--- {method} {params} ---")
        symbolic_traces, n_symbols = _discretize(method, params, train_list)

        alphabet_info = alphabet_usage(symbolic_traces, n_symbols)
        run_lengths = run_length_distribution(symbolic_traces)
        trans_matrix, alphabet = transition_matrix(symbolic_traces, n_symbols)
        n_distinct_transitions = int((trans_matrix > 0).sum())
        trans_entropy = transition_entropy(trans_matrix)

        # Stats summary
        rl_arr = np.array(run_lengths)
        result = {
            "method":   method,
            "params":   params,
            "alphabet": alphabet_info,
            "run_length_median": float(np.median(rl_arr)),
            "run_length_mean":   float(np.mean(rl_arr)),
            "run_length_max":    int(np.max(rl_arr)),
            "run_length_min":    int(np.min(rl_arr)),
            "run_length_std":    float(np.std(rl_arr)),
            "n_distinct_transitions": n_distinct_transitions,
            "transition_entropy":     trans_entropy,
        }
        per_method_results.append(result)

        print(f"  Alphabet:         {alphabet_info['n_symbols_used']}/{n_symbols} symbols used "
              f"(usage rate {alphabet_info['usage_rate']:.2f})")
        print(f"  Run length:       median={result['run_length_median']:.1f}, "
              f"mean={result['run_length_mean']:.1f}, max={result['run_length_max']}")
        print(f"  Transitions:      {n_distinct_transitions} distinct, "
              f"entropy={trans_entropy:.2f} bits")
        print()

        # Plots
        plot_symbol_frequency(method, alphabet_info["freq_raw"],
                              out_dir / f"symbol_frequency_{method}.png")
        plot_run_length_distribution(method, run_lengths,
                                     out_dir / f"run_length_{method}.png")
        plot_transition_matrix(method, trans_matrix, alphabet,
                               out_dir / f"transition_matrix_{method}.png")

    log["methods"] = per_method_results
    with open(out_dir / "results.json", "w") as f:
        json.dump(log, f, indent=2)
    print(f"  Saved: {out_dir / 'results.json'}")

    save_summary_table(per_method_results, out_dir)

    print(f"\nDone. Results -> {out_dir}")