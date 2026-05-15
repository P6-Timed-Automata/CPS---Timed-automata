"""
run_benchmark.py
================
Discretization + TA-learning benchmark across Naive, SAX, and Persist for
multiple TAG k values.

Per-trace design:
  For each variant (method, params) and each TAG k value, fit one TA per
  trace and report aggregate statistics (median, min, max) across the 20
  traces. This makes timing and structure metrics robust to single-trace
  variability — the original motivation for running 20 traces.

For each TAG k value, results go to TA_Benchmark/<timestamp>/k<n>/ with
its own benchmark_log.json. plot_benchmark.py can then be pointed at one
k subfolder to produce a plot set for that k.

Methodology note (for the thesis):
  This benchmark selects parameters per method. The downstream anomaly-
  detection pipeline (exp_51_52 etc.) uses those parameters with corpus
  fitting (one TA from all training traces) to produce the actual TA.
  Two-stage workflow: benchmark for parameter selection, pipeline for
  the production TA.
"""

import json
import os
import tempfile
import time
import uuid
from datetime import datetime
from pathlib import Path

import numpy as np

from Discretization.discretizationSetup import (
    csv_to_temp_time_list,
    format_output,
    map_bins_to_symbols,
)
from Discretization.naive import equal_width_discretization
from Discretization.sax import sax_discretization_multi, sax_bins_in_original_space
from Discretization.persist import (
    Persist,
    get_best_bins,
    discretize_traces_with_bins,
    flatten_traces_to_ts,
)
from TAG.TALearner import TALearner


# ---------------------------------------------------------------------------
# Trace loading helpers
# ---------------------------------------------------------------------------

def load_trace(path):
    """Load a single CSV trace and return (times, values) as numpy arrays."""
    data = np.genfromtxt(path, delimiter=";", skip_header=1)
    return data[:, 0], data[:, 1]


# ---------------------------------------------------------------------------
# Discretization + MAE helpers
# ---------------------------------------------------------------------------

def discretized_to_step(discretized_trace, bins_celsius):
    """
    Map a discretized trace [(label, time), ...] back to bin-center values
    for MAE computation against the raw signal.
    """
    bin_centers = (bins_celsius[:-1] + bins_celsius[1:]) / 2
    times = np.array([t for _, t in discretized_trace], dtype=float)
    labels = np.array([l for l, _ in discretized_trace], dtype=int)
    assert (labels >= 0).all() and (labels < len(bin_centers)).all(), (
        f"Labels out of [0, {len(bin_centers)-1}]: "
        f"min={labels.min()}, max={labels.max()}"
    )
    return times, bin_centers[labels]


def compute_errors(t_disc, v_disc, t_raw, v_raw):
    """
    Compute MAE between a discretized signal (at times t_disc with bin-center
    values v_disc) and the raw signal (interpolated to t_disc).
    """
    v_raw_interp = np.interp(t_disc, t_raw, v_raw)
    residuals = v_disc - v_raw_interp
    mae = float(np.mean(np.abs(residuals)))
    return mae, residuals


# ---------------------------------------------------------------------------
# Per-trace discretization (fits bins on one trace only)
# ---------------------------------------------------------------------------

def discretize_one_trace(method_type, params, trace_data):
    """
    Fit discretization on a single trace and return:
      - discretized_trace : [(label, time), ...] for this trace
      - bins_celsius      : bin edges in original (Celsius) space

    Wraps the trace in a 1-element list for compatibility with discretization
    APIs that expect a list of traces.
    """
    if method_type == "naive":
        traces, bins_c = equal_width_discretization([trace_data], k=params["bins"])
        return traces[0], bins_c

    elif method_type == "persist":
        ts = flatten_traces_to_ts([trace_data])
        persist_obj = Persist(
            ts,
            break_min=params["bins"],
            break_max=params["bins"],
            skip=np.array([4, 4]),
        )
        bins_c = get_best_bins(persist_obj, ts)
        traces = discretize_traces_with_bins([trace_data], bins_c)
        return traces[0], bins_c

    elif method_type == "sax":
        traces, bins_z, mean_, std_ = sax_discretization_multi(
            [trace_data], w=params["w"], k=params["bins"]
        )
        bins_c = sax_bins_in_original_space(bins_z, mean_, std_)
        return traces[0], bins_c

    else:
        raise ValueError(f"Unknown method: {method_type}")


# ---------------------------------------------------------------------------
# TA learning from one trace's symbolic sequence
# ---------------------------------------------------------------------------

def learn_ta_from_one_trace(discretized_trace, bins_c, tag_k):
    """
    Learn a TA from a single discretized trace. Returns:
      (n_states, n_edges, is_consistent, n_inconsistencies)

    Consistency: TAG should produce a TA that accepts its training trace(s).
    A non-zero inconsistency count means the learned TA cannot replay one
    or more of its own training traces — a sign that learning over-merged,
    over-split, or hit a corner case.
    """
    actual_bins = len(bins_c) - 1
    symbolic_traces, _, _ = map_bins_to_symbols( #tried to fix
        [discretized_trace], bins_c
)
    # Unique tmp file so concurrent runs and Windows file locks don't collide.
    tmp_file = os.path.join(
        tempfile.gettempdir(),
        f"tag_input_{uuid.uuid4().hex}.txt",
    )
    try:
        format_output(symbolic_traces, tmp_file)
        learner = TALearner(tmp_file, display=False, k=tag_k)

        n_states = len(learner.ta.states)
        n_edges = len(learner.ta.edges)

        # Check whether the learned TA still accepts the training trace.
        # show=False, p=False suppress the default printing.
        inconsistency_count = learner.ta.inconsistency_nb(
            learner.tss, timed=True, show=False, p=False
        )
        is_consistent = (inconsistency_count == 0)

        return n_states, n_edges, is_consistent, inconsistency_count
    finally:
        if os.path.exists(tmp_file):
            try:
                os.remove(tmp_file)
            except OSError:
                # Windows occasionally holds locks briefly; not worth crashing over.
                pass


# ---------------------------------------------------------------------------
# Per-variant experiment
# ---------------------------------------------------------------------------

def run_variant(label, method_type, params, data_lists, trace_files,
                t_raws, v_raws, tag_k):
    """
    One variant = one (method, params) combination.
    Fits one TA per trace, then aggregates across traces.
    """
    n_traces = len(data_lists)

    per_trace_mae = []
    per_trace_resids = []
    per_trace_steps = []
    per_trace_times = []
    per_trace_states = []
    per_trace_edges = []
    per_trace_consistent = []
    per_trace_actual_bins = []

    for i, trace_data in enumerate(data_lists):
        t0 = time.perf_counter()

        # Fit discretization on this trace only.
        discretized_trace, bins_c = discretize_one_trace(method_type, params, trace_data)

        # Learn a TA from this trace's symbolic sequence.
        n_states, n_edges, is_consistent, _ = learn_ta_from_one_trace(
            discretized_trace, bins_c, tag_k
        )

        elapsed = time.perf_counter() - t0

        per_trace_times.append(elapsed)
        per_trace_states.append(n_states)
        per_trace_edges.append(n_edges)
        per_trace_consistent.append(is_consistent)

        actual_bins = len(bins_c) - 1
        if method_type == "persist":
            # get_best_bins adds two outer edges that aren't true cuts;
            # subtract one to recover the number of bins Persist actually
            # chose.
            actual_bins -= 1
        per_trace_actual_bins.append(actual_bins)

        # MAE: discretized trace versus its own raw signal.
        t_d, v_d = discretized_to_step(discretized_trace, bins_c)
        mae, resids = compute_errors(t_d, v_d, t_raws[i], v_raws[i])
        per_trace_mae.append(mae)
        per_trace_resids.append(resids)
        per_trace_steps.append((t_d, v_d))

    # Pick the median-MAE trace for plotting (Concern 8).
    # np.argsort gives ascending indices; the middle index lands at the median.
    median_idx = int(np.argsort(per_trace_mae)[len(per_trace_mae) // 2])

    result = {
        "label":              label,
        "method_type":        method_type,
        "params":             params,
        "tag_k":              tag_k,

        # MAE across traces — median + min/max (Note 13: robust statistics)
        "mae_median":         float(np.median(per_trace_mae)),
        "mae_mean":           float(np.mean(per_trace_mae)),    # kept for completeness
        "mae_std":            float(np.std(per_trace_mae)),
        "mae_min":            float(np.min(per_trace_mae)),
        "mae_max":            float(np.max(per_trace_mae)),

        # Timing across traces
        "time_median":        float(np.median(per_trace_times)),
        "time_mean":          float(np.mean(per_trace_times)),
        "time_std":           float(np.std(per_trace_times)),
        "time_min":           float(np.min(per_trace_times)),
        "time_max":           float(np.max(per_trace_times)),

        # TA structure across traces — median + min/max
        "n_states_median":  float(np.median(per_trace_states)),
        "n_states_mean":    float(np.mean(per_trace_states)),
        "n_states_std":     float(np.std(per_trace_states)),
        "n_states_min":     int(np.min(per_trace_states)),
        "n_states_max":     int(np.max(per_trace_states)),

        "n_edges_median":   float(np.median(per_trace_edges)),
        "n_edges_mean":     float(np.mean(per_trace_edges)),
        "n_edges_std":      float(np.std(per_trace_edges)),
        "n_edges_min":      int(np.min(per_trace_edges)),
        "n_edges_max":      int(np.max(per_trace_edges)),

        # Consistency: how many of the per-trace TAs accepted their own
        # training data (Bug 3 fix — surfaced in the log).
        "n_consistent":       int(sum(per_trace_consistent)),
        "n_total":            n_traces,

        # Per-trace breakdown for any future analysis
        "per_trace": [
            {
                "trace_path":  trace_files[j],
                "mae":         per_trace_mae[j],
                "time":        per_trace_times[j],
                "n_states":    per_trace_states[j],
                "n_edges":     per_trace_edges[j],
                "actual_bins": per_trace_actual_bins[j],
                "consistent":  per_trace_consistent[j],
            }
            for j in range(n_traces)
        ],

        # Plot data from the median-MAE trace, not from trace 0
        "plot_trace_path":    trace_files[median_idx],
        "plot_trace_index":   median_idx,
        "plot_t_d":           per_trace_steps[median_idx][0].tolist(),
        "plot_v_d":           per_trace_steps[median_idx][1].tolist(),
        "plot_resids":        per_trace_resids[median_idx].tolist(),
    }

    n_inconsistent = result["n_total"] - result["n_consistent"]
    inc_flag = "" if n_inconsistent == 0 else f" [{n_inconsistent}/{result['n_total']} INCONSISTENT]"
    print(
        f"    MAE={result['mae_median']:.3f} "
        f"(median, range {result['mae_min']:.3f}-{result['mae_max']:.3f}) | "
        f"time={result['time_median']:.3f}+/-{result['time_std']:.3f}s | "
        f"states={result['n_states_median']:.1f} "
        f"({result['n_states_min']}-{result['n_states_max']}) | "
        f"edges={result['n_edges_median']:.1f}"
        f"{inc_flag}"
    )

    return result


def run_experiment(method_name, variants_dict, data_lists, trace_files,
                   t_raws, v_raws, tag_k):
    """Run all variants for one method at one tag_k. Returns sorted by median MAE."""
    results = []
    for label, (method_type, params) in variants_dict.items():
        print(f"  [{method_name}] {label} ...", flush=True)
        result = run_variant(
            label=label,
            method_type=method_type,
            params=params,
            data_lists=data_lists,
            trace_files=trace_files,
            t_raws=t_raws,
            v_raws=v_raws,
            tag_k=tag_k,
        )
        results.append(result)
    # Sort by median MAE so the "best" variant is at index 0 — robust to
    # outliers that would skew a mean-based ranking.
    results.sort(key=lambda r: r["mae_median"])
    return results


# ---------------------------------------------------------------------------
# Config file
# ---------------------------------------------------------------------------

def save_config(run_dir, tag_k_values, trace_files,
                naive_vars, sax_vars, persist_vars):
    """Save a plain-text summary of all benchmark parameters."""
    def _variant_lines(variants):
        return [
            f"    {label:30s} -> "
            + ", ".join(f"{k}={v}" for k, v in params.items())
            for label, (_, params) in variants.items()
        ]

    lines = [
        "=" * 60,
        "Run configuration -- Benchmark (per-trace design)",
        "=" * 60,
        "",
        f"Timestamp     : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"TAG k values  : {tag_k_values}",
        f"Num traces    : {len(trace_files)}",
        "",
        "--- Trace files ---",
    ]
    for p in trace_files:
        lines.append(f"  {p}")

    lines += ["", "--- Naive variants ---"] + _variant_lines(naive_vars)
    lines += ["", "--- SAX variants ---"] + _variant_lines(sax_vars)
    lines += ["", "--- Persist variants ---"] + _variant_lines(persist_vars)
    lines += ["", "--- Output folder ---", f"  {run_dir}", "", "=" * 60]

    config_path = run_dir / "config.txt"
    config_path.write_text("\n".join(lines))
    print(f"  Config saved: {config_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":

    BASE_DIR = Path(__file__).resolve().parent.parent.parent

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_dir = BASE_DIR / "Data" / "Graphs" / "TA_Benchmark" / timestamp
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"Run folder: {run_dir}")

    # ----- Configuration -----
    TAG_K_VALUES = [2,4]

    base = BASE_DIR / "Data" / "synthetic_data" / "noisy_test"
    trace_files = sorted(str(p) for p in base.glob("noisy_test_tid*.csv"))[:20]
    assert len(trace_files) == 20, f"Expected 20 traces, found {len(trace_files)}"

    naive_vars = {
        f"bins={b}": ("naive", {"bins": b})
        for b in [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
    }
    sax_vars = {
        f"w={w}, bins={b}": ("sax", {"w": w, "bins": b})
        for w in [24, 48, 96, 144]  
        for b in [5, 10, 15]
    }
    persist_vars = {
        f"k={b}": ("persist", {"bins": b})
        for b in [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
    }

    # ----- Load data once (shared across all k values) -----
    print("Loading traces...")
    data_lists = csv_to_temp_time_list(input_files=trace_files)
    t_raws, v_raws = zip(*[load_trace(p) for p in trace_files])
    t_raws = list(t_raws)
    v_raws = list(v_raws)
    print(f"Loaded {len(data_lists)} traces.\n")

    # ----- Config summary -----
    print("=== Config ===")
    save_config(run_dir, TAG_K_VALUES, trace_files,
                naive_vars, sax_vars, persist_vars)

    # ----- Run benchmark for each k value -----
    for tag_k in TAG_K_VALUES:
        print(f"\n{'#' * 60}")
        print(f"### TAG k = {tag_k}")
        print(f"{'#' * 60}")

        k_dir = run_dir / f"k{tag_k}"
        k_dir.mkdir(parents=True, exist_ok=True)
        log_path = str(k_dir / "benchmark_log.json")

        # The plotter reads t_raw and v_raw to draw the raw-signal overlay.
        # Provide the median-MAE trace's raw data per variant via plot_t_d/v_d
        # (already in each result); also keep a global t_raw/v_raw of trace 0
        # for backward compatibility with the existing plot script.
        log = {
            "timestamp":   timestamp,
            "tag_k":       tag_k,
            "trace_files": trace_files,
            "t_raw":       t_raws[0].tolist(),
            "v_raw":       v_raws[0].tolist(),
            "methods":     {},
        }

        def _save_log():
            with open(log_path, "w") as f:
                json.dump(log, f, indent=2)
            print(f"  Log updated: {log_path}")

        print(f"\n=== Naive (k={tag_k}) ===")
        log["methods"]["Naive"] = run_experiment(
            "Naive", naive_vars, data_lists, trace_files,
            t_raws, v_raws, tag_k,
        )
        _save_log()

        print(f"\n=== SAX (k={tag_k}) ===")
        log["methods"]["SAX"] = run_experiment(
            "SAX", sax_vars, data_lists, trace_files,
            t_raws, v_raws, tag_k,
        )
        _save_log()

        print(f"\n=== Persist (k={tag_k}) ===")
        log["methods"]["Persist"] = run_experiment(
            "Persist", persist_vars, data_lists, trace_files,
            t_raws, v_raws, tag_k,
        )
        _save_log()

        # Per-k summary
        print(f"\nk={tag_k}: best variant per method (by median MAE):")
        for method_name, results in log["methods"].items():
            best = results[0]
            n_inc = best["n_total"] - best["n_consistent"]
            inc = f" [{n_inc}/{best['n_total']} INCONSISTENT]" if n_inc else ""
            print(
                f"  {method_name:8s}: {best['label']:25s} "
                f"MAE={best['mae_median']:.3f}, "
                f"states={best['n_states_median']:.1f}, "
                f"time={best['time_median']:.3f}s"
                f"{inc}"
            )

    print(f"\nAll done. Results in: {run_dir}")