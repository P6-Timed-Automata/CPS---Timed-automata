"""
run_benchmark.py
================
Run discretization + TA-learning experiments for Naive, SAX, and Persist methods.
Results are written to a timestamped subfolder under TA_Benchmark/.

Usage:
    python run_benchmark.py
"""

import json
import os
import time
from datetime import datetime
from pathlib import Path

import numpy as np

from Discretization.discretizationSetup import (
    csv_to_temp_time_list,
    format_output,
    map_bins_to_symbols,
)
from Discretization.naive import equal_width_discretization
from Discretization.sax import sax_discretization_multi
from Discretization.persist import (
    Persist,
    get_best_bins,
    discretize_traces_with_bins,
    flatten_traces_to_ts,
)
from TAG.TALearner import TALearner


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_trace(path):
    data = np.genfromtxt(path, delimiter=";", skip_header=1)
    return data[:, 0], data[:, 1]


def sax_bins_celsius(bins_z, global_mean, global_std):
    return np.sort(bins_z) * global_std + global_mean


def discretized_to_step(discretized_trace, bins_celsius):
    bin_centers = (bins_celsius[:-1] + bins_celsius[1:]) / 2
    times  = np.array([t for _, t in discretized_trace], dtype=float)
    labels = np.array([l for l, _ in discretized_trace], dtype=int)
    labels = np.clip(labels, 0, len(bin_centers) - 1)
    return times, bin_centers[labels]


def compute_errors(t_disc, v_disc, t_raw, v_raw):
    v_raw_interp = np.interp(t_disc, t_raw, v_raw)
    residuals    = v_disc - v_raw_interp
    mae = float(np.mean(np.abs(residuals)))
    return mae, residuals


# ---------------------------------------------------------------------------
# Config file
# ---------------------------------------------------------------------------

def save_config(run_dir, tag_k, trace_files, naive_vars, sax_vars, persist_vars):
    """Save a plain-text summary of all benchmark parameters."""

    def _variant_lines(variants):
        lines = []
        for label, (method_type, params) in variants.items():
            param_str = ", ".join(f"{k}={v}" for k, v in params.items())
            lines.append(f"    {label:30s} -> {param_str}")
        return lines

    lines = [
        "=" * 60,
        "Run configuration -- Benchmark",
        "=" * 60,
        "",
        f"Timestamp    : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"TAG k-future : {tag_k}",
        f"Num traces   : {len(trace_files)}",
        "",
        "--- Trace files ---",
        ]
    for p in trace_files:
        lines.append(f"  {p}")

    lines += ["", "--- Naive variants ---"]
    lines += _variant_lines(naive_vars)

    lines += ["", "--- SAX variants ---"]
    lines += _variant_lines(sax_vars)

    lines += ["", "--- Persist variants ---"]
    lines += _variant_lines(persist_vars)

    lines += ["", "--- Output folder ---", f"  {run_dir}", "", "=" * 60]

    config_path = run_dir / "config.txt"
    with open(config_path, "w") as f:
        f.write("\n".join(lines))
    print(f"  Config saved: {config_path}")


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

def run_ta_pipeline(method_type, params, single_trace_data, tag_k=2):
    tmp_file = os.path.join(os.getcwd(), "tmp_ta_learning_data.txt")

    if method_type == "naive":
        traces, bins_c = equal_width_discretization(single_trace_data, k=params["bins"])

    elif method_type == "persist":
        ts = flatten_traces_to_ts(single_trace_data)
        persist_obj = Persist(ts, break_min=2, break_max=params["bins"], skip=np.array([4, 4]))
        bins_c = get_best_bins(persist_obj, ts)
        traces = discretize_traces_with_bins(single_trace_data, bins_c)

    elif method_type == "sax":
        traces, bins_z, mean_, std_ = sax_discretization_multi(
            single_trace_data, w=params["w"], k=params["bins"]
        )
        bins_c = sax_bins_celsius(bins_z, mean_, std_)

    else:
        raise ValueError(f"Unknown method: {method_type}")

    actual_bins = len(bins_c) - 1
    symbolic_trace, symbol_map, _ = map_bins_to_symbols(traces, actual_bins, bins_c)
    format_output(symbolic_trace, tmp_file)
    learner = TALearner(tmp_file, display=False, k=tag_k)

    n_states = len(learner.ta.states)
    n_edges  = len(learner.ta.edges)

    return traces, bins_c, actual_bins, n_states, n_edges


# ---------------------------------------------------------------------------
# Experiment loop
# ---------------------------------------------------------------------------

def run_experiment(method_name, t_raw, v_raw, variants_dict, data_lists, trace_files, tag_k=2):
    results = []

    for label, (method_type, params) in variants_dict.items():
        print(f"  [{method_name}] {label} ...", flush=True)

        per_trace_mae    = []
        per_trace_times  = []
        per_trace_states = []
        per_trace_edges  = []
        per_trace_bins   = []
        plot_t_d = plot_v_d = plot_resids = None

        for i, (trace_path, trace_data) in enumerate(zip(trace_files, data_lists)):
            t_i, v_i = load_trace(trace_path)

            t0 = time.perf_counter()
            disc_traces, bins_c, actual_bins, n_states, n_edges = run_ta_pipeline(
                method_type, params, [trace_data], tag_k=tag_k
            )
            elapsed = time.perf_counter() - t0

            per_trace_times.append(elapsed)
            per_trace_bins.append(actual_bins)
            per_trace_states.append(n_states)
            per_trace_edges.append(n_edges)

            t_d, v_d = discretized_to_step(disc_traces[0], bins_c)
            mae, resids = compute_errors(t_d, v_d, t_i, v_i)
            per_trace_mae.append(mae)

            if i == 0:
                plot_t_d    = t_d.tolist()
                plot_v_d    = v_d.tolist()
                plot_resids = resids.tolist()

        result = {
            "label":         label,
            "method_type":   method_type,
            "params":        params,
            "mae_mean":      float(np.mean(per_trace_mae)),
            "mae_std":       float(np.std(per_trace_mae)),
            "time_mean":     float(np.mean(per_trace_times)),
            "time_std":      float(np.std(per_trace_times)),
            "n_states_mean": float(np.mean(per_trace_states)),
            "n_states_std":  float(np.std(per_trace_states)),
            "n_states_min":  float(np.min(per_trace_states)),
            "n_states_max":  float(np.max(per_trace_states)),
            "n_edges_mean":  float(np.mean(per_trace_edges)),
            "n_edges_std":   float(np.std(per_trace_edges)),
            "n_edges_min":   float(np.min(per_trace_edges)),
            "n_edges_max":   float(np.max(per_trace_edges)),
            "per_trace": [
                {
                    "trace_path":  trace_files[j],
                    "mae":         per_trace_mae[j],
                    "time":        per_trace_times[j],
                    "n_states":    per_trace_states[j],
                    "n_edges":     per_trace_edges[j],
                    "actual_bins": per_trace_bins[j],
                }
                for j in range(len(trace_files))
            ],
            "plot_t_d":    plot_t_d,
            "plot_v_d":    plot_v_d,
            "plot_resids": plot_resids,
        }

        print(
            f"    MAE={result['mae_mean']:.3f}+/-{result['mae_std']:.3f} | "
            f"time={result['time_mean']:.3f}+/-{result['time_std']:.3f}s | "
            f"states={result['n_states_mean']:.1f} | "
            f"edges={result['n_edges_mean']:.1f}"
        )
        results.append(result)

    results.sort(key=lambda x: x["mae_mean"])
    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":

    BASE_DIR = Path(__file__).resolve().parent.parent.parent

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_dir   = BASE_DIR / "Data" / "Graphs" / "TA_Benchmark" / timestamp
    run_dir.mkdir(parents=True, exist_ok=True)
    log_path  = str(run_dir / "benchmark_log.json")
    print(f"Run folder: {run_dir}")

    base = BASE_DIR / "Data" / "synthetic_data-absolute" / "noisy_test"
    trace_files = [
        str(base / f"noisy_test_tid{i}.csv")
        for i in range(1, 21)
    ]

    TAG_K = 2

    t_raw, v_raw = load_trace(trace_files[0])
    data_lists   = csv_to_temp_time_list(input_files=trace_files)
    print(f"Loaded {len(data_lists)} traces.")

    # Define variants before saving config so config captures them
    naive_vars = {
        f"bins={b}": ("naive", {"bins": b})
        for b in [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
    }
    sax_vars = {
        f"w={w}, bins={b}": ("sax", {"w": w, "bins": b})
        for w in [24, 72, 96, 144, 288]
        for b in [4, 8, 16]
    }
    persist_vars = {
        f"break_max={b}": ("persist", {"bins": b})
        for b in [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17]
    }

    # Save config immediately
    print("\n=== Config ===")
    save_config(run_dir, TAG_K, trace_files, naive_vars, sax_vars, persist_vars)

    log = {
        "timestamp":   timestamp,
        "tag_k":       TAG_K,
        "trace_files": trace_files,
        "t_raw":       t_raw.tolist(),
        "v_raw":       v_raw.tolist(),
        "methods":     {},
    }

    def _save_log():
        with open(log_path, "w") as f:
            json.dump(log, f, indent=2)
        print(f"  Log updated: {log_path}")

    print("\n=== Naive ===")
    log["methods"]["Naive"] = run_experiment(
        "Naive", t_raw, v_raw, naive_vars, data_lists, trace_files, TAG_K
    )
    _save_log()

    print("\n=== SAX ===")
    log["methods"]["SAX"] = run_experiment(
        "SAX", t_raw, v_raw, sax_vars, data_lists, trace_files, TAG_K
    )
    _save_log()

    print("\n=== Persist ===")
    log["methods"]["Persist"] = run_experiment(
        "Persist", t_raw, v_raw, persist_vars, data_lists, trace_files, TAG_K
    )
    _save_log()

    print(f"\nAll done. Results in: {run_dir}")