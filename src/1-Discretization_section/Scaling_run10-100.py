"""
run_scaling.py
==============
Run TA-learning scaling experiments across datasets, methods, and TAG k values.

For each (dataset, method, params, tag_k) combination, measures three timings
as a function of training corpus size:
  - disc_time   : just the discretization step
  - learn_time  : just the TAG learning step
  - total_time  : discretization + symbolic conversion + TAG learning

Also records TA state count, edge count, and consistency at each corpus size.

Output layout:
    Data/Graphs/ScalingExperiments/<timestamp>/
        config.txt
        k2/
            scaling_log.json
            scaling_raw.csv
        k3/
            scaling_log.json
            scaling_raw.csv
        k4/
            ...

Usage:
    python run_scaling.py
"""

import csv
import json
import os
import tempfile
import time
import uuid
from datetime import datetime
from pathlib import Path

import numpy as np

from DataProcessing.processData import get_trace_files
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


# =============================================================================
# CONFIG FILE
# =============================================================================

def save_config(out_dir, tag_k_values, max_traces, repeats, datasets, experiments):
    """Save a plain-text summary of all scaling experiment parameters."""
    lines = [
        "=" * 60,
        "Run configuration -- Scaling Experiment",
        "=" * 60,
        "",
        f"Timestamp     : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"TAG k values  : {tag_k_values}",
        f"Max traces    : {max_traces}",
        f"Repeats       : {repeats}",
        "",
        "--- Datasets ---",
        ]
    for name, folder in datasets:
        lines.append(f"  {name:12s}: {folder}")

    lines += ["", "--- Methods ---"]
    for method_type, params in experiments:
        param_str = ", ".join(f"{k}={v}" for k, v in params.items())
        lines.append(f"  {method_type:10s}: {param_str}")

    lines += ["", "--- Output folder ---", f"  {out_dir}", "", "=" * 60]

    config_path = out_dir / "config.txt"
    config_path.write_text("\n".join(lines))
    print(f"  Config saved: {config_path}")


# =============================================================================
# DISCRETIZATION ROUTING
# =============================================================================

def _discretize(method_type, params, subset_data):
    """Fit discretization on a corpus subset and return (traces, bins_c)."""
    if method_type == "naive":
        return equal_width_discretization(subset_data, k=params["bins"])

    elif method_type == "sax":
        traces, bins_z, mean_, std_ = sax_discretization_multi(
            subset_data, w=params["w"], k=params["bins"]
        )
        bins_c = sax_bins_in_original_space(bins_z, mean_, std_)
        return traces, bins_c

    elif method_type == "persist":
        ts = flatten_traces_to_ts(subset_data)
        # Fix Persist's chosen k to params["bins"] so the same alphabet size
        # is compared across methods at each corpus size.
        persist_obj = Persist(
            ts,
            break_min=params["bins"],
            break_max=params["bins"],
            skip=np.array([4, 4]),
        )
        bins_c = get_best_bins(persist_obj, ts)
        traces = discretize_traces_with_bins(subset_data, bins_c)
        return traces, bins_c

    else:
        raise ValueError(f"Unknown method: {method_type}")


def check_traces(folder, required):
    available = sorted(get_trace_files(folder_path=str(folder)))
    print(f"Found {len(available)} trace files in:\n  {folder}")
    if len(available) < required:
        raise RuntimeError(
            f"Need at least {required} traces, but only {len(available)} found."
        )
    print(f"Using first {required}.\n")
    return available[:required]


def _param_label(method_type, params):
    if method_type == "persist":
        return f"k={params['bins']}"
    return "_".join(f"{k}{v}" for k, v in params.items())


# =============================================================================
# CSV LOGGING (per-repeat raw measurements)
# =============================================================================

CSV_HEADER = [
    "timestamp", "dataset", "method", "params",
    "trace_count", "repeat", "tag_k",
    "disc_time", "learn_time", "total_time",
    "actual_bins", "n_states", "n_edges", "n_inconsistencies",
]


def append_scaling_log(log_path, row):
    """Append one row to the CSV log, creating the file with header if needed."""
    file_exists = os.path.exists(log_path)
    with open(log_path, "a", newline="") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(CSV_HEADER)
        writer.writerow(row)


# =============================================================================
# STATS HELPERS
# =============================================================================

def _stat_summary(values):
    """Return a dict of {median, min, max, mean, std, per_repeat} for a list."""
    arr = np.asarray(values, dtype=float)
    return {
        "median":     float(np.median(arr)),
        "min":        float(np.min(arr)),
        "max":        float(np.max(arr)),
        "mean":       float(np.mean(arr)),
        "std":        float(np.std(arr)),
        "per_repeat": values if isinstance(values, list) else arr.tolist(),
    }


# =============================================================================
# SINGLE EXPERIMENT (one method, one tag_k, scaling across n)
# =============================================================================

def run_scaling_experiment(
        dataset_name, all_data, method_type, params,
        csv_log_path, max_traces, repeats, tag_k,
):
    trace_counts = list(range(1, max_traces + 1))

    # Per-n aggregates — each is a list of stat_summary dicts
    disc_time_stats = []
    learn_time_stats = []
    total_time_stats = []
    state_stats = []
    edge_stats = []
    consistency_counts = []
    actual_bins_per_n = []

    for n in trace_counts:
        subset = all_data[:n]

        disc_times = []
        learn_times = []
        total_times = []
        states_per_repeat = []
        edges_per_repeat = []
        consistent_per_repeat = []
        actual_bins_for_this_n = None

        for repeat_id in range(repeats):
            tmp_file = os.path.join(
                tempfile.gettempdir(),
                f"scaling_{uuid.uuid4().hex}.txt",
            )
            try:
                # ------- Discretization timing -------
                t0 = time.perf_counter()
                traces, bins_c = _discretize(method_type, params, subset)
                t_after_disc = time.perf_counter()

                # Symbolic conversion + file I/O (counted toward total only).
                symbolic_traces, _, _ = map_bins_to_symbols(traces, bins_c)
                format_output(symbolic_traces, tmp_file)
                t_before_learn = time.perf_counter()

                # ------- TAG learning timing -------
                learner = TALearner(tmp_file, display=False, k=tag_k)
                t_after_learn = time.perf_counter()

                disc_time = t_after_disc - t0
                learn_time = t_after_learn - t_before_learn
                total_time = t_after_learn - t0

                n_states = len(learner.ta.states)
                n_edges = len(learner.ta.edges)
                n_inconsistencies = learner.ta.inconsistency_nb(
                    learner.tss, timed=True, show=False, p=False
                )
                is_consistent = (n_inconsistencies == 0)

                if actual_bins_for_this_n is None:
                    actual_bins_for_this_n = len(bins_c) - 1
                    if method_type == "persist":
                        actual_bins_for_this_n -= 1

                disc_times.append(disc_time)
                learn_times.append(learn_time)
                total_times.append(total_time)
                states_per_repeat.append(n_states)
                edges_per_repeat.append(n_edges)
                consistent_per_repeat.append(is_consistent)

                append_scaling_log(csv_log_path, [
                    datetime.now().isoformat(),
                    dataset_name, method_type, str(params),
                    n, repeat_id, tag_k,
                    disc_time, learn_time, total_time,
                    actual_bins_for_this_n, n_states, n_edges, n_inconsistencies,
                ])

            finally:
                if os.path.exists(tmp_file):
                    try:
                        os.remove(tmp_file)
                    except OSError:
                        pass

        disc_time_stats.append(_stat_summary(disc_times))
        learn_time_stats.append(_stat_summary(learn_times))
        total_time_stats.append(_stat_summary(total_times))
        state_stats.append(_stat_summary(states_per_repeat))
        edge_stats.append(_stat_summary(edges_per_repeat))
        consistency_counts.append(int(sum(consistent_per_repeat)))
        actual_bins_per_n.append(actual_bins_for_this_n)

        inc_flag = ("" if consistency_counts[-1] == repeats
                    else f" [✗{repeats - consistency_counts[-1]}/{repeats}]")
        print(
            f"  n={n:2d}  total={total_time_stats[-1]['median']:.3f}s "
            f"(disc={disc_time_stats[-1]['median']:.3f} "
            f"+ learn={learn_time_stats[-1]['median']:.3f}) | "
            f"states={int(state_stats[-1]['median']):3d}  "
            f"edges={int(edge_stats[-1]['median']):3d}"
            f"{inc_flag}",
            flush=True,
        )

    return {
        "dataset":         dataset_name,
        "method":          method_type,
        "params":          params,
        "tag_k":           tag_k,
        "label":           f"{dataset_name} -- {method_type.upper()} "
                           f"({_param_label(method_type, params)})",
        "trace_counts":    trace_counts,
        "actual_bins":     actual_bins_per_n,
        "n_repeats":       repeats,

        # Three separate timing series; each entry is a stat_summary dict
        "disc_time":       disc_time_stats,
        "learn_time":      learn_time_stats,
        "total_time":      total_time_stats,

        # Structure metrics
        "n_states":        state_stats,
        "n_edges":         edge_stats,

        # Consistency count per n (out of `repeats`)
        "n_consistent":    consistency_counts,
    }


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":

    BASE_DIR = Path(__file__).resolve().parent.parent.parent

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    OUT_DIR = BASE_DIR / "Data" / "Graphs" / "ScalingExperiments" / timestamp
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Run folder: {OUT_DIR}\n")

    MAX_TRACES = 100
    REPEATS = 5
    TAG_K_VALUES = [2, 4]

    DATASETS = [
        ("clean", BASE_DIR / "Data" / "synthetic_data-absolute" / "clean_train"),
        ("noisy", BASE_DIR / "Data" / "synthetic_data-absolute" / "noisy_train"),
    ]

    # Use the best parameters identified by the benchmark for each method.
    # All methods compared at the same alphabet size for cleanness.
    EXPERIMENTS = [
        ("naive",   {"bins": 10}),
        ("sax",     {"w": 144, "bins": 10}),
        ("persist", {"bins": 11}),
    ]

    print("=== Config ===")
    save_config(OUT_DIR, TAG_K_VALUES, MAX_TRACES, REPEATS, DATASETS, EXPERIMENTS)
    print()

    # ----- Load all dataset data once (shared across all k values) -----
    print("=== Loading datasets ===")
    dataset_data = {}
    for dataset_name, trace_folder in DATASETS:
        print(f"  {dataset_name}: {trace_folder}")
        all_files = check_traces(trace_folder, MAX_TRACES)
        dataset_data[dataset_name] = csv_to_temp_time_list(input_files=all_files)
    print()

    # ----- Run scaling experiment for each k value -----
    for tag_k in TAG_K_VALUES:
        print(f"\n{'#' * 70}")
        print(f"### TAG k = {tag_k}")
        print(f"{'#' * 70}\n")

        k_dir = OUT_DIR / f"k{tag_k}"
        k_dir.mkdir(parents=True, exist_ok=True)
        csv_log = str(k_dir / "scaling_raw.csv")
        json_log = str(k_dir / "scaling_log.json")

        log = {
            "timestamp":     timestamp,
            "tag_k":         tag_k,
            "repeats":       REPEATS,
            "max_traces":    MAX_TRACES,
            "results":       [],
        }

        for dataset_name, _ in DATASETS:
            print("=" * 70)
            print(f"DATASET: {dataset_name}  (k={tag_k})")
            print("=" * 70)

            all_data = dataset_data[dataset_name]

            for method_type, params in EXPERIMENTS:
                print("-" * 60)
                print(f"Method : {method_type.upper()}")
                print(f"Params : {params}")
                print("-" * 60)

                result = run_scaling_experiment(
                    dataset_name=dataset_name,
                    all_data=all_data,
                    method_type=method_type,
                    params=params,
                    csv_log_path=csv_log,
                    max_traces=MAX_TRACES,
                    repeats=REPEATS,
                    tag_k=tag_k,
                )

                log["results"].append(result)

                # Save incrementally so partial results survive a crash.
                with open(json_log, "w") as f:
                    json.dump(log, f, indent=2)

                print()

        print(f"  k={tag_k} done. Results in: {k_dir}\n")

    print(f"\nAll done. Results in: {OUT_DIR}")