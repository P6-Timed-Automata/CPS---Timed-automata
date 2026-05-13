"""
run_scaling.py
==============
Run TA-learning scaling experiments across datasets and methods.
Results are written to a timestamped subfolder under ScalingExperiments/
so runs never overwrite each other.

Usage:
    python run_scaling.py
"""

import csv
import json
import os
import time
from datetime import datetime
from pathlib import Path

import numpy as np

from DataProcessing.processData import get_trace_files
from Discretization.discretizationSetup import (
    csv_to_temp_time_list,
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


# =============================================================================
# FORMAT OUTPUT (collapsed — consecutive same-symbol events merged)
# =============================================================================

def format_output_collapsed(symbolic_res_list: list, output_path: str) -> None:
    """
    Write timed strings to file, collapsing consecutive same-symbol events.
    e.g. [a:0, a:300, a:300, c:300] -> "a:600 c:300"
    This ensures TAG guards represent thermal dwell times, not sampling intervals.
    """
    lines = []
    for trace in symbolic_res_list:
        if not trace:
            continue
        collapsed = []
        current_symbol, current_time = trace[0]
        accumulated = 0
        prev_time = current_time

        for i, (symbol, value) in enumerate(trace):
            if i == 0:
                current_symbol = symbol
                prev_time = value
                accumulated = 0
                continue
            delay = max(0, int(float(value) - float(prev_time)))
            prev_time = value
            if symbol == current_symbol:
                accumulated += delay
            else:
                collapsed.append(f"{current_symbol}:{accumulated}")
                current_symbol = symbol
                accumulated = delay

        collapsed.append(f"{current_symbol}:{accumulated}")
        lines.append(" ".join(collapsed))

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        f.write("\n".join(lines))


# =============================================================================
# HELPERS
# =============================================================================

def _discretize(method_type: str, params: dict, subset_data: list):
    if method_type == "naive":
        return equal_width_discretization(subset_data, k=params["bins"])

    elif method_type == "sax":
        traces, bins_z, mean_, std_ = sax_discretization_multi(
            subset_data, w=params["w"], k=params["bins"]
        )
        bins_c = np.sort(bins_z) * std_ + mean_
        return traces, bins_c

    elif method_type == "persist":
        ts = flatten_traces_to_ts(subset_data)
        persist_obj = Persist(
            ts, break_min=2, break_max=params["bins"], skip=np.array([4, 4])
        )
        bins_c = get_best_bins(persist_obj, ts)
        traces = discretize_traces_with_bins(subset_data, bins_c)
        return traces, bins_c

    else:
        raise ValueError(f"Unknown method: {method_type}")


def check_traces(folder: Path, required: int) -> list:
    available = sorted(get_trace_files(folder_path=str(folder)))
    print(f"Found {len(available)} trace files in:\n  {folder}")
    if len(available) < required:
        raise RuntimeError(
            f"Need at least {required} traces, but only {len(available)} found."
        )
    print(f"Using first {required}.\n")
    return available[:required]


def _param_label(method_type: str, params: dict) -> str:
    """Human-readable parameter string, consistent with run_benchmark.py naming."""
    if method_type == "persist":
        return f"break_max={params['bins']}"
    return "_".join(f"{k}{v}" for k, v in params.items())


# =============================================================================
# CSV LOGGING (raw per-repeat data)
# =============================================================================

def append_scaling_log(
        log_path: str,
        dataset_name: str,
        method_type: str,
        params: dict,
        trace_count: int,
        repeat_id: int,
        runtime: float,
        actual_bins: int,
        tag_k: int,
) -> None:
    file_exists = os.path.exists(log_path)
    with open(log_path, "a", newline="") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow([
                "timestamp", "dataset", "method", "params",
                "trace_count", "repeat", "runtime_seconds",
                "actual_bins", "tag_k",
            ])
        writer.writerow([
            datetime.now().isoformat(),
            dataset_name,
            method_type,
            str(params),
            trace_count,
            repeat_id,
            runtime,
            actual_bins,
            tag_k,
        ])


# =============================================================================
# SINGLE EXPERIMENT
# =============================================================================

def run_scaling_experiment(
        dataset_name: str,
        all_data: list,
        method_type: str,
        params: dict,
        output_folder: str,
        csv_log_path: str,
        max_traces: int = 20,
        repeats: int = 5,
        tag_k: int = 2,
) -> dict:

    tmp_file = os.path.join(os.getcwd(), "_tmp_scaling_input.txt")
    trace_counts = list(range(1, max_traces + 1))

    means = []
    stds  = []

    for n in trace_counts:
        subset = all_data[:n]

        traces, bins_c = _discretize(method_type, params, subset)

        # Persist overcounts by 1 — correct for logging
        actual_bins = len(bins_c) - 1
        if method_type == "persist":
            actual_bins -= 1

        symbolic_traces, _, _ = map_bins_to_symbols(traces, len(bins_c) - 1, bins_c)
        format_output_collapsed(symbolic_traces, tmp_file)

        run_times = []
        for repeat_id in range(repeats):
            t0      = time.perf_counter()
            TALearner(tmp_file, display=False, k=tag_k)
            runtime = time.perf_counter() - t0
            run_times.append(runtime)
            append_scaling_log(
                log_path     = csv_log_path,
                dataset_name = dataset_name,
                method_type  = method_type,
                params       = params,
                trace_count  = n,
                repeat_id    = repeat_id,
                runtime      = runtime,
                actual_bins  = actual_bins,
                tag_k        = tag_k,
            )

        m = float(np.mean(run_times))
        s = float(np.std(run_times))
        means.append(m)
        stds.append(s)
        print(f"  n={n:2d}  mean={m:.3f}s  std={s:.3f}s", flush=True)

    if os.path.exists(tmp_file):
        os.remove(tmp_file)

    return {
        "dataset":      dataset_name,
        "method":       method_type,
        "params":       params,
        "label":        f"{dataset_name} — {method_type.upper()} ({_param_label(method_type, params)})",
        "trace_counts": trace_counts,
        "means":        means,
        "stds":         stds,
    }


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":

    BASE_DIR = Path(__file__).resolve().parent.parent.parent

    # ---- Timestamped output folder -----------------------------------------
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    OUT_DIR   = BASE_DIR / "Data" / "Graphs" / "ScalingExperiments" / timestamp
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    CSV_LOG  = str(OUT_DIR / "scaling_raw.csv")
    JSON_LOG = str(OUT_DIR / "scaling_log.json")

    print(f"Run folder: {OUT_DIR}\n")

    MAX_TRACES = 25
    REPEATS    = 3
    TAG_K      = 2

    # ---- Datasets ----------------------------------------------------------
    DATASETS = [
        ("clean", BASE_DIR / "Data" / "synthetic_data-absolute" / "clean_train"),
        ("noisy", BASE_DIR / "Data" / "synthetic_data-absolute" / "noisy_train"),
        # ("negative", BASE_DIR / "Data" / "synthetic_data" / "negative"),
    ]

    # ---- Experiments -------------------------------------------------------
    EXPERIMENTS = [
        ("naive",   {"bins": 15}),
        ("sax",     {"w": 144, "bins": 15}),
        # ("persist", {"bins": 5}),
    ]

    # ---- Run ---------------------------------------------------------------
    log = {
        "timestamp":  timestamp,
        "tag_k":      TAG_K,
        "repeats":    REPEATS,
        "max_traces": MAX_TRACES,
        "results":    [],
    }

    for dataset_name, trace_folder in DATASETS:
        print("=" * 70)
        print(f"DATASET: {dataset_name}")
        print("=" * 70)

        all_files = check_traces(trace_folder, MAX_TRACES)
        all_data  = csv_to_temp_time_list(input_files=all_files)

        for method_type, params in EXPERIMENTS:
            print(f"{'-' * 60}")
            print(f"Method : {method_type.upper()}")
            print(f"Params : {params}")
            print(f"{'-' * 60}")

            result = run_scaling_experiment(
                dataset_name  = dataset_name,
                all_data      = all_data,
                method_type   = method_type,
                params        = params,
                output_folder = str(OUT_DIR),
                csv_log_path  = CSV_LOG,
                max_traces    = MAX_TRACES,
                repeats       = REPEATS,
                tag_k         = TAG_K,
            )

            log["results"].append(result)

            # Save after every experiment so partial runs are recoverable
            with open(JSON_LOG, "w") as f:
                json.dump(log, f, indent=2)

            print()

    print(f"\nAll done. Results in: {OUT_DIR}")