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

DISJOINT REPEATS (scheme B):
  At corpus size n, repeat r uses traces [r*n, (r+1)*n) from the pool.
  This means each repeat trains on a non-overlapping subset, giving
  meaningful variance across repeats (random subset selection variance,
  not just timing jitter).

  Requires pool of size >= REPEATS * MAX_TRACES per dataset. If the
  configured training folder has fewer, the script will auto-generate
  the missing traces using Generate_data.CONFIG.

Crash-resilient design (for SLURM):
  - Recursion limit raised at startup.
  - Each repeat wrapped in try/except. Failures logged, sweep continues.
  - JSON saved after every n.
  - stop_on_failure=True for Persist.

Output layout:
    Data/Graphs/ScalingExperiments/<timestamp>/
        config.txt
        k2/
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
import sys
import tempfile
import time
import uuid
from datetime import datetime
from pathlib import Path

# Raise recursion limit before TAG imports.
sys.setrecursionlimit(50000)

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


# Make Generate_data importable. It lives in src/2-Synthetic_section/.
ROOT = Path(__file__).resolve().parent.parent.parent
SYNTHETIC_SECTION = ROOT / "src" / "2-Synthetic_section"
sys.path.insert(0, str(SYNTHETIC_SECTION))
sys.path.insert(0, str(ROOT))


STOP_ON_FAILURE_PER_METHOD = {
    "naive":   False,
    "sax":     False,
    "persist": True,
}


# =============================================================================
# CONFIG FILE
# =============================================================================

def save_config(out_dir, tag_k_values, max_traces, repeats, datasets, experiments,
                pool_size_needed):
    lines = [
        "=" * 60,
        "Run configuration -- Scaling Experiment (disjoint, crash-resilient)",
        "=" * 60,
        "",
        f"Timestamp        : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"TAG k values     : {tag_k_values}",
        f"Max traces       : {max_traces}",
        f"Repeats          : {repeats}",
        f"Pool size needed : {pool_size_needed} per dataset (disjoint repeats)",
        f"Recursion lim    : {sys.getrecursionlimit()}",
        "",
        "--- Subset selection scheme ---",
        f"  At n={max_traces} (largest), repeat r uses traces "
        f"[{(repeats - 1) * max_traces}, {repeats * max_traces})",
        f"  Each (n, r) cell trains on disjoint subset of {max_traces} traces.",
        "",
        "--- Datasets ---",
        ]
    for name, folder in datasets:
        lines.append(f"  {name:12s}: {folder}")

    lines += ["", "--- Methods ---"]
    for method_type, params in experiments:
        param_str = ", ".join(f"{k}={v}" for k, v in params.items())
        stop = STOP_ON_FAILURE_PER_METHOD.get(method_type, False)
        lines.append(f"  {method_type:10s}: {param_str}  "
                     f"(stop_on_failure={stop})")

    lines += ["", "--- Output folder ---", f"  {out_dir}", "", "=" * 60]

    config_path = out_dir / "config.txt"
    config_path.write_text("\n".join(lines))
    print(f"  Config saved: {config_path}")


# =============================================================================
# POOL MANAGEMENT — auto-generate missing traces
# =============================================================================
def ensure_pool_size(dataset_name, folder, required_size):
    """
    Ensure `folder` contains at least `required_size` traces.

    If fewer exist, regenerate the full pool using Generate_data.CONFIG and
    save only the missing-index files (no overwriting).

    Returns the sorted list of trace file paths (length >= required_size).
    """
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)

    existing_files = sorted(folder.glob("*.csv"))
    n_existing = len(existing_files)

    print(f"  Pool check for '{dataset_name}': {n_existing} existing, "
          f"need {required_size}.")

    if n_existing >= required_size:
        return [str(p) for p in existing_files[:required_size]]

    # Need to generate more
    n_to_generate = required_size - n_existing
    print(f"  Generating {n_to_generate} additional traces "
          f"for '{dataset_name}'...")

    try:
        from Generate_data import CONFIG as DATA_CONFIG
    except ImportError as e:
        raise RuntimeError(
            f"Cannot auto-generate: Generate_data not importable. "
            f"Check that src/2-Synthetic_section/ is on sys.path. "
            f"Original error: {e}"
        ) from e

    from Generators import generate_trace_set

    # Map dataset name to sub-config key. DATA_CONFIG has "clean" and "noisy".
    sub_config_key = None
    if dataset_name in DATA_CONFIG and isinstance(DATA_CONFIG[dataset_name], dict):
        sub_config_key = dataset_name
    else:
        # Substring fallback: "clean_train" → "clean"
        for k in DATA_CONFIG:
            if isinstance(DATA_CONFIG[k], dict) and k in dataset_name:
                sub_config_key = k
                break

    if sub_config_key is None:
        raise ValueError(
            f"Dataset '{dataset_name}' not matched to any key in "
            f"Generate_data.CONFIG. Available config keys: "
            f"{[k for k, v in DATA_CONFIG.items() if isinstance(v, dict)]}"
        )

    sub_config = DATA_CONFIG[sub_config_key]
    seed_key = f"seed_{sub_config_key}"
    if seed_key not in DATA_CONFIG:
        raise KeyError(
            f"Expected seed key '{seed_key}' in Generate_data.CONFIG."
        )
    seed = DATA_CONFIG[seed_key]

    # generate_trace_set is prefix-stable: full_pool[:n_existing] equals
    # what the original generate_all_data.py wrote.
    print(f"    Calling generate_trace_set(n_traces={required_size}, "
          f"seed={seed}, **{sub_config_key}_config)")
    full_pool = generate_trace_set(
        n_traces=required_size,
        seed=seed,
        **sub_config,
    )

    # Sanity check: compare first existing trace to generator output.
    # If they don't match, the existing data was generated with different
    # config — bail out rather than mixing inconsistent traces.
    if n_existing > 0:
        first_existing = existing_files[0]
        data = np.genfromtxt(first_existing, delimiter=";", skip_header=1)
        existing_temps = data[:, 1]
        generated_temps = full_pool[0][1]
        if not np.allclose(existing_temps, generated_temps,
                           rtol=1e-3, atol=1e-3):
            raise RuntimeError(
                f"Existing first trace in {folder} does not match the "
                f"generator's output for seed={seed}, config={sub_config_key}. "
                f"Possible causes: existing data uses different CONFIG, "
                f"different seed, or Generators.py changed. "
                f"Fix by either deleting {folder} (forcing fresh generation) "
                f"or aligning CONFIG to match the existing traces."
            )

    # Save only the missing indices (n_existing onward).
    # generate_all_data.py uses prefix "{dataset_name}_train" with 1-indexed
    # filenames via save_traces. Continue that pattern.
    prefix = f"{sub_config_key}_train"
    traces_to_save = full_pool[n_existing:]
    _save_traces_with_index_offset(
        traces_to_save, folder, prefix, start_index=n_existing
    )

    # Re-list and return
    existing_files = sorted(folder.glob("*.csv"))
    if len(existing_files) < required_size:
        raise RuntimeError(
            f"After generation, folder has {len(existing_files)} traces "
            f"(need {required_size}). Generator wrote fewer files than expected."
        )

    print(f"  Pool for '{dataset_name}': now {len(existing_files)} traces total.")
    return [str(p) for p in existing_files[:required_size]]


def _save_traces_with_index_offset(traces, folder, prefix, start_index):
    """
    Save traces with filenames matching generate_all_data.py's convention:
        {prefix}_tid{N}.csv     (1-indexed, no zero-padding)
        time_s;temperature      (header)

    Numbers start from start_index + 1. So if start_index=100, the first
    saved trace is tid101.csv, next is tid102.csv, etc.
    """
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    for i, (times, temps) in enumerate(traces):
        idx = start_index + i + 1   # 1-indexed, matching save_traces()
        filename = folder / f"{prefix}_tid{idx}.csv"
        with open(filename, "w") as f:
            f.write("time_s;temperature\n")
            for t, v in zip(times, temps):
                f.write(f"{int(t)};{float(v):.5f}\n")
    print(f"    Saved {len(traces)} traces to {folder} "
          f"(indices {start_index + 1} to {start_index + len(traces)})")


# =============================================================================
# SUBSET SELECTION
# =============================================================================

def _subset_for_repeat(all_data, n, repeat_id):
    """
    Return the disjoint subset for (n, repeat_id).
    Repeat r at size n uses indices [r*n, (r+1)*n).
    """
    start = repeat_id * n
    end = start + n
    if end > len(all_data):
        raise IndexError(
            f"Pool exhausted: need traces [{start}, {end}) but pool has only "
            f"{len(all_data)}. Increase pool size or reduce MAX_TRACES/REPEATS."
        )
    return all_data[start:end]


# =============================================================================
# DISCRETIZATION ROUTING
# =============================================================================

def _discretize(method_type, params, subset_data):
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


def _param_label(method_type, params):
    if method_type == "persist":
        return f"k={params['bins']}"
    return "_".join(f"{k}{v}" for k, v in params.items())


# =============================================================================
# CSV LOGGING
# =============================================================================

CSV_HEADER = [
    "timestamp", "dataset", "method", "params",
    "trace_count", "repeat", "tag_k", "status",
    "trace_indices_start", "trace_indices_end",
    "disc_time", "learn_time", "total_time",
    "actual_bins", "n_states", "n_edges", "n_inconsistencies",
    "error_type", "error_msg",
]


def append_scaling_log(log_path, row):
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
    if not values:
        return {
            "median":     None,
            "min":        None,
            "max":        None,
            "mean":       None,
            "std":        None,
            "per_repeat": [],
        }
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
# SINGLE REPEAT (with crash protection)
# =============================================================================

def _run_one_repeat(method_type, params, subset, tag_k):
    tmp_file = os.path.join(
        tempfile.gettempdir(),
        f"scaling_{uuid.uuid4().hex}.txt",
    )
    try:
        t0 = time.perf_counter()
        traces, bins_c = _discretize(method_type, params, subset)
        t_after_disc = time.perf_counter()

        symbolic_traces, _, _ = map_bins_to_symbols(traces, bins_c)
        format_output(symbolic_traces, tmp_file)
        t_before_learn = time.perf_counter()

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

        actual_bins = len(bins_c) - 1
        if method_type == "persist":
            actual_bins -= 1

        return {
            "status":            "ok",
            "disc_time":         disc_time,
            "learn_time":        learn_time,
            "total_time":        total_time,
            "n_states":          n_states,
            "n_edges":           n_edges,
            "n_inconsistencies": n_inconsistencies,
            "is_consistent":     (n_inconsistencies == 0),
            "actual_bins":       actual_bins,
        }

    except RecursionError as e:
        return {"status": "failed", "error_type": "RecursionError", "error_msg": str(e)}
    except MemoryError as e:
        return {"status": "failed", "error_type": "MemoryError", "error_msg": str(e)}
    except Exception as e:
        return {"status": "failed", "error_type": type(e).__name__, "error_msg": str(e)}
    finally:
        if os.path.exists(tmp_file):
            try:
                os.remove(tmp_file)
            except OSError:
                pass


# =============================================================================
# SINGLE EXPERIMENT
# =============================================================================

def run_scaling_experiment(
        dataset_name, all_data, method_type, params,
        csv_log_path, max_traces, repeats, tag_k,
        json_log=None, json_log_path=None,
        stop_on_failure=False,
):
    trace_counts = list(range(1, max_traces + 1))

    disc_time_stats = []
    learn_time_stats = []
    total_time_stats = []
    state_stats = []
    edge_stats = []
    consistency_counts = []
    actual_bins_per_n = []

    status_per_n = []
    n_failed_per_n = []
    failure_reasons_per_n = []
    subset_indices_per_n = []   # NEW: track which traces each repeat used

    completed_n_count = 0

    for n in trace_counts:
        disc_times = []
        learn_times = []
        total_times = []
        states_per_repeat = []
        edges_per_repeat = []
        consistent_per_repeat = []
        actual_bins_for_this_n = None
        failure_reasons = []
        subset_indices_this_n = []   # one tuple (start, end) per repeat

        for repeat_id in range(repeats):
            # NEW: disjoint subset selection
            try:
                subset = _subset_for_repeat(all_data, n, repeat_id)
            except IndexError as e:
                # Pool exhausted for this (n, repeat). Record as failed.
                failure_reasons.append({
                    "repeat":     repeat_id,
                    "error_type": "PoolExhausted",
                    "error_msg":  str(e),
                })
                append_scaling_log(csv_log_path, [
                    datetime.now().isoformat(),
                    dataset_name, method_type, str(params),
                    n, repeat_id, tag_k, "failed",
                    repeat_id * n, (repeat_id + 1) * n,
                    "", "", "",
                    "", "", "", "",
                    "PoolExhausted", str(e)[:200],
                    ])
                subset_indices_this_n.append([repeat_id * n, (repeat_id + 1) * n])
                continue

            subset_indices_this_n.append([repeat_id * n, (repeat_id + 1) * n])
            r = _run_one_repeat(method_type, params, subset, tag_k)

            if r["status"] == "ok":
                disc_times.append(r["disc_time"])
                learn_times.append(r["learn_time"])
                total_times.append(r["total_time"])
                states_per_repeat.append(r["n_states"])
                edges_per_repeat.append(r["n_edges"])
                consistent_per_repeat.append(r["is_consistent"])
                if actual_bins_for_this_n is None:
                    actual_bins_for_this_n = r["actual_bins"]

                append_scaling_log(csv_log_path, [
                    datetime.now().isoformat(),
                    dataset_name, method_type, str(params),
                    n, repeat_id, tag_k, "ok",
                    repeat_id * n, (repeat_id + 1) * n,
                    r["disc_time"], r["learn_time"], r["total_time"],
                    r["actual_bins"], r["n_states"], r["n_edges"],
                    r["n_inconsistencies"],
                    "", "",
                    ])
            else:
                failure_reasons.append({
                    "repeat":     repeat_id,
                    "error_type": r["error_type"],
                    "error_msg":  r["error_msg"][:300],
                })

                append_scaling_log(csv_log_path, [
                    datetime.now().isoformat(),
                    dataset_name, method_type, str(params),
                    n, repeat_id, tag_k, "failed",
                    repeat_id * n, (repeat_id + 1) * n,
                    "", "", "",
                    "", "", "", "",
                    r["error_type"], r["error_msg"][:200],
                    ])

        disc_time_stats.append(_stat_summary(disc_times))
        learn_time_stats.append(_stat_summary(learn_times))
        total_time_stats.append(_stat_summary(total_times))
        state_stats.append(_stat_summary(states_per_repeat))
        edge_stats.append(_stat_summary(edges_per_repeat))
        consistency_counts.append(int(sum(consistent_per_repeat)))
        actual_bins_per_n.append(actual_bins_for_this_n)
        subset_indices_per_n.append(subset_indices_this_n)

        n_ok = len(disc_times)
        n_failed = repeats - n_ok
        n_failed_per_n.append(n_failed)
        failure_reasons_per_n.append(failure_reasons)

        if n_ok == 0:
            status = "failed"
        elif n_failed == 0:
            status = "ok"
        else:
            status = "partial"
        status_per_n.append(status)
        completed_n_count += 1

        if n_ok > 0:
            inc_count = consistency_counts[-1]
            inc_flag = ("" if inc_count == n_ok
                        else f" [✗{n_ok - inc_count}/{n_ok}]")
            failed_flag = "" if n_failed == 0 else f" [{n_failed}/{repeats} FAILED]"
            print(
                f"  n={n:3d}  total={total_time_stats[-1]['median']:.3f}s "
                f"(disc={disc_time_stats[-1]['median']:.3f} "
                f"+ learn={learn_time_stats[-1]['median']:.3f}) | "
                f"states={int(state_stats[-1]['median']):3d}  "
                f"edges={int(edge_stats[-1]['median']):3d}"
                f"{inc_flag}{failed_flag}",
                flush=True,
            )
        else:
            err_types = sorted({fr["error_type"] for fr in failure_reasons})
            print(
                f"  n={n:3d}  ALL {repeats} REPEATS FAILED "
                f"({', '.join(err_types)})",
                flush=True,
            )

        if json_log is not None and json_log_path is not None:
            with open(json_log_path, "w") as f:
                json.dump(json_log, f, indent=2)

        if status == "failed" and stop_on_failure:
            remaining = len(trace_counts) - completed_n_count
            print(f"  Stopping sweep at n={n} ({method_type}); "
                  f"skipping {remaining} larger n values.", flush=True)
            break

    return {
        "dataset":         dataset_name,
        "method":          method_type,
        "params":          params,
        "tag_k":           tag_k,
        "label":           f"{dataset_name} -- {method_type.upper()} "
                           f"({_param_label(method_type, params)})",
        "trace_counts":    trace_counts[:completed_n_count],
        "actual_bins":     actual_bins_per_n,
        "n_repeats":       repeats,

        "disc_time":       disc_time_stats,
        "learn_time":      learn_time_stats,
        "total_time":      total_time_stats,

        "n_states":        state_stats,
        "n_edges":         edge_stats,

        "n_consistent":    consistency_counts,

        # NEW
        "status_per_n":           status_per_n,
        "n_failed_per_n":         n_failed_per_n,
        "failure_reasons_per_n":  failure_reasons_per_n,
        "subset_indices_per_n":   subset_indices_per_n,
        "stopped_early":          completed_n_count < len(trace_counts),
    }


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":

    BASE_DIR = ROOT

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    OUT_DIR = BASE_DIR / "Data" / "Graphs" / "ScalingExperiments" / timestamp
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Run folder: {OUT_DIR}")
    print(f"Python recursion limit raised to {sys.getrecursionlimit()}\n")

    MAX_TRACES = 100
    REPEATS = 5
    TAG_K_VALUES = [2, 4]

    DATASETS = [
        ("clean", BASE_DIR / "Data" / "synthetic_data-absolute" / "clean_train"),
        ("noisy", BASE_DIR / "Data" / "synthetic_data-absolute" / "noisy_train"),
    ]

    EXPERIMENTS = [
        ("naive",   {"bins": 5}),
        ("sax",     {"w": 144, "bins": 5}),
        ("persist", {"bins": 6}),
    ]

    # NEW: pool size required for disjoint repeats
    POOL_SIZE_NEEDED = REPEATS * MAX_TRACES
    print(f"=== Pool size required (disjoint repeats) ===")
    print(f"  REPEATS × MAX_TRACES = {REPEATS} × {MAX_TRACES} = {POOL_SIZE_NEEDED} "
          f"per dataset\n")

    print("=== Config ===")
    save_config(OUT_DIR, TAG_K_VALUES, MAX_TRACES, REPEATS,
                DATASETS, EXPERIMENTS, POOL_SIZE_NEEDED)
    print()

    # ----- Ensure pool and load -----
    print("=== Ensuring trace pools and loading ===")
    dataset_data = {}
    for dataset_name, trace_folder in DATASETS:
        # Ensure enough traces exist; auto-generate if not
        all_files = ensure_pool_size(dataset_name, trace_folder, POOL_SIZE_NEEDED)
        # Load and shuffle deterministically — see note below
        traces = csv_to_temp_time_list(input_files=all_files)
        dataset_data[dataset_name] = traces
        print(f"  Loaded {len(traces)} traces for '{dataset_name}'.")
    print()

    # ----- Run scaling experiment for each k value -----
    for tag_k in TAG_K_VALUES:
        print(f"\n{'#' * 70}")
        print(f"### TAG k = {tag_k}")
        print(f"{'#' * 70}\n")

        k_dir = OUT_DIR / f"k{tag_k}"
        k_dir.mkdir(parents=True, exist_ok=True)
        csv_log = str(k_dir / "scaling_raw.csv")
        json_log_path = str(k_dir / "scaling_log.json")

        log = {
            "timestamp":          timestamp,
            "tag_k":              tag_k,
            "repeats":            REPEATS,
            "max_traces":         MAX_TRACES,
            "pool_size_needed":   POOL_SIZE_NEEDED,
            "subset_scheme":      "disjoint",
            "results":            [],
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

                stop = STOP_ON_FAILURE_PER_METHOD.get(method_type, False)

                placeholder = {
                    "dataset": dataset_name,
                    "method":  method_type,
                    "params":  params,
                    "tag_k":   tag_k,
                    "status":  "in_progress",
                }
                log["results"].append(placeholder)
                with open(json_log_path, "w") as f:
                    json.dump(log, f, indent=2)

                result = run_scaling_experiment(
                    dataset_name=dataset_name,
                    all_data=all_data,
                    method_type=method_type,
                    params=params,
                    csv_log_path=csv_log,
                    max_traces=MAX_TRACES,
                    repeats=REPEATS,
                    tag_k=tag_k,
                    json_log=log,
                    json_log_path=json_log_path,
                    stop_on_failure=stop,
                )

                log["results"][-1] = result
                with open(json_log_path, "w") as f:
                    json.dump(log, f, indent=2)

                print()

        n_failures = 0
        for r in log["results"]:
            if isinstance(r.get("n_failed_per_n"), list):
                n_failures += sum(r["n_failed_per_n"])
        if n_failures > 0:
            print(f"\nk={tag_k} summary: {n_failures} total failed repeats")
            for r in log["results"]:
                if not isinstance(r.get("status_per_n"), list):
                    continue
                if any(s != "ok" for s in r["status_per_n"]):
                    failed_n_values = [
                        n for n, s in zip(r["trace_counts"], r["status_per_n"])
                        if s != "ok"
                    ]
                    print(f"  {r['dataset']:8s} {r['method']:8s} {r['params']} | "
                          f"non-ok at n={failed_n_values}")

        print(f"\n  k={tag_k} done. Results in: {k_dir}\n")

    print(f"\nAll done. Results in: {OUT_DIR}")