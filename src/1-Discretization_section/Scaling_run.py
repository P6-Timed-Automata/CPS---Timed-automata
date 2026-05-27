"""
run_scaling.py
==============
TA-learning scaling experiments across (dataset, method, params, tag_k).
For each combination, measures TAG learning time, TA state count, and
TA edge count as a function of training corpus size (n = 1..MAX_TRACES).

DISJOINT REPEATS:
  At corpus size n, repeat r uses traces [r*n, (r+1)*n) from the pool.
  Each (n, r) cell trains on a non-overlapping subset, so reported
  variance reflects subset-selection variance rather than timing jitter.
  Requires pool size >= REPEATS * MAX_TRACES per dataset; missing
  traces are auto-generated via Generate_data.CONFIG.

Crash resilience: per-repeat try/except, JSON saved after every n,
Persist uses stop_on_failure=True. A placeholder entry is written
before each variant runs so a mid-variant crash leaves a recoverable
log (the plotter skips placeholders).

Output: Data/Graphs/ScalingExperiments/<timestamp>/<job>/k<n>/scaling_log.json
"""

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
        "Run configuration -- Scaling Experiment",
        "=" * 60,
        "",
        f"Timestamp        : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"TAG k values     : {tag_k_values}",
        f"Max traces       : {max_traces}",
        f"Repeats          : {repeats}",
        f"Pool size needed : {pool_size_needed} per dataset (disjoint repeats)",
        "",
        "--- Subset selection ---",
        f"  At n={max_traces} (largest), repeat r uses traces "
        f"[{(repeats - 1) * max_traces}, {repeats * max_traces})",
        f"  Each (n, r) cell trains on a disjoint subset of size n.",
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
    (out_dir / "config.txt").write_text("\n".join(lines))
    print(f"  Config saved: {out_dir / 'config.txt'}")


# =============================================================================
# POOL MANAGEMENT — auto-generate missing traces
# =============================================================================

def ensure_pool_size(dataset_name, folder, required_size):
    """
    Ensure `folder` contains at least `required_size` traces. If fewer
    exist, regenerate the full pool using Generate_data.CONFIG and save
    only the missing-index slices (no overwriting).
    """
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)

    existing_files = sorted(folder.glob("*.csv"))
    n_existing = len(existing_files)
    print(f"  Pool check for '{dataset_name}': {n_existing} existing, "
          f"need {required_size}.")

    if n_existing >= required_size:
        return [str(p) for p in existing_files[:required_size]]

    print(f"  Generating {required_size - n_existing} additional traces "
          f"for '{dataset_name}'...")

    try:
        from Generate_data import CONFIG as DATA_CONFIG
    except ImportError as e:
        raise RuntimeError(
            f"Cannot auto-generate: Generate_data not importable. "
            f"Original error: {e}"
        ) from e
    from Generators import generate_trace_set

    # Match dataset name to the right sub-config block in Generate_data.CONFIG.
    sub_config_key = None
    if dataset_name in DATA_CONFIG and isinstance(DATA_CONFIG[dataset_name], dict):
        sub_config_key = dataset_name
    else:
        for k in DATA_CONFIG:
            if isinstance(DATA_CONFIG[k], dict) and k in dataset_name:
                sub_config_key = k
                break
    if sub_config_key is None:
        raise ValueError(
            f"Dataset '{dataset_name}' not matched to any key in "
            f"Generate_data.CONFIG."
        )

    sub_config = DATA_CONFIG[sub_config_key]
    seed_key = f"seed_{sub_config_key}"
    if seed_key not in DATA_CONFIG:
        raise KeyError(f"Expected seed key '{seed_key}' in Generate_data.CONFIG.")
    seed = DATA_CONFIG[seed_key]

    # generate_trace_set is prefix-stable: full_pool[:n_existing] equals
    # whatever generate_all_data.py originally wrote.
    full_pool = generate_trace_set(
        n_traces=required_size, seed=seed, **sub_config,
    )

    # Sanity check: the existing first trace must match the generator's
    # output. If not, the existing data was generated with different config
    # and silently extending it would mix inconsistent traces.
    if n_existing > 0:
        data = np.genfromtxt(existing_files[0], delimiter=";", skip_header=1)
        if not np.allclose(data[:, 1], full_pool[0][1], rtol=1e-3, atol=1e-3):
            raise RuntimeError(
                f"Existing first trace in {folder} does not match generator "
                f"output for seed={seed}, config={sub_config_key}. Delete the "
                f"folder to regenerate, or align CONFIG to the existing data."
            )

    prefix = f"{sub_config_key}_train"
    _save_traces_with_index_offset(
        full_pool[n_existing:], folder, prefix, start_index=n_existing,
    )

    existing_files = sorted(folder.glob("*.csv"))
    if len(existing_files) < required_size:
        raise RuntimeError(
            f"After generation, folder has {len(existing_files)} traces "
            f"(need {required_size})."
        )
    print(f"  Pool for '{dataset_name}': now {len(existing_files)} traces total.")
    return [str(p) for p in existing_files[:required_size]]


def _save_traces_with_index_offset(traces, folder, prefix, start_index):
    """Save traces with the 1-indexed `{prefix}_tid{N}.csv` naming used
    by generate_all_data.py's save_traces()."""
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    for i, (times, temps) in enumerate(traces):
        idx = start_index + i + 1
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
    """Disjoint subset for (n, repeat_id): traces [r*n, (r+1)*n)."""
    start = repeat_id * n
    end = start + n
    if end > len(all_data):
        raise IndexError(
            f"Pool exhausted: need traces [{start}, {end}) but pool has only "
            f"{len(all_data)}."
        )
    return all_data[start:end]


# =============================================================================
# DISCRETIZATION ROUTING
# =============================================================================

def _discretize(method_type, params, subset_data):
    if method_type == "naive":
        return equal_width_discretization(subset_data, k=params["bins"])

    if method_type == "sax":
        traces, bins_z, mean_, std_ = sax_discretization_multi(
            subset_data, w=params["w"], k=params["bins"]
        )
        bins_c = sax_bins_in_original_space(bins_z, mean_, std_)
        return traces, bins_c

    if method_type == "persist":
        ts = flatten_traces_to_ts(subset_data)
        persist_obj = Persist(
            ts,
            break_min=params["bins"], break_max=params["bins"],
            skip=np.array([4, 4]),
        )
        bins_c = get_best_bins(persist_obj, ts)
        traces = discretize_traces_with_bins(subset_data, bins_c)
        return traces, bins_c

    raise ValueError(f"Unknown method: {method_type}")


# =============================================================================
# STATS
# =============================================================================

def _stat_summary(values):
    """median / min / max only. Returns None-filled dict for empty input
    so the plotter can detect 'all repeats failed at this n' cells."""
    if not values:
        return {"median": None, "min": None, "max": None}
    arr = np.asarray(values, dtype=float)
    return {
        "median": float(np.median(arr)),
        "min":    float(np.min(arr)),
        "max":    float(np.max(arr)),
    }


# =============================================================================
# SINGLE REPEAT (timed TAG learning)
# =============================================================================

def _run_one_repeat(method_type, params, subset, tag_k):
    """Discretize + TAG-learn one subset; time only the TAG step.
    Returns {'status':'ok', 'learn_time', 'n_states', 'n_edges'} or
    {'status':'failed', 'error_type', 'error_msg'}."""
    tmp_file = os.path.join(
        tempfile.gettempdir(),
        f"scaling_{uuid.uuid4().hex}.txt",
    )
    try:
        traces, bins_c = _discretize(method_type, params, subset)
        symbolic_traces, _, _ = map_bins_to_symbols(traces, bins_c)
        format_output(symbolic_traces, tmp_file)

        t0 = time.perf_counter()
        learner = TALearner(tmp_file, display=False, k=tag_k)
        learn_time = time.perf_counter() - t0

        return {
            "status":     "ok",
            "learn_time": learn_time,
            "n_states":   len(learner.ta.states),
            "n_edges":    len(learner.ta.edges),
        }
    except Exception as e:
        return {
            "status":     "failed",
            "error_type": type(e).__name__,
            "error_msg":  str(e),
        }
    finally:
        if os.path.exists(tmp_file):
            try:
                os.remove(tmp_file)
            except OSError:
                pass


# =============================================================================
# SCALING EXPERIMENT FOR ONE (dataset, method, params, k)
# =============================================================================

def run_scaling_experiment(
        dataset_name, all_data, method_type, params,
        max_traces, repeats, tag_k,
        json_log, json_log_path,
        stop_on_failure=False,
):
    trace_counts = list(range(1, max_traces + 1))
    learn_time_stats, state_stats, edge_stats = [], [], []
    completed_n_count = 0

    for n in trace_counts:
        learn_times, states, edges = [], [], []
        n_failed = 0
        last_error_type = None

        for repeat_id in range(repeats):
            try:
                subset = _subset_for_repeat(all_data, n, repeat_id)
            except IndexError:
                n_failed += 1
                last_error_type = "PoolExhausted"
                continue

            r = _run_one_repeat(method_type, params, subset, tag_k)
            if r["status"] == "ok":
                learn_times.append(r["learn_time"])
                states.append(r["n_states"])
                edges.append(r["n_edges"])
            else:
                n_failed += 1
                last_error_type = r["error_type"]

        learn_time_stats.append(_stat_summary(learn_times))
        state_stats.append(_stat_summary(states))
        edge_stats.append(_stat_summary(edges))
        completed_n_count += 1

        n_ok = len(learn_times)
        if n_ok > 0:
            failed_flag = "" if n_failed == 0 else f" [{n_failed}/{repeats} FAILED]"
            print(
                f"  n={n:3d}  learn={learn_time_stats[-1]['median']:.3f}s | "
                f"states={int(state_stats[-1]['median']):3d}  "
                f"edges={int(edge_stats[-1]['median']):3d}"
                f"{failed_flag}",
                flush=True,
            )
        else:
            print(f"  n={n:3d}  ALL {repeats} REPEATS FAILED "
                  f"({last_error_type})", flush=True)

        # Save after every n for crash resilience
        with open(json_log_path, "w") as f:
            json.dump(json_log, f, indent=2)

        if n_ok == 0 and stop_on_failure:
            remaining = len(trace_counts) - completed_n_count
            print(f"  Stopping sweep at n={n} ({method_type}); "
                  f"skipping {remaining} larger n values.", flush=True)
            break

    return {
        "dataset":      dataset_name,
        "method":       method_type,
        "params":       params,
        "tag_k":        tag_k,
        "trace_counts": trace_counts[:completed_n_count],
        "learn_time":   learn_time_stats,
        "n_states":     state_stats,
        "n_edges":      edge_stats,
    }


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":

    BASE_DIR = ROOT
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    OUT_DIR = (BASE_DIR / "Data" / "Graphs" / "ScalingExperiments"
               / timestamp / os.environ.get("SLURM_JOB_ID", "local"))
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Run folder: {OUT_DIR}\n")

    MAX_TRACES = 100
    REPEATS = 5
    TAG_K_VALUES = [4]

    DATASETS = [
        ("clean", BASE_DIR / "Data" / "synthetic_data-absolute" / "clean_train"),
        ("noisy", BASE_DIR / "Data" / "synthetic_data-absolute" / "noisy_train"),
    ]

    EXPERIMENTS = [
        ("naive",   {"bins": 5}),
        ("sax",     {"w": 144, "bins": 5}),
        ("persist", {"bins": 6}),
    ]

    POOL_SIZE_NEEDED = REPEATS * MAX_TRACES
    print(f"Pool size required (REPEATS × MAX_TRACES): "
          f"{REPEATS} × {MAX_TRACES} = {POOL_SIZE_NEEDED} per dataset\n")

    save_config(OUT_DIR, TAG_K_VALUES, MAX_TRACES, REPEATS,
                DATASETS, EXPERIMENTS, POOL_SIZE_NEEDED)

    print("\n=== Ensuring trace pools and loading ===")
    dataset_data = {}
    for dataset_name, trace_folder in DATASETS:
        all_files = ensure_pool_size(dataset_name, trace_folder, POOL_SIZE_NEEDED)
        traces = csv_to_temp_time_list(input_files=all_files)
        dataset_data[dataset_name] = traces
        print(f"  Loaded {len(traces)} traces for '{dataset_name}'.")

    for tag_k in TAG_K_VALUES:
        print(f"\n{'#' * 70}\n### TAG k = {tag_k}\n{'#' * 70}\n")

        k_dir = OUT_DIR / f"k{tag_k}"
        k_dir.mkdir(parents=True, exist_ok=True)
        json_log_path = str(k_dir / "scaling_log.json")

        log = {
            "timestamp":  timestamp,
            "tag_k":      tag_k,
            "repeats":    REPEATS,
            "max_traces": MAX_TRACES,
            "results":    [],
        }

        for dataset_name, _ in DATASETS:
            print(f"\n{'=' * 70}\nDATASET: {dataset_name}  (k={tag_k})\n{'=' * 70}")
            all_data = dataset_data[dataset_name]

            for method_type, params in EXPERIMENTS:
                print(f"\n{'-' * 60}\nMethod : {method_type.upper()}\n"
                      f"Params : {params}\n{'-' * 60}")
                stop = STOP_ON_FAILURE_PER_METHOD.get(method_type, False)

                # Placeholder for crash resilience: if the runner dies mid-variant,
                # the plotter will see status='in_progress' and skip this entry.
                log["results"].append({
                    "dataset": dataset_name,
                    "method":  method_type,
                    "params":  params,
                    "tag_k":   tag_k,
                    "status":  "in_progress",
                })
                with open(json_log_path, "w") as f:
                    json.dump(log, f, indent=2)

                result = run_scaling_experiment(
                    dataset_name=dataset_name,
                    all_data=all_data,
                    method_type=method_type,
                    params=params,
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

        print(f"\n  k={tag_k} done. Results in: {k_dir}\n")

    print(f"\nAll done. Results in: {OUT_DIR}")