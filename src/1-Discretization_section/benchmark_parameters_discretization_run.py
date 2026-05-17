"""
run_benchmark.py
================
Discretization + TA-learning benchmark across Naive, SAX, and Persist for
multiple TAG k values.

Per-trace design:
  For each variant (method, params) and each TAG k value, fit one TA per
  trace and report aggregate statistics (median, min, max) across the 20
  traces.

Plot reference traces:
  Three reference traces are chosen at startup and shared across all
  methods, so signal plots can be compared apples-to-apples (same
  underlying trace). The plotter produces three signal figures per
  method, one per reference trace.

Crash-resilient design (for SLURM):
  - Recursion limit raised at startup so TAG's recursive traversals
    don't hit Python's 1000-frame default for larger alphabets.
  - Each variant wrapped in try/except. Failures logged as
    status="failed"; sweep continues.
  - benchmark_log.json saved to disk after every variant.
  - Persist uses stop_on_failure=True.

For each TAG k value, results go to TA_Benchmark/<timestamp>/k<n>/.
"""

import json
import os
import sys
import tempfile
import time
import uuid
from datetime import datetime
from pathlib import Path

import numpy as np

sys.setrecursionlimit(50000)

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
# Reference-trace selection
# ---------------------------------------------------------------------------

# NEW: how many reference traces to track per variant for plotting
N_REFERENCE_TRACES = 3

# NEW: which trace indices to use as references.
# Default: first, middle, last. Other options below.
def pick_reference_trace_indices(n_traces, v_raws):
    """
    Choose 3 trace indices to serve as plotting references across methods.

    Current strategy: first, middle, last (simple and reproducible).

    Alternatives (commented out):
      - By signal variance ("easy/median/hard"): sort by std(v_raw).
      - By peak-to-peak range: sort by ptp(v_raw).
    """
    # Strategy A: fixed indices (first, middle, last) — current default
    return [0, n_traces // 2, n_traces - 1]

    # Strategy B: by signal variance — uncomment to enable
    # variances = np.array([float(np.std(v)) for v in v_raws])
    # order = np.argsort(variances)   # ascending: low variance first
    # return [int(order[0]),                    # easiest (low variance)
    #         int(order[n_traces // 2]),        # median
    #         int(order[-1])]                   # hardest (high variance)


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
    """Map a discretized trace back to bin-center values for MAE computation."""
    bin_centers = (bins_celsius[:-1] + bins_celsius[1:]) / 2
    times = np.array([t for _, t in discretized_trace], dtype=float)
    labels = np.array([l for l, _ in discretized_trace], dtype=int)
    assert (labels >= 0).all() and (labels < len(bin_centers)).all(), (
        f"Labels out of [0, {len(bin_centers)-1}]: "
        f"min={labels.min()}, max={labels.max()}"
    )
    return times, bin_centers[labels]


def compute_errors(t_disc, v_disc, t_raw, v_raw):
    """Compute MAE between a discretized signal and the raw signal."""
    v_raw_interp = np.interp(t_disc, t_raw, v_raw)
    residuals = v_disc - v_raw_interp
    mae = float(np.mean(np.abs(residuals)))
    return mae, residuals


# ---------------------------------------------------------------------------
# Per-trace discretization
# ---------------------------------------------------------------------------

def discretize_one_trace(method_type, params, trace_data):
    """Fit discretization on a single trace and return (trace, bins_c)."""
    if method_type == "naive":
        traces, bins_c = equal_width_discretization([trace_data], k=params["bins"])
        return traces[0], bins_c
    elif method_type == "persist":
        ts = flatten_traces_to_ts([trace_data])
        persist_obj = Persist(
            ts, break_min=params["bins"], break_max=params["bins"],
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
# TA learning
# ---------------------------------------------------------------------------

def learn_ta_from_one_trace(discretized_trace, bins_c, tag_k):
    """Learn a TA from one discretized trace."""
    symbolic_traces, _, _ = map_bins_to_symbols([discretized_trace], bins_c)
    tmp_file = os.path.join(
        tempfile.gettempdir(),
        f"tag_input_{uuid.uuid4().hex}.txt",
    )
    try:
        format_output(symbolic_traces, tmp_file)
        learner = TALearner(tmp_file, display=False, k=tag_k)
        n_states = len(learner.ta.states)
        n_edges = len(learner.ta.edges)
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
                pass


# ---------------------------------------------------------------------------
# Per-variant experiment
# ---------------------------------------------------------------------------

def run_variant(label, method_type, params, data_lists, trace_files,
                t_raws, v_raws, tag_k, reference_indices):   # NEW param
    """Outer wrapper with crash protection."""
    try:
        return _run_variant_inner(
            label, method_type, params, data_lists, trace_files,
            t_raws, v_raws, tag_k, reference_indices,
        )
    except RecursionError as e:
        msg = f"RecursionError (likely TAG state traversal): {e}"
        print(f"    FAILED — {msg}", flush=True)
        return _failed_result(label, method_type, params, tag_k,
                              "RecursionError", str(e))
    except MemoryError as e:
        msg = f"MemoryError: {e}"
        print(f"    FAILED — {msg}", flush=True)
        return _failed_result(label, method_type, params, tag_k,
                              "MemoryError", str(e))
    except Exception as e:
        error_type = type(e).__name__
        msg = f"{error_type}: {str(e)[:200]}"
        print(f"    FAILED — {msg}", flush=True)
        return _failed_result(label, method_type, params, tag_k,
                              error_type, str(e))


def _failed_result(label, method_type, params, tag_k, error_type, error_msg):
    return {
        "label":       label,
        "method_type": method_type,
        "params":      params,
        "tag_k":       tag_k,
        "status":      "failed",
        "error_type":  error_type,
        "error_msg":   error_msg,
    }


def _run_variant_inner(label, method_type, params, data_lists, trace_files,
                       t_raws, v_raws, tag_k, reference_indices):
    """Body of one variant. Stores plot data for each reference trace."""
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

        discretized_trace, bins_c = discretize_one_trace(method_type, params, trace_data)
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
            actual_bins -= 1
        per_trace_actual_bins.append(actual_bins)

        t_d, v_d = discretized_to_step(discretized_trace, bins_c)
        mae, resids = compute_errors(t_d, v_d, t_raws[i], v_raws[i])
        per_trace_mae.append(mae)
        per_trace_resids.append(resids)
        per_trace_steps.append((t_d, v_d))

    # NEW: build plot data for each reference trace
    reference_plot_data = []
    for ref_idx in reference_indices:
        reference_plot_data.append({
            "trace_index":     ref_idx,
            "trace_path":      trace_files[ref_idx],
            "mae":             per_trace_mae[ref_idx],
            "n_states":        per_trace_states[ref_idx],
            "n_edges":         per_trace_edges[ref_idx],
            "actual_bins":     per_trace_actual_bins[ref_idx],
            "consistent":      per_trace_consistent[ref_idx],
            "t_d":             per_trace_steps[ref_idx][0].tolist(),
            "v_d":             per_trace_steps[ref_idx][1].tolist(),
            "resids":          per_trace_resids[ref_idx].tolist(),
        })

    # Backward-compat: also keep the old "median MAE" trace
    median_idx = int(np.argsort(per_trace_mae)[len(per_trace_mae) // 2])

    result = {
        "label":              label,
        "method_type":        method_type,
        "params":             params,
        "tag_k":              tag_k,
        "status":             "ok",

        "mae_median":         float(np.median(per_trace_mae)),
        "mae_mean":           float(np.mean(per_trace_mae)),
        "mae_std":            float(np.std(per_trace_mae)),
        "mae_min":            float(np.min(per_trace_mae)),
        "mae_max":            float(np.max(per_trace_mae)),

        "time_median":        float(np.median(per_trace_times)),
        "time_mean":          float(np.mean(per_trace_times)),
        "time_std":           float(np.std(per_trace_times)),
        "time_min":           float(np.min(per_trace_times)),
        "time_max":           float(np.max(per_trace_times)),

        "n_states_median":    float(np.median(per_trace_states)),
        "n_states_mean":      float(np.mean(per_trace_states)),
        "n_states_std":       float(np.std(per_trace_states)),
        "n_states_min":       int(np.min(per_trace_states)),
        "n_states_max":       int(np.max(per_trace_states)),

        "n_edges_median":     float(np.median(per_trace_edges)),
        "n_edges_mean":       float(np.mean(per_trace_edges)),
        "n_edges_std":        float(np.std(per_trace_edges)),
        "n_edges_min":        int(np.min(per_trace_edges)),
        "n_edges_max":        int(np.max(per_trace_edges)),

        "n_consistent":       int(sum(per_trace_consistent)),
        "n_total":            n_traces,

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

        # Backward-compat: keep median-MAE trace as the primary plot trace
        "plot_trace_path":    trace_files[median_idx],
        "plot_trace_index":   median_idx,
        "plot_t_d":           per_trace_steps[median_idx][0].tolist(),
        "plot_v_d":           per_trace_steps[median_idx][1].tolist(),
        "plot_resids":        per_trace_resids[median_idx].tolist(),

        # NEW: reference traces (3 per variant, same indices across all methods)
        "reference_traces":   reference_plot_data,
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


# ---------------------------------------------------------------------------
# Run all variants for one method
# ---------------------------------------------------------------------------

def run_experiment(method_name, variants_dict, data_lists, trace_files,
                   t_raws, v_raws, tag_k, reference_indices,   # NEW param
                   log, log_path, log_method_key,
                   stop_on_failure=False):
    results = []
    log["methods"][log_method_key] = results

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
            reference_indices=reference_indices,
        )
        results.append(result)

        with open(log_path, "w") as f:
            json.dump(log, f, indent=2)

        if result.get("status") == "failed" and stop_on_failure:
            n_remaining = len(variants_dict) - len(results)
            print(f"  [{method_name}] Stopping after failure at '{label}'; "
                  f"skipping {n_remaining} remaining variants.", flush=True)
            break

    def _sort_key(r):
        if r.get("status") == "failed":
            return (1, float("inf"))
        return (0, r["mae_median"])

    results.sort(key=_sort_key)
    log["methods"][log_method_key] = results

    with open(log_path, "w") as f:
        json.dump(log, f, indent=2)

    return results


# ---------------------------------------------------------------------------
# Config file
# ---------------------------------------------------------------------------

def save_config(run_dir, tag_k_values, trace_files, reference_indices,   # NEW param
                naive_vars, sax_vars, persist_vars):
    def _variant_lines(variants):
        return [
            f"    {label:30s} -> "
            + ", ".join(f"{k}={v}" for k, v in params.items())
            for label, (_, params) in variants.items()
        ]

    lines = [
        "=" * 60,
        "Run configuration -- Benchmark (per-trace, crash-resilient)",
        "=" * 60,
        "",
        f"Timestamp     : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"TAG k values  : {tag_k_values}",
        f"Num traces    : {len(trace_files)}",
        f"Recursion lim : {sys.getrecursionlimit()}",
        "",
        f"Reference plot traces (used for signal comparisons):",
        ]
    for ref_idx in reference_indices:
        lines.append(f"    Trace #{ref_idx}: {trace_files[ref_idx]}")

    lines += ["", "--- All trace files ---"]
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
    print(f"Python recursion limit raised to {sys.getrecursionlimit()}")

    TAG_K_VALUES = [2, 4]

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
        for b in [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
    }

    print("Loading traces...")
    data_lists = csv_to_temp_time_list(input_files=trace_files)
    t_raws, v_raws = zip(*[load_trace(p) for p in trace_files])
    t_raws = list(t_raws)
    v_raws = list(v_raws)
    print(f"Loaded {len(data_lists)} traces.\n")

    # NEW: pick reference traces once, share across all methods and k values
    reference_indices = pick_reference_trace_indices(len(trace_files), v_raws)
    print(f"Reference plot trace indices: {reference_indices}")
    for ref_idx in reference_indices:
        print(f"  [{ref_idx}] {trace_files[ref_idx]}")
    print()

    print("=== Config ===")
    save_config(run_dir, TAG_K_VALUES, trace_files, reference_indices,
                naive_vars, sax_vars, persist_vars)

    for tag_k in TAG_K_VALUES:
        print(f"\n{'#' * 60}")
        print(f"### TAG k = {tag_k}")
        print(f"{'#' * 60}")

        k_dir = run_dir / f"k{tag_k}"
        k_dir.mkdir(parents=True, exist_ok=True)
        log_path = str(k_dir / "benchmark_log.json")

        log = {
            "timestamp":         timestamp,
            "tag_k":             tag_k,
            "trace_files":       trace_files,
            "reference_indices": reference_indices,   # NEW
            "t_raw":             t_raws[0].tolist(),
            "v_raw":             v_raws[0].tolist(),
            "methods":           {},
        }

        print(f"\n=== Naive (k={tag_k}) ===")
        run_experiment(
            "Naive", naive_vars, data_lists, trace_files,
            t_raws, v_raws, tag_k, reference_indices,
            log=log, log_path=log_path, log_method_key="Naive",
            stop_on_failure=False,
        )

        print(f"\n=== SAX (k={tag_k}) ===")
        run_experiment(
            "SAX", sax_vars, data_lists, trace_files,
            t_raws, v_raws, tag_k, reference_indices,
            log=log, log_path=log_path, log_method_key="SAX",
            stop_on_failure=False,
        )

        print(f"\n=== Persist (k={tag_k}) ===")
        run_experiment(
            "Persist", persist_vars, data_lists, trace_files,
            t_raws, v_raws, tag_k, reference_indices,
            log=log, log_path=log_path, log_method_key="Persist",
            stop_on_failure=True,
        )

        # Per-k summary
        print(f"\nk={tag_k}: best variant per method (by median MAE):")
        for method_name, results in log["methods"].items():
            best = next((r for r in results if r.get("status") == "ok"), None)
            if best is None:
                n_failed = sum(1 for r in results if r.get("status") == "failed")
                print(f"  {method_name:8s}: all {n_failed} variants failed")
                continue
            n_inc = best["n_total"] - best["n_consistent"]
            inc = f" [{n_inc}/{best['n_total']} INCONSISTENT]" if n_inc else ""
            print(
                f"  {method_name:8s}: {best['label']:25s} "
                f"MAE={best['mae_median']:.3f}, "
                f"states={best['n_states_median']:.1f}, "
                f"time={best['time_median']:.3f}s"
                f"{inc}"
            )

        n_failures = sum(
            1
            for results in log["methods"].values()
            for r in results
            if r.get("status") == "failed"
        )
        if n_failures > 0:
            print(f"\nk={tag_k}: {n_failures} variants failed across all methods.")
            for method_name, results in log["methods"].items():
                for r in results:
                    if r.get("status") == "failed":
                        print(f"  {method_name} | {r['label']} | "
                              f"{r['error_type']}: {r['error_msg'][:100]}")

    print(f"\nAll done. Results in: {run_dir}")