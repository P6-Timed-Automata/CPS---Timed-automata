"""
run_benchmark.py
================
Discretization + TA-learning benchmark across Naive, SAX, and Persist
for TAG k=4.

For each variant (method, params), one TA is fit per trace on the 20-trace
noisy synthetic test set; median / min / max statistics are reported across
the 20 per-trace TAs. A single reference trace (index 0) is recorded so all
methods can be compared on the same underlying signal in plotter outputs.

Crash resilience: each variant is wrapped in try/except; failures are
logged with status="failed" and the sweep continues. The benchmark log
is saved after every variant. Persist uses stop_on_failure=True.

Output: TA_Benchmark/<timestamp>/k<n>/benchmark_log.json
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
# Trace loading
# ---------------------------------------------------------------------------

def load_trace(path):
    data = np.genfromtxt(path, delimiter=";", skip_header=1)
    return data[:, 0], data[:, 1]


# ---------------------------------------------------------------------------
# Discretization + MAE
# ---------------------------------------------------------------------------

def discretized_to_step(discretized_trace, bins_celsius):
    """Map discretized symbols back to bin-centre values for MAE computation."""
    bin_centers = (bins_celsius[:-1] + bins_celsius[1:]) / 2
    times = np.array([t for _, t in discretized_trace], dtype=float)
    labels = np.array([l for l, _ in discretized_trace], dtype=int)
    assert (labels >= 0).all() and (labels < len(bin_centers)).all(), (
        f"Labels out of [0, {len(bin_centers) - 1}]: "
        f"min={labels.min()}, max={labels.max()}"
    )
    return times, bin_centers[labels]


def compute_errors(t_disc, v_disc, t_raw, v_raw):
    """MAE between discretized step signal and raw signal."""
    v_raw_interp = np.interp(t_disc, t_raw, v_raw)
    residuals = v_disc - v_raw_interp
    return float(np.mean(np.abs(residuals))), residuals


def discretize_one_trace(method_type, params, trace_data):
    """Fit discretization on one trace; return (discretized_trace, bins_celsius)."""
    if method_type == "naive":
        traces, bins_c = equal_width_discretization([trace_data], k=params["bins"])
        return traces[0], bins_c

    if method_type == "persist":
        ts = flatten_traces_to_ts([trace_data])
        persist_obj = Persist(
            ts, break_min=params["bins"], break_max=params["bins"],
            skip=np.array([4, 4]),
        )
        bins_c = get_best_bins(persist_obj, ts)
        traces = discretize_traces_with_bins([trace_data], bins_c)
        return traces[0], bins_c

    if method_type == "sax":
        traces, bins_z, mean_, std_ = sax_discretization_multi(
            [trace_data], w=params["w"], k=params["bins"]
        )
        bins_c = sax_bins_in_original_space(bins_z, mean_, std_)
        return traces[0], bins_c

    raise ValueError(f"Unknown method: {method_type}")


# ---------------------------------------------------------------------------
# TA learning
# ---------------------------------------------------------------------------

def learn_ta_from_one_trace(discretized_trace, bins_c, tag_k):
    """Learn a TA from one discretized trace.
    Returns (n_states, n_edges, is_consistent)."""
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
        return n_states, n_edges, (inconsistency_count == 0)
    finally:
        if os.path.exists(tmp_file):
            try:
                os.remove(tmp_file)
            except OSError:
                pass


# ---------------------------------------------------------------------------
# Per-variant experiment (with crash protection)
# ---------------------------------------------------------------------------

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


def run_variant(label, method_type, params, data_lists, trace_files,
                t_raws, v_raws, tag_k, reference_index):
    try:
        return _run_variant_inner(
            label, method_type, params, data_lists, trace_files,
            t_raws, v_raws, tag_k, reference_index,
        )
    except Exception as e:
        error_type = type(e).__name__
        print(f"    FAILED — {error_type}: {str(e)[:200]}", flush=True)
        return _failed_result(label, method_type, params, tag_k,
                              error_type, str(e))


def _run_variant_inner(label, method_type, params, data_lists, trace_files,
                       t_raws, v_raws, tag_k, reference_index):
    """Discretize + learn TA per trace; aggregate stats; record reference trace."""
    n_traces = len(data_lists)

    per_mae,    per_resids,   per_steps      = [], [], []
    per_times,  per_states,   per_edges      = [], [], []
    per_consistent = []

    for i, trace_data in enumerate(data_lists):
        t0 = time.perf_counter()
        discretized_trace, bins_c = discretize_one_trace(method_type, params, trace_data)
        n_states, n_edges, is_consistent = learn_ta_from_one_trace(
            discretized_trace, bins_c, tag_k
        )
        per_times.append(time.perf_counter() - t0)
        per_states.append(n_states)
        per_edges.append(n_edges)
        per_consistent.append(is_consistent)

        t_d, v_d = discretized_to_step(discretized_trace, bins_c)
        mae, resids = compute_errors(t_d, v_d, t_raws[i], v_raws[i])
        per_mae.append(mae)
        per_resids.append(resids)
        per_steps.append((t_d, v_d))

    # Reference trace for the signal-overlay plot (same index across all
    # variants and methods so the figures share the same raw trace).
    ref_t_d, ref_v_d = per_steps[reference_index]
    reference_trace = {
        "trace_path": trace_files[reference_index],
        "t_d":        ref_t_d.tolist(),
        "v_d":        ref_v_d.tolist(),
        "resids":     per_resids[reference_index].tolist(),
    }

    result = {
        "label":           label,
        "method_type":     method_type,
        "params":          params,
        "tag_k":           tag_k,
        "status":          "ok",

        "mae_median":      float(np.median(per_mae)),
        "mae_min":         float(np.min(per_mae)),
        "mae_max":         float(np.max(per_mae)),

        "time_median":     float(np.median(per_times)),
        "time_std":        float(np.std(per_times)),

        "n_states_median": float(np.median(per_states)),
        "n_states_min":    int(np.min(per_states)),
        "n_states_max":    int(np.max(per_states)),

        "n_edges_median":  float(np.median(per_edges)),
        "n_edges_min":     int(np.min(per_edges)),
        "n_edges_max":     int(np.max(per_edges)),

        "n_consistent":    int(sum(per_consistent)),
        "n_total":         n_traces,

        "reference_traces": [reference_trace],
    }

    n_inc = result["n_total"] - result["n_consistent"]
    inc_flag = "" if n_inc == 0 else f" [{n_inc}/{result['n_total']} INCONSISTENT]"
    print(
        f"    MAE={result['mae_median']:.3f} "
        f"(range {result['mae_min']:.3f}-{result['mae_max']:.3f}) | "
        f"time={result['time_median']:.3f}±{result['time_std']:.3f}s | "
        f"states={result['n_states_median']:.1f} "
        f"({result['n_states_min']}-{result['n_states_max']}) | "
        f"edges={result['n_edges_median']:.1f}"
        f"{inc_flag}"
    )
    return result


# ---------------------------------------------------------------------------
# Method-level loop
# ---------------------------------------------------------------------------

def run_experiment(method_name, variants_dict, data_lists, trace_files,
                   t_raws, v_raws, tag_k, reference_index,
                   log, log_path, log_method_key,
                   stop_on_failure=False):
    results = []
    log["methods"][log_method_key] = results

    for label, (method_type, params) in variants_dict.items():
        print(f"  [{method_name}] {label} ...", flush=True)
        result = run_variant(
            label=label, method_type=method_type, params=params,
            data_lists=data_lists, trace_files=trace_files,
            t_raws=t_raws, v_raws=v_raws,
            tag_k=tag_k, reference_index=reference_index,
        )
        results.append(result)

        # Save after every variant for crash resilience
        with open(log_path, "w") as f:
            json.dump(log, f, indent=2)

        if result.get("status") == "failed" and stop_on_failure:
            n_remaining = len(variants_dict) - len(results)
            print(f"  [{method_name}] Stopping after failure at '{label}'; "
                  f"skipping {n_remaining} remaining variants.", flush=True)
            break

    # ok variants ascending by MAE; failures pushed to the end
    results.sort(key=lambda r: (1, float("inf")) if r.get("status") == "failed"
    else (0, r["mae_median"]))
    log["methods"][log_method_key] = results
    with open(log_path, "w") as f:
        json.dump(log, f, indent=2)


# ---------------------------------------------------------------------------
# Config (reproducibility record)
# ---------------------------------------------------------------------------

def save_config(run_dir, tag_k_values, trace_files, reference_index,
                naive_vars, sax_vars, persist_vars):
    def _lines(variants):
        return [
            f"    {label:30s} -> "
            + ", ".join(f"{k}={v}" for k, v in params.items())
            for label, (_, params) in variants.items()
        ]

    lines = [
        "=" * 60,
        "Run configuration -- Benchmark",
        "=" * 60,
        "",
        f"Timestamp     : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"TAG k values  : {tag_k_values}",
        f"Num traces    : {len(trace_files)}",
        f"Reference     : trace #{reference_index} ({trace_files[reference_index]})",
        "",
        "--- All trace files ---",
        *[f"  {p}" for p in trace_files],
        "",
        "--- Naive variants ---",   *_lines(naive_vars),
        "",
        "--- SAX variants ---",     *_lines(sax_vars),
        "",
        "--- Persist variants ---", *_lines(persist_vars),
        "",
        "--- Output folder ---",
        f"  {run_dir}",
        "",
        "=" * 60,
        ]
    (run_dir / "config.txt").write_text("\n".join(lines))
    print(f"  Config saved: {run_dir / 'config.txt'}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    BASE_DIR = Path(__file__).resolve().parent.parent.parent

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_dir = BASE_DIR / "Data" / "Graphs" / "TA_Benchmark" / timestamp
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"Run folder: {run_dir}")

    TAG_K_VALUES = [4]

    base = BASE_DIR / "Data" / "synthetic_data" / "noisy_test"
    trace_files = sorted(str(p) for p in base.glob("noisy_test_tid*.csv"))[:20]
    assert len(trace_files) == 20, f"Expected 20 traces, found {len(trace_files)}"

    naive_vars = {
        f"bins={b}": ("naive", {"bins": b}) for b in range(2, 16)
    }
    sax_vars = {
        f"w={w}, bins={b}": ("sax", {"w": w, "bins": b})
        for w in [24, 48, 96, 144] for b in [5, 10, 15]
    }
    persist_vars = {
        f"k={b}": ("persist", {"bins": b}) for b in range(2, 16)
    }

    print("Loading traces...")
    data_lists = csv_to_temp_time_list(input_files=trace_files)
    t_raws, v_raws = zip(*[load_trace(p) for p in trace_files])
    t_raws, v_raws = list(t_raws), list(v_raws)
    print(f"Loaded {len(data_lists)} traces.\n")

    # First trace as the shared reference, so all methods plot the same
    # underlying signal in the figure outputs.
    reference_index = 0
    print(f"Reference plot trace: #{reference_index} "
          f"{trace_files[reference_index]}\n")

    save_config(run_dir, TAG_K_VALUES, trace_files, reference_index,
                naive_vars, sax_vars, persist_vars)

    for tag_k in TAG_K_VALUES:
        print(f"\n{'#' * 60}\n### TAG k = {tag_k}\n{'#' * 60}")

        k_dir = run_dir / f"k{tag_k}"
        k_dir.mkdir(parents=True, exist_ok=True)
        log_path = str(k_dir / "benchmark_log.json")

        log = {
            "timestamp":       timestamp,
            "tag_k":           tag_k,
            "trace_files":     trace_files,
            "reference_index": reference_index,
            "methods":         {},
        }

        print(f"\n=== Naive (k={tag_k}) ===")
        run_experiment("Naive", naive_vars, data_lists, trace_files,
                       t_raws, v_raws, tag_k, reference_index,
                       log=log, log_path=log_path, log_method_key="Naive")

        print(f"\n=== SAX (k={tag_k}) ===")
        run_experiment("SAX", sax_vars, data_lists, trace_files,
                       t_raws, v_raws, tag_k, reference_index,
                       log=log, log_path=log_path, log_method_key="SAX")

        print(f"\n=== Persist (k={tag_k}) ===")
        run_experiment("Persist", persist_vars, data_lists, trace_files,
                       t_raws, v_raws, tag_k, reference_index,
                       log=log, log_path=log_path, log_method_key="Persist",
                       stop_on_failure=True)

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
            print(f"  {method_name:8s}: {best['label']:25s} "
                  f"MAE={best['mae_median']:.3f}, "
                  f"states={best['n_states_median']:.1f}, "
                  f"time={best['time_median']:.3f}s{inc}")

    print(f"\nAll done. Results in: {run_dir}")