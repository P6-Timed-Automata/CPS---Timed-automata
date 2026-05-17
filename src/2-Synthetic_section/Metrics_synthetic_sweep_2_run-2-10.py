"""
exp_test_noise_run.py
=====================
Test-time noise tolerance — clean-trained vs noisy-trained TAs.

Complements exp_53 (which sweeps TRAINING noise against a fixed clean test set)
by sweeping TEST noise against two fixed training regimes. Lets you compare
deployment-time noise robustness of a TA learned from clean data vs. one
learned from noisy data.

Design
------
For each training_condition in {clean, noisy}:
    For each method, seed:
        - Generate one training set using CONFIG[condition] profile
        - Train the TA once (discretizer + TAG learner)
        - For each test_noise_level:
            - Generate test positives with this seed and noise level
              (clean trace shape, only noise varies)
            - Discretize using training bins (no leakage)
            - Evaluate against generated test_pos + fixed neg_traces
            - Record metrics
Aggregate across seeds (median / min / max), keeping the same status
schema (ok / partial / failed) as exp_53.

Crash resilience
----------------
- Training failures mark all noise levels failed for that (condition, method,
  seed). No point evaluating without a TA.
- Eval failures are recorded per-cell.
- Incremental save after each (condition, method) block.

Run:  python exp_test_noise_run.py
Plot: python exp_test_noise_plot.py
"""

import json
import os
import subprocess
import sys
import tempfile
from collections import defaultdict
from datetime import datetime
from pathlib import Path

# Raise recursion limit before TAG imports.
sys.setrecursionlimit(50000)

import numpy as np
from scipy.stats import norm as scipy_norm

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from Generate_data import CONFIG
from Generators import generate_trace_set, NEG_MODE_NAMES, load_traces_by_mode
from Pipeline import (
    to_list_format,
    _write_collapsed,
    _preprocess_test,
    _preprocess_test_sax,
)
from Discretization.naive import equal_width_discretization
from Discretization.sax import sax_discretization_multi
from Discretization.persist import (
    Persist, get_best_bins, discretize_traces_with_bins, flatten_traces_to_ts,
)
from Discretization.discretizationSetup import map_bins_to_symbols
from TAG.TALearner import TALearner


# =============================================================================
# CONFIG
# =============================================================================

N_TRAIN = CONFIG["n_train"]

# Test-time noise levels. Use the same grid as exp_53 for direct comparison.
TEST_NOISE_LEVELS = [0.02, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.7, 1.0,1.5,2.0,3.0]


N_SEEDS = 5
TAG_K   = 2

TRAINING_CONDITIONS = ["clean", "noisy"]

METHODS = [
    ("naive",   {"bins": 10}),
    ("sax144",  {"w": 144, "bins": 10}),
    ("sax48",   {"w": 48,  "bins": 10}),
    ("sax24",   {"w": 24,  "bins": 10}),
    ("persist", {"bins": 11}),
]

# Test traces use CLEAN trace shape (tight base variation); only noise_std
# varies per level. This isolates the test-noise effect from any signal-shape
# confound and keeps both training conditions evaluated against identical sets.
TEST_PROFILE = {
    "base_temp":     CONFIG["clean"]["base_temp"],
    "amplitude":     CONFIG["clean"]["amplitude"],
    "base_temp_std": CONFIG["clean"]["base_temp_std"],
    "amplitude_std": CONFIG["clean"]["amplitude_std"],
    "phase_std_h":   CONFIG["clean"]["phase_std_h"],
}

N_TEST_POS = 60


# =============================================================================
# TRAIN / EVAL HELPERS  (split so one trained TA can be reused across noise levels)
# =============================================================================

def _train_ta(method, params, train_traces, tag_k):
    """
    Discretize training data and learn a TA.

    Returns
    -------
    learner    : TALearner instance (learner.ta is the Automaton).
    bins       : bin edges in original value space (for Naive/Persist eval).
    n_symbols  : alphabet size.
    sax_aux    : None for naive/persist; for SAX, the dict of training stats
                 (w, breakpoints, global_mean, global_std) needed to apply the
                 same PAA+breakpoint scheme to test traces.

    Raises on failure; caller is responsible for try/except.
    """
    train_list = to_list_format(train_traces)
    sax_aux = None

    if method == "naive":
        traces_disc, bins = equal_width_discretization(train_list, k=params["bins"])
        n_symbols = len(bins) - 1
        sym_train, _, _ = map_bins_to_symbols(traces_disc, bins)

    elif method.startswith("sax"):
        w = params["w"]
        k = params["bins"]
        breakpoints = scipy_norm.ppf(np.linspace(0, 1, k + 1)[1:-1])
        all_v = np.concatenate(
            [np.array([v for v, _ in tr]) for tr in train_list]
        )
        global_mean = float(all_v.mean())
        global_std  = float(all_v.std()) if all_v.std() != 0 else 1.0

        traces_disc, bins_z, _, _ = sax_discretization_multi(train_list, w=w, k=k)
        bins      = np.sort(bins_z) * global_std + global_mean
        n_symbols = k
        sym_train, _, _ = map_bins_to_symbols(traces_disc, np.sort(bins_z))
        sax_aux = {
            "w":           w,
            "breakpoints": breakpoints,
            "global_mean": global_mean,
            "global_std":  global_std,
        }

    elif method == "persist":
        ts = flatten_traces_to_ts(train_list)
        persist_obj = Persist(
            ts, break_min=2, break_max=params["bins"], skip=np.array([4, 4])
        )
        bins        = get_best_bins(persist_obj, ts)
        n_symbols   = len(bins) - 1
        traces_disc = discretize_traces_with_bins(train_list, bins)
        sym_train, _, _ = map_bins_to_symbols(traces_disc, bins)

    else:
        raise ValueError(f"Unknown method: {method}")

    # Write training timed strings to a temp file and learn the TA.
    fd, tmp_path = tempfile.mkstemp(suffix=".txt", prefix="exp_tn_tmp_")
    os.close(fd)
    try:
        _write_collapsed(sym_train, tmp_path)
        learner = TALearner(tss_path=tmp_path, display=False, k=tag_k)
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

    return learner, bins, n_symbols, sax_aux


def _eval_ta(learner, bins, n_symbols, sax_aux, method,
             test_pos_traces, test_neg_traces, neg_modes):
    """
    Discretize test data using training bins/SAX stats, then evaluate the
    learned TA as a binary classifier. Mirrors Pipeline.run_pipeline's eval
    block. Returns (overall_dict, per_mode_dict).
    """
    pos_list = to_list_format(test_pos_traces)
    neg_list = to_list_format(test_neg_traces)

    if method.startswith("sax"):
        pos_strings = _preprocess_test_sax(
            pos_list, sax_aux["w"], n_symbols,
            sax_aux["breakpoints"], sax_aux["global_mean"], sax_aux["global_std"],
        )
        neg_strings = _preprocess_test_sax(
            neg_list, sax_aux["w"], n_symbols,
            sax_aux["breakpoints"], sax_aux["global_mean"], sax_aux["global_std"],
        )
    else:
        pos_strings = _preprocess_test(pos_list, bins, n_symbols)
        neg_strings = _preprocess_test(neg_list, bins, n_symbols)

    overall = learner.ta.evaluate_classifier(
        positive_tss=pos_strings, negative_tss=neg_strings, timed=True,
    )
    # Strip non-scalar / non-JSON fields.
    overall = {
        k: v for k, v in overall.items()
        if not isinstance(v, list) and k not in ("save_path", "run_id")
    }

    # Per-mode rejection rates.
    per_mode = {}
    if neg_modes is not None:
        mode_indices = defaultdict(list)
        for i, m in enumerate(neg_modes):
            mode_indices[m].append(i)

        for mode_idx, indices in mode_indices.items():
            mode_neg = [neg_strings[i] for i in indices]
            if not mode_neg:
                continue
            mode_metrics = learner.ta.evaluate_classifier(
                positive_tss=pos_strings, negative_tss=mode_neg, timed=True,
            )
            name = NEG_MODE_NAMES.get(mode_idx, str(mode_idx))
            per_mode[name] = {
                "NAR":       mode_metrics["NAR"],
                "rejection": 100.0 - mode_metrics["NAR"],
                "precision": mode_metrics["precision"],
                "recall":    mode_metrics["recall"],
                "f1":        mode_metrics["f1"],
            }

    return overall, per_mode


# =============================================================================
# AGGREGATION (skips failed seeds; same schema as exp_53)
# =============================================================================

def _aggregate_overall(per_seed_overall):
    if not per_seed_overall:
        return {}
    metrics = per_seed_overall[0].keys()
    out = {}
    for m in metrics:
        vals = [s[m] for s in per_seed_overall
                if isinstance(s.get(m), (int, float))]
        if not vals:
            continue
        out[m] = {
            "median":   float(np.median(vals)),
            "min":      float(np.min(vals)),
            "max":      float(np.max(vals)),
            "mean":     float(np.mean(vals)),
            "std":      float(np.std(vals)),
            "per_seed": vals,
        }
    return out


def _aggregate_per_mode(per_seed_per_mode):
    if not per_seed_per_mode:
        return {}
    mode_names = set()
    for pm in per_seed_per_mode:
        mode_names.update(pm.keys())
    out = {}
    for mode in mode_names:
        rejections = [pm[mode]["rejection"] for pm in per_seed_per_mode
                      if mode in pm]
        if rejections:
            out[mode] = {
                "rejection_median":   float(np.median(rejections)),
                "rejection_min":      float(np.min(rejections)),
                "rejection_max":      float(np.max(rejections)),
                "rejection_per_seed": rejections,
            }
    return out


# =============================================================================
# CONFIG LOGGING
# =============================================================================

def _git_hash():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT
        ).decode().strip()
    except Exception:
        return "unknown"


def _save_config(out_dir, neg_traces, base_seed):
    lines = [
        "=" * 60,
        "Run configuration — Test-Noise Sweep (clean vs noisy training)",
        "=" * 60,
        "",
        f"Timestamp     : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"Git hash      : {_git_hash()}",
        f"TAG k-future  : {TAG_K}",
        f"Seeds per pt  : {N_SEEDS}",
        f"Base seed     : {base_seed}",
        f"Recursion lim : {sys.getrecursionlimit()}",
        "",
        f"--- Test-time noise levels ({len(TEST_NOISE_LEVELS)}) ---",
        f"  {TEST_NOISE_LEVELS}",
        "",
        "--- Training conditions ---",
        ]
    for cond in TRAINING_CONDITIONS:
        p = CONFIG[cond]
        lines.append(
            f"  {cond:6s}: base_temp_std={p['base_temp_std']}  "
            f"amplitude_std={p['amplitude_std']}  "
            f"phase_std_h={p['phase_std_h']}  "
            f"noise_std={p['noise_std']}"
        )

    lines += [
        "",
        "--- Test trace profile (clean shape; only noise_std sweeps) ---",
        f"  base_temp_std : {TEST_PROFILE['base_temp_std']}",
        f"  amplitude_std : {TEST_PROFILE['amplitude_std']}",
        f"  phase_std_h   : {TEST_PROFILE['phase_std_h']}",
        "",
        "--- Methods ---",
    ]
    for method, params in METHODS:
        param_str = ", ".join(f"{k}={v}" for k, v in params.items())
        lines.append(f"  {method:8s}: {param_str}")

    n_train_runs = len(TRAINING_CONDITIONS) * len(METHODS) * N_SEEDS
    n_eval_runs  = n_train_runs * len(TEST_NOISE_LEVELS)
    lines += [
        "",
        "--- Dataset sizes ---",
        f"  train per (condition, seed) : {N_TRAIN} traces",
        f"  test positives per level    : {N_TEST_POS} traces (regenerated per seed)",
        f"  negatives                   : {len(neg_traces)} traces (fixed)",
        "",
        "--- Total runs ---",
        f"  Trainings: {len(TRAINING_CONDITIONS)} × {len(METHODS)} × "
        f"{N_SEEDS} = {n_train_runs}",
        f"  Evals:     {n_train_runs} × {len(TEST_NOISE_LEVELS)} = {n_eval_runs}",
        "",
        "--- Output folder ---",
        f"  {out_dir}",
        "",
        "=" * 60,
        ]
    (out_dir / "config.txt").write_text("\n".join(lines))
    print(f"  Saved: {out_dir / 'config.txt'}")


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    out_dir = ROOT / "Data" / "Graphs" / "TestNoise_clean_vs_noisy" / timestamp
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output folder: {out_dir}")
    print(f"Recursion limit: {sys.getrecursionlimit()}\n")

    neg_traces, neg_modes = load_traces_by_mode(
        ROOT / "Data" / "synthetic_data" / "negative"
    )

    base_seed = CONFIG["seed_clean"]
    _save_config(out_dir, neg_traces, base_seed)

    n_train_runs = len(TRAINING_CONDITIONS) * len(METHODS) * N_SEEDS
    n_eval_runs  = n_train_runs * len(TEST_NOISE_LEVELS)
    print(f"\nTotal trainings: {n_train_runs}")
    print(f"Total evals:     {n_eval_runs}\n")

    log = {
        "timestamp":           timestamp,
        "git_hash":            _git_hash(),
        "tag_k":               TAG_K,
        "n_train":             N_TRAIN,
        "n_seeds":             N_SEEDS,
        "n_test_pos":          N_TEST_POS,
        "n_neg":               len(neg_traces),
        "test_noise_levels":   TEST_NOISE_LEVELS,
        "training_conditions": TRAINING_CONDITIONS,
        "training_profiles":   {c: CONFIG[c] for c in TRAINING_CONDITIONS},
        "test_profile":        TEST_PROFILE,
        "methods":             [{"method": m, "params": p} for m, p in METHODS],
        "results":             [],
    }

    # entries[(condition, noise_level)] = entry dict; linearized into log["results"]
    # at every incremental save.
    entries = {
        (cond, noise): {
            "training_condition": cond,
            "noise_std":          noise,
            "methods":            {},
        }
        for cond in TRAINING_CONDITIONS
        for noise in TEST_NOISE_LEVELS
    }

    train_counter = 0

    for cond in TRAINING_CONDITIONS:
        cond_profile = CONFIG[cond]
        print(f"=== Training condition: {cond} ===")

        for method, params in METHODS:
            per_seed_buckets = {noise: [] for noise in TEST_NOISE_LEVELS}

            for seed_offset in range(N_SEEDS):
                seed = base_seed + seed_offset
                train_counter += 1

                # ---- Generate training data ----
                train_traces = generate_trace_set(
                    n_traces      = N_TRAIN,
                    seed          = seed,
                    base_temp     = cond_profile["base_temp"],
                    amplitude     = cond_profile["amplitude"],
                    base_temp_std = cond_profile["base_temp_std"],
                    amplitude_std = cond_profile["amplitude_std"],
                    phase_std_h   = cond_profile["phase_std_h"],
                    noise_std     = cond_profile["noise_std"],
                )

                # ---- Train TA once ----
                try:
                    learner, bins, n_symbols, sax_aux = _train_ta(
                        method, params, train_traces, TAG_K,
                    )
                    n_states = len(learner.ta.states)
                    n_edges  = len(learner.ta.edges)
                    print(
                        f"  [{train_counter:3d}/{n_train_runs}] "
                        f"{cond:5s}/{method:8s} seed={seed}  "
                        f"trained: {n_states} states, {n_edges} edges"
                    )
                except Exception as e:
                    error_type = type(e).__name__
                    error_msg  = str(e)
                    print(
                        f"  [{train_counter:3d}/{n_train_runs}] "
                        f"{cond:5s}/{method:8s} seed={seed}  "
                        f"TRAIN FAILED ({error_type}): {error_msg[:100]}"
                    )
                    # Cascade failure to all noise levels for this seed.
                    for noise in TEST_NOISE_LEVELS:
                        per_seed_buckets[noise].append({
                            "status":     "failed",
                            "seed":       seed,
                            "stage":      "train",
                            "error_type": error_type,
                            "error_msg":  error_msg,
                        })
                    continue

                # ---- Evaluate across all test noise levels ----
                for noise in TEST_NOISE_LEVELS:
                    test_pos = generate_trace_set(
                        n_traces      = N_TEST_POS,
                        seed          = seed,
                        base_temp     = TEST_PROFILE["base_temp"],
                        amplitude     = TEST_PROFILE["amplitude"],
                        base_temp_std = TEST_PROFILE["base_temp_std"],
                        amplitude_std = TEST_PROFILE["amplitude_std"],
                        phase_std_h   = TEST_PROFILE["phase_std_h"],
                        noise_std     = noise,
                    )
                    try:
                        overall, per_mode = _eval_ta(
                            learner, bins, n_symbols, sax_aux, method,
                            test_pos, neg_traces, neg_modes,
                        )
                        per_seed_buckets[noise].append({
                            "status":   "ok",
                            "seed":     seed,
                            "overall":  overall,
                            "per_mode": per_mode,
                            "n_states": n_states,
                            "n_edges":  n_edges,
                        })
                    except Exception as e:
                        per_seed_buckets[noise].append({
                            "status":     "failed",
                            "seed":       seed,
                            "stage":      "eval",
                            "error_type": type(e).__name__,
                            "error_msg":  str(e),
                        })

            # ---- Aggregate across seeds for each (cond, method, noise) cell ----
            for noise in TEST_NOISE_LEVELS:
                seed_results = per_seed_buckets[noise]
                ok_results   = [r for r in seed_results if r["status"] == "ok"]
                n_ok         = len(ok_results)
                n_failed     = N_SEEDS - n_ok

                if n_ok == 0:
                    cell_status = "failed"
                elif n_failed == 0:
                    cell_status = "ok"
                else:
                    cell_status = "partial"

                cell = {
                    "params":           params,
                    "status":           cell_status,
                    "n_seeds_ok":       n_ok,
                    "n_seeds_failed":   n_failed,
                    "per_seed_results": seed_results,
                }
                if n_ok > 0:
                    cell["overall"]  = _aggregate_overall(
                        [r["overall"] for r in ok_results]
                    )
                    cell["per_mode"] = _aggregate_per_mode(
                        [r["per_mode"] for r in ok_results]
                    )
                    states = [r["n_states"] for r in ok_results]
                    edges  = [r["n_edges"]  for r in ok_results]
                    cell["n_states_median"] = float(np.median(states))
                    cell["n_states_min"]    = int(np.min(states))
                    cell["n_states_max"]    = int(np.max(states))
                    cell["n_edges_median"]  = float(np.median(edges))
                    cell["n_edges_min"]     = int(np.min(edges))
                    cell["n_edges_max"]     = int(np.max(edges))

                entries[(cond, noise)]["methods"][method] = cell

                # Compact summary line
                if cell_status == "failed":
                    print(
                        f"    noise={noise:.3f}  {method:8s}  "
                        f"ALL {N_SEEDS} SEEDS FAILED"
                    )
                else:
                    f1 = cell["overall"]["f1"]
                    partial = (
                        f" [{n_failed}/{N_SEEDS} seeds failed]"
                        if n_failed > 0 else ""
                    )
                    print(
                        f"    noise={noise:.3f}  {method:8s}  "
                        f"F1 median={f1['median']:.3f} "
                        f"(range {f1['min']:.3f}–{f1['max']:.3f}){partial}"
                    )

            # ---- Incremental save after each (cond, method) block ----
            log["results"] = [
                entries[(c, n)]
                for c in TRAINING_CONDITIONS
                for n in TEST_NOISE_LEVELS
            ]
            with open(out_dir / "results.json", "w") as f:
                json.dump(log, f, indent=2)

        print()

    print(f"  Saved: {out_dir / 'results.json'}")

    # ---- End-of-run failure summary ----
    failed_cells = []
    for entry in log["results"]:
        for method, cell in entry["methods"].items():
            n_failed = cell.get("n_seeds_failed", 0)
            if n_failed == 0:
                continue
            err_types = {
                sr.get("error_type", "?")
                for sr in cell.get("per_seed_results", [])
                if sr.get("status") == "failed"
            }
            failed_cells.append((
                entry["training_condition"], entry["noise_std"], method,
                n_failed, cell.get("status"), err_types,
            ))

    if failed_cells:
        print("\nFailure breakdown:")
        for cond, noise, method, n_failed, status, errs in failed_cells:
            marker = "ALL" if status == "failed" else f"{n_failed}/{N_SEEDS}"
            print(
                f"  {cond:5s} noise={noise:.3f} {method:8s} | "
                f"{marker} seeds failed ({', '.join(sorted(errs))})"
            )

    print(f"\nRun complete. To generate plots:")
    print(f"  python exp_test_noise_plot.py")
    print(f"  python exp_test_noise_plot.py --log {out_dir / 'results.json'}")