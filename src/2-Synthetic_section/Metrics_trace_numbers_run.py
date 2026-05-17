"""
Metrics_scaling_run.py
======================
Experiment 5.x — Effect of training-corpus size on anomaly detection metrics.

For each N in N_TRAIN_LEVELS:
  for each method:
    for each repeat r in 0..REPEATS-1:
      train on traces[r*N : (r+1)*N] from a prefix-stable pool
      evaluate against fixed clean_test + fixed negatives
Aggregate F1 / precision / recall across repeats (median, min, max).

The training pool is generated once from CONFIG["clean"] (same generator the
synthetic experiments use), with deterministic seed. Pool size = REPEATS *
max(N_TRAIN_LEVELS), so all repeats see DISJOINT training sets.

Crash-resilient: each repeat in try/except, results.json saved incrementally
after every (method, N) cell. Cell status = ok / partial / failed.

Output (timestamped folder under Graphs/Metrics_scaling/):
  config.txt
  results.json
"""

import json
import subprocess
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path

sys.setrecursionlimit(50000)

import numpy as np

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from Generate_data import load_all_data, CONFIG
from Generators import generate_trace_set, NEG_MODE_NAMES
from Pipeline import run_pipeline


# =============================================================================
# CONFIG
# =============================================================================

# Training-corpus sizes to sweep.
N_TRAIN_LEVELS = [10, 25, 50, 75, 100, 140]

# Independent repeats per (method, N) cell. Each repeat uses a DISJOINT slice
# of the training pool, so they're genuinely independent samples.
REPEATS = 5

TAG_K = 4   # match exp_51_52 (final config)

# Single variant per method — keeps the sweep readable. Bump variants up
# later if you want a 2D method × N grid.
METHODS = [
    ("naive",   {"bins": 5}),
    ("sax",     {"w": 48, "bins": 5}),
    ("persist", {"bins": 6}),
]


# =============================================================================
# HELPERS
# =============================================================================

def _git_hash():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT
        ).decode().strip()
    except Exception:
        return "unknown"


def _save_config(out_dir, n_test_pos, n_neg, neg_modes, pool_size, base_seed):
    lines = [
        "=" * 60,
        "Run configuration — Training-corpus scaling sweep",
        "=" * 60,
        "",
        f"Timestamp     : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"Git hash      : {_git_hash()}",
        f"TAG k-future  : {TAG_K}",
        f"Repeats / N   : {REPEATS}",
        f"Base seed     : {base_seed}",
        f"Recursion lim : {sys.getrecursionlimit()}",
        "",
        f"--- Training-corpus sizes ({len(N_TRAIN_LEVELS)}) ---",
        f"  {N_TRAIN_LEVELS}",
        "",
        f"--- Training pool ---",
        f"  Total traces  : {pool_size}",
        f"  Slice scheme  : repeat r uses traces [r*N, (r+1)*N) — disjoint",
        f"  Generator     : CONFIG['clean'] profile, seed={base_seed}",
        "",
        "--- Methods ---",
        ]
    for method, params in METHODS:
        param_str = ", ".join(f"{k}={v}" for k, v in params.items())
        lines.append(f"  {method:8s}: {param_str}")

    lines += [
        "",
        "--- Test sets (fixed across sweep) ---",
        f"  test positives : {n_test_pos} traces (from clean_test on disk)",
        f"  negatives      : {n_neg} traces (from disk)",
        "",
        "--- Negative modes ---",
    ]
    mode_counts = Counter(neg_modes)
    for mode_int, count in sorted(mode_counts.items()):
        lines.append(f"  {NEG_MODE_NAMES[mode_int]:10s}: {count} traces")

    total = len(N_TRAIN_LEVELS) * len(METHODS) * REPEATS
    lines += [
        "",
        f"--- Total runs ---",
        f"  {len(N_TRAIN_LEVELS)} sizes × {len(METHODS)} methods × {REPEATS} repeats = {total}",
        "",
        "--- Output folder ---",
        f"  {out_dir}",
        "",
        "=" * 60,
        ]
    (out_dir / "config.txt").write_text("\n".join(lines))
    print(f"  Saved: {out_dir / 'config.txt'}")


# =============================================================================
# SINGLE REPEAT (with crash protection)
# =============================================================================

def _run_one_repeat(method, params, train_traces, test_pos, test_neg,
                    neg_modes, tag_k, repeat_idx):
    """Run one pipeline call. Returns ok+metrics or failed+error info."""
    try:
        result = run_pipeline(
            method=method,
            params=params,
            train_traces=train_traces,
            test_pos_traces=test_pos,
            test_neg_traces=test_neg,
            tag_k=tag_k,
            neg_modes=neg_modes,
        )
        ov = result["overall"]
        ov_clean = {k: v for k, v in ov.items()
                    if not isinstance(v, list)
                    and k not in ("save_path", "run_id")}
        return {
            "status":   "ok",
            "repeat":   repeat_idx,
            "overall":  ov_clean,
            "per_mode": result["per_mode"],
            "n_states": result["n_states"],
            "n_edges":  result["n_edges"],
        }
    except Exception as e:
        return {
            "status":     "failed",
            "repeat":     repeat_idx,
            "error_type": type(e).__name__,
            "error_msg":  str(e),
        }


# =============================================================================
# AGGREGATION
# =============================================================================

def _aggregate_overall(per_repeat_overall):
    if not per_repeat_overall:
        return {}
    metrics = per_repeat_overall[0].keys()
    out = {}
    for m in metrics:
        vals = [s[m] for s in per_repeat_overall
                if isinstance(s.get(m), (int, float))]
        if not vals:
            continue
        out[m] = {
            "median":     float(np.median(vals)),
            "min":        float(np.min(vals)),
            "max":        float(np.max(vals)),
            "mean":       float(np.mean(vals)),
            "std":        float(np.std(vals)),
            "per_repeat": vals,
        }
    return out


def _aggregate_per_mode(per_repeat_per_mode):
    if not per_repeat_per_mode:
        return {}
    mode_names = set()
    for pm in per_repeat_per_mode:
        mode_names.update(pm.keys())
    out = {}
    for mode in mode_names:
        rejections = [pm[mode]["rejection"] for pm in per_repeat_per_mode
                      if mode in pm]
        if rejections:
            out[mode] = {
                "rejection_median":     float(np.median(rejections)),
                "rejection_min":        float(np.min(rejections)),
                "rejection_max":        float(np.max(rejections)),
                "rejection_per_repeat": rejections,
            }
    return out


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    out_dir = ROOT / "Data" / "Graphs" / "Metrics_scaling" / timestamp
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output folder: {out_dir}")
    print(f"Recursion limit: {sys.getrecursionlimit()}\n")

    # Fixed test sets (from disk, same as exp_51_52 / exp_53).
    data = load_all_data()
    test_pos   = data["clean_test"]
    neg_traces = data["neg_traces"]
    neg_modes  = data["neg_modes"]

    # Prefix-stable training pool. Pool size = REPEATS * MAX_N so repeat r
    # uses traces [r*N, (r+1)*N), all disjoint across r.
    MAX_N      = max(N_TRAIN_LEVELS)
    POOL_SIZE  = REPEATS * MAX_N
    base_seed  = CONFIG["seed_clean"]
    print(f"Generating training pool: {POOL_SIZE} traces "
          f"(seed={base_seed}, clean profile)")
    training_pool = generate_trace_set(
        n_traces      = POOL_SIZE,
        seed          = base_seed,
        base_temp     = CONFIG["clean"]["base_temp"],
        amplitude     = CONFIG["clean"]["amplitude"],
        base_temp_std = CONFIG["clean"]["base_temp_std"],
        amplitude_std = CONFIG["clean"]["amplitude_std"],
        phase_std_h   = CONFIG["clean"]["phase_std_h"],
        noise_std     = CONFIG["clean"]["noise_std"],
    )

    _save_config(out_dir, len(test_pos), len(neg_traces), neg_modes,
                 POOL_SIZE, base_seed)

    total_runs = len(N_TRAIN_LEVELS) * len(METHODS) * REPEATS
    print(f"\nTotal pipeline runs: {total_runs}\n")

    log = {
        "timestamp":      timestamp,
        "git_hash":       _git_hash(),
        "tag_k":          TAG_K,
        "n_train_levels": N_TRAIN_LEVELS,
        "n_repeats":      REPEATS,
        "n_test":         len(test_pos),
        "n_neg":          len(neg_traces),
        "pool_size":      POOL_SIZE,
        "methods":        [{"method": m, "params": p} for m, p in METHODS],
        "results":        [],
    }

    run_counter = 0
    for n in N_TRAIN_LEVELS:
        print(f"--- N = {n}  ({N_TRAIN_LEVELS.index(n) + 1}/{len(N_TRAIN_LEVELS)}) ---")
        n_entry = {"n_train": n, "methods": {}}

        for method, params in METHODS:
            per_repeat_results = []
            per_repeat_overall_ok = []
            per_repeat_per_mode_ok = []
            per_repeat_n_states_ok = []
            per_repeat_n_edges_ok = []

            for r in range(REPEATS):
                run_counter += 1
                slice_start = r * n
                slice_end   = (r + 1) * n
                train_traces = training_pool[slice_start:slice_end]

                rr = _run_one_repeat(
                    method, params, train_traces, test_pos, neg_traces,
                    neg_modes, TAG_K, r,
                )
                per_repeat_results.append(rr)

                if rr["status"] == "ok":
                    per_repeat_overall_ok.append(rr["overall"])
                    per_repeat_per_mode_ok.append(rr["per_mode"])
                    per_repeat_n_states_ok.append(rr["n_states"])
                    per_repeat_n_edges_ok.append(rr["n_edges"])
                    ov = rr["overall"]
                    print(
                        f"  [{run_counter:3d}/{total_runs}] {method:8s} N={n} r={r}  "
                        f"P={ov['precision']:.3f} R={ov['recall']:.3f} F1={ov['f1']:.3f}"
                    )
                else:
                    print(
                        f"  [{run_counter:3d}/{total_runs}] {method:8s} N={n} r={r}  "
                        f"FAILED ({rr['error_type']}): {rr['error_msg'][:100]}"
                    )

            # Aggregate this cell
            n_ok     = len(per_repeat_overall_ok)
            n_failed = REPEATS - n_ok
            if n_ok == 0:
                cell_status = "failed"
            elif n_failed == 0:
                cell_status = "ok"
            else:
                cell_status = "partial"

            cell = {
                "params":             params,
                "status":             cell_status,
                "n_repeats_ok":       n_ok,
                "n_repeats_failed":   n_failed,
                "per_repeat_results": per_repeat_results,
            }
            if n_ok > 0:
                cell["overall"]  = _aggregate_overall(per_repeat_overall_ok)
                cell["per_mode"] = _aggregate_per_mode(per_repeat_per_mode_ok)
                cell["n_states_median"] = float(np.median(per_repeat_n_states_ok))
                cell["n_states_min"]    = int(np.min(per_repeat_n_states_ok))
                cell["n_states_max"]    = int(np.max(per_repeat_n_states_ok))
                cell["n_edges_median"]  = float(np.median(per_repeat_n_edges_ok))
                cell["n_edges_min"]     = int(np.min(per_repeat_n_edges_ok))
                cell["n_edges_max"]     = int(np.max(per_repeat_n_edges_ok))

            n_entry["methods"][method] = cell

            if cell_status == "failed":
                print(f"  {method:8s} ALL {REPEATS} REPEATS FAILED at N={n}")
            else:
                f1 = cell["overall"]["f1"]
                partial = f" [{n_failed}/{REPEATS} repeats failed]" if n_failed > 0 else ""
                print(f"  {method:8s} aggregated: F1 median={f1['median']:.3f} "
                      f"(range {f1['min']:.3f}-{f1['max']:.3f}){partial}")

        log["results"].append(n_entry)

        # Incremental save after every N level.
        with open(out_dir / "results.json", "w") as f:
            json.dump(log, f, indent=2)
        print()

    print(f"  Saved: {out_dir / 'results.json'}")

    # End-of-run failure summary
    n_total_failed = sum(
        cell.get("n_repeats_failed", 0)
        for entry in log["results"]
        for cell in entry["methods"].values()
    )
    if n_total_failed > 0:
        print(f"\n{n_total_failed} repeat(s) failed across the sweep.")

    print(f"\nRun complete. To generate plots:")
    print(f"  python Metrics_scaling_plot.py")
    print(f"  python Metrics_scaling_plot.py --log {out_dir / 'results.json'}")