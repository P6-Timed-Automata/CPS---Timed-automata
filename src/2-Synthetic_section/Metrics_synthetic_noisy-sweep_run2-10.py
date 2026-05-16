"""
exp_53_run.py
=============
Experiment 5.3 — Noise tolerance threshold (multi-seed, crash-resilient).

For each noise level in NOISE_LEVELS, for each method, generates training
data N_SEEDS times with different random seeds, runs the pipeline on each,
and records F1 / precision / recall per seed.

Crash-resilient design (for SLURM):
  - Recursion limit raised at startup.
  - Each (noise, method, seed) pipeline run is wrapped in try/except.
    Failures (RecursionError, MemoryError, anything else) are logged as
    status="failed" and the sweep continues with the next seed/method/level.
  - JSON saved after every noise level so SLURM SIGKILL preserves earlier
    levels.
  - A (noise, method) cell is "ok" if at least one seed succeeded,
    "partial" if some seeds failed, "failed" if all seeds failed.

The aggregation skips failed seeds. Median/range are computed from
successful seeds only; if all seeds fail, the cell has status="failed"
and no metrics.

Writes results.json. The companion plotter (exp_53_plot.py) reads that file.

Requires: run generate_all_data.py first.

Output (timestamped folder under Graphs/Metrics_noisy_sweep/):
  config.txt
  results.json
"""

import json
import subprocess
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path

# Raise recursion limit before TAG-related imports.
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

N_TRAIN = CONFIG["n_train"]

NOISE_LEVELS = [0.02, 0.05, 0.1, 0.15, 0.2, 0.25,0.3,0.4,0.5]

# Number of seeds per (method, noise_level). Multiplies runtime by this factor.
N_SEEDS = 3

F1_THRESHOLD = 0.7

TAG_K = 2

METHODS = [
    ("naive",   {"bins": 10}),
    ("sax144",     {"w": 144, "bins": 10}),   # was w=288 (degenerate); using 144
    ("sax48",     {"w": 48, "bins": 10}),
   ("sax24",     {"w": 24, "bins": 10}),
    ("persist", {"bins": 11}),
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


def _save_config(out_dir, data, base_seed):
    lines = [
        "=" * 55,
        "Run configuration -- Noise Sweep (multi-seed, crash-resilient)",
        "=" * 55,
        "",
        f"Timestamp      : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"Git hash       : {_git_hash()}",
        f"TAG k-future   : {TAG_K}",
        f"F1 threshold   : {F1_THRESHOLD}",
        f"Seeds per pt   : {N_SEEDS}",
        f"Base seed      : {base_seed}  (per-seed offsets are 0..{N_SEEDS-1})",
        f"Recursion lim  : {sys.getrecursionlimit()}",
        "",
        "--- Noise levels swept ---",
        f"  {NOISE_LEVELS}",
        f"  Clean baseline: {CONFIG['clean']['noise_std']}",
        f"  Noisy baseline: {CONFIG['noisy']['noise_std']}",
        "",
        "--- Base training parameters (noise_std and seed overridden) ---",
        f"  base_temp     : {CONFIG['clean']['base_temp']}",
        f"  amplitude     : {CONFIG['clean']['amplitude']}",
        f"  base_temp_std : {CONFIG['clean']['base_temp_std']}",
        f"  amplitude_std : {CONFIG['clean']['amplitude_std']}",
        f"  phase_std_h   : {CONFIG['clean']['phase_std_h']}",
        "",
        "--- Methods ---",
        ]
    for method, params in METHODS:
        param_str = ", ".join(f"{k}={v}" for k, v in params.items())
        lines.append(f"  {method:8s}: {param_str}")

    lines += [
        "",
        "--- Dataset sizes ---",
        f"  train per (level, seed) : {N_TRAIN} traces",
        f"  test positives          : {len(data['clean_test'])} traces (fixed)",
        f"  negatives               : {len(data['neg_traces'])} traces (fixed)",
        "",
        "--- Negative modes ---",
    ]
    mode_counts = Counter(data["neg_modes"])
    for mode_int, count in sorted(mode_counts.items()):
        lines.append(f"  {NEG_MODE_NAMES[mode_int]:10s}: {count} traces")

    lines += [
        "",
        "--- Total runs ---",
        f"  {len(NOISE_LEVELS)} noise levels x {len(METHODS)} methods x "
        f"{N_SEEDS} seeds = {len(NOISE_LEVELS) * len(METHODS) * N_SEEDS}",
        "",
        "--- Output folder ---",
        f"  {out_dir}",
        "",
        "=" * 55,
        ]

    config_path = out_dir / "config.txt"
    config_path.write_text("\n".join(lines))
    print(f"  Saved: {config_path}")


# =============================================================================
# SINGLE SEED RUN (with crash protection)
# =============================================================================

def _run_one_seed(method, params, train_traces, test_pos, test_neg,
                  neg_modes, tag_k, seed):
    """
    Run one pipeline call. Returns a dict with status='ok' + metrics, or
    status='failed' + error info. Never raises.
    """
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
            "status":    "ok",
            "seed":      seed,
            "overall":   ov_clean,
            "per_mode":  result["per_mode"],
            "n_states":  result["n_states"],
            "n_edges":   result["n_edges"],
        }
    except RecursionError as e:
        return {
            "status":     "failed",
            "seed":       seed,
            "error_type": "RecursionError",
            "error_msg":  str(e),
        }
    except MemoryError as e:
        return {
            "status":     "failed",
            "seed":       seed,
            "error_type": "MemoryError",
            "error_msg":  str(e),
        }
    except Exception as e:
        return {
            "status":     "failed",
            "seed":       seed,
            "error_type": type(e).__name__,
            "error_msg":  str(e),
        }


# =============================================================================
# AGGREGATION (skips failed seeds)
# =============================================================================

def _aggregate_overall(per_seed_overall):
    """
    Aggregate a list of overall-metric dicts across successful seeds.
    Returns a dict with median/min/max/mean/std for each metric.
    Returns {} if no successful seeds.
    """
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
    """Aggregate per-mode metrics across successful seeds."""
    if not per_seed_per_mode:
        return {}
    mode_names = set()
    for pm in per_seed_per_mode:
        mode_names.update(pm.keys())

    out = {}
    for mode in mode_names:
        rejections = []
        for pm in per_seed_per_mode:
            if mode in pm:
                rejections.append(pm[mode]["rejection"])
        if rejections:
            out[mode] = {
                "rejection_median":   float(np.median(rejections)),
                "rejection_min":      float(np.min(rejections)),
                "rejection_max":      float(np.max(rejections)),
                "rejection_per_seed": rejections,
            }
    return out


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    out_dir = ROOT / "Data" / "Graphs" / "Metrics_noisy_sweep" / timestamp
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output folder: {out_dir}")
    print(f"Python recursion limit raised to {sys.getrecursionlimit()}\n")

    data = load_all_data()
    test_pos = data["clean_test"]
    neg_traces = data["neg_traces"]
    neg_modes = data["neg_modes"]

    base_seed = CONFIG["seed_clean"]

    print("=== Config ===")
    _save_config(out_dir, data, base_seed)

    total_runs = len(NOISE_LEVELS) * len(METHODS) * N_SEEDS
    print(f"\nTotal pipeline runs: {total_runs}\n")

    log = {
        "timestamp":    timestamp,
        "git_hash":     _git_hash(),
        "tag_k":        TAG_K,
        "n_train":      N_TRAIN,
        "n_seeds":      N_SEEDS,
        "n_test":       len(test_pos),
        "n_neg":        len(neg_traces),
        "noise_levels": NOISE_LEVELS,
        "f1_threshold": F1_THRESHOLD,
        "clean_noise":  CONFIG["clean"]["noise_std"],
        "noisy_noise":  CONFIG["noisy"]["noise_std"],
        "methods":      [{"method": m, "params": p} for m, p in METHODS],
        "results":      [],
    }

    run_counter = 0
    for noise_idx, noise in enumerate(NOISE_LEVELS):
        print(f"--- noise_std = {noise:.3f}  ({noise_idx + 1}/{len(NOISE_LEVELS)}) ---")
        noise_entry = {"noise_std": noise, "methods": {}}

        for method, params in METHODS:
            per_seed_results = []   # ALL seeds, both ok and failed
            per_seed_overall_ok = []
            per_seed_per_mode_ok = []
            per_seed_n_states_ok = []
            per_seed_n_edges_ok = []

            for seed_offset in range(N_SEEDS):
                run_counter += 1
                seed = base_seed + seed_offset

                # Regenerate training data with this seed and noise level
                train_traces = generate_trace_set(
                    n_traces=N_TRAIN,
                    seed=seed,
                    base_temp=CONFIG["clean"]["base_temp"],
                    amplitude=CONFIG["clean"]["amplitude"],
                    base_temp_std=CONFIG["clean"]["base_temp_std"],
                    amplitude_std=CONFIG["clean"]["amplitude_std"],
                    phase_std_h=CONFIG["clean"]["phase_std_h"],
                    noise_std=noise,
                )

                r = _run_one_seed(method, params, train_traces,
                                  test_pos, neg_traces, neg_modes,
                                  TAG_K, seed)
                per_seed_results.append(r)

                if r["status"] == "ok":
                    per_seed_overall_ok.append(r["overall"])
                    per_seed_per_mode_ok.append(r["per_mode"])
                    per_seed_n_states_ok.append(r["n_states"])
                    per_seed_n_edges_ok.append(r["n_edges"])

                    ov = r["overall"]
                    print(
                        f"  [{run_counter:3d}/{total_runs}] {method:8s} seed={seed} "
                        f"P={ov['precision']:.3f} R={ov['recall']:.3f} F1={ov['f1']:.3f}"
                    )
                else:
                    print(
                        f"  [{run_counter:3d}/{total_runs}] {method:8s} seed={seed} "
                        f"FAILED ({r['error_type']}): {r['error_msg'][:100]}"
                    )

            # Aggregate
            agg_overall = _aggregate_overall(per_seed_overall_ok)
            agg_per_mode = _aggregate_per_mode(per_seed_per_mode_ok)
            n_ok = len(per_seed_overall_ok)
            n_failed = N_SEEDS - n_ok

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
                "per_seed_results": per_seed_results,
            }

            # Only include aggregated stats if there's at least one success
            if n_ok > 0:
                cell.update({
                    "overall":          agg_overall,
                    "per_mode":         agg_per_mode,
                    "n_states_median":  float(np.median(per_seed_n_states_ok)),
                    "n_states_min":     int(np.min(per_seed_n_states_ok)),
                    "n_states_max":     int(np.max(per_seed_n_states_ok)),
                    "n_edges_median":   float(np.median(per_seed_n_edges_ok)),
                    "n_edges_min":      int(np.min(per_seed_n_edges_ok)),
                    "n_edges_max":      int(np.max(per_seed_n_edges_ok)),
                })

            noise_entry["methods"][method] = cell

            # Per-(noise, method) summary
            if cell_status == "failed":
                print(f"  {method:8s} ALL {N_SEEDS} SEEDS FAILED at noise={noise:.3f}")
            else:
                f1_med = agg_overall["f1"]["median"]
                f1_lo = agg_overall["f1"]["min"]
                f1_hi = agg_overall["f1"]["max"]
                partial = f" [{n_failed}/{N_SEEDS} seeds failed]" if n_failed > 0 else ""
                print(f"  {method:8s} aggregated: F1 median={f1_med:.3f} "
                      f"(range {f1_lo:.3f}-{f1_hi:.3f}){partial}")

        log["results"].append(noise_entry)

        # Incremental save after every noise level
        with open(out_dir / "results.json", "w") as f:
            json.dump(log, f, indent=2)
        print()

    print(f"  Saved: {out_dir / 'results.json'}")

    # End-of-run failure summary
    n_total_failed_seeds = 0
    n_failed_cells = 0
    for noise_entry in log["results"]:
        for method, cell in noise_entry["methods"].items():
            n_total_failed_seeds += cell.get("n_seeds_failed", 0)
            if cell.get("status") == "failed":
                n_failed_cells += 1

    if n_total_failed_seeds > 0:
        print(f"\n{n_total_failed_seeds} seed(s) failed across "
              f"{n_failed_cells} fully-failed cell(s).")
        print("Failure breakdown:")
        for noise_entry in log["results"]:
            noise = noise_entry["noise_std"]
            for method, cell in noise_entry["methods"].items():
                n_failed = cell.get("n_seeds_failed", 0)
                if n_failed == 0:
                    continue
                # Collect distinct error types
                err_types = set()
                for sr in cell.get("per_seed_results", []):
                    if sr.get("status") == "failed":
                        err_types.add(sr.get("error_type", "?"))
                marker = "ALL" if cell.get("status") == "failed" else f"{n_failed}/{N_SEEDS}"
                print(f"  noise={noise:.3f} {method:8s} | "
                      f"{marker} seeds failed ({', '.join(sorted(err_types))})")

    print(f"\nRun complete. To generate plots:")
    print(f"  python exp_53_plot.py")
    print(f"or:")
    print(f"  python exp_53_plot.py --log {out_dir / 'results.json'}")