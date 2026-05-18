"""
Metrics_bins_run.py
===================
Experiment 5.x — Effect of bin count (alphabet size) on anomaly detection.

For each effective_bins in BIN_LEVELS:
  for each method:
    for each seed in 0..N_SEEDS-1:
      regenerate training data with this seed (CONFIG["clean"] profile)
      discretize with method + bins
      train TA, evaluate against fixed clean_test + fixed negatives
Aggregate F1 / precision / recall across seeds (median, min, max).

Bin-count convention:
  BIN_LEVELS is the *effective* number of symbols / intervals (X-axis label).
  Per the project's Persist offset rule, Persist receives bins+1.
  Naive and SAX receive bins as-is.

Crash-resilient: each seed in try/except, results.json saved incrementally
after every (bins, method) cell.

Output (timestamped folder under Graphs/Metrics_bins/):
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

# Effective bin counts to sweep. Naive/SAX use these directly; Persist gets +1
# (so the X-axis represents the same alphabet size across all three methods).
BIN_LEVELS = [3, 5, 7, 10, 12, 15]

# Independent seeds per (method, bins) cell. Each seed regenerates training
# data, giving an estimate of training-data variance.
N_SEEDS = 5

TAG_K   = 4     # match exp_51_52
SAX_W   = 48    # held fixed; we're sweeping bins, not w

METHOD_NAMES = ["naive", "sax", "persist"]


def make_params(method, effective_bins):
    """Translate effective bin count into the method's params dict."""
    if method == "naive":
        return {"bins": effective_bins}
    elif method == "sax":
        return {"w": SAX_W, "bins": effective_bins}
    elif method == "persist":
        return {"bins": effective_bins + 1}     # Persist offset convention
    else:
        raise ValueError(f"Unknown method: {method}")


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


def _save_config(out_dir, n_train, n_test_pos, n_neg, neg_modes, base_seed):
    lines = [
        "=" * 60,
        "Run configuration — Bin-count sweep",
        "=" * 60,
        "",
        f"Timestamp     : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"Git hash      : {_git_hash()}",
        f"TAG k-future  : {TAG_K}",
        f"Seeds / cell  : {N_SEEDS}",
        f"Base seed     : {base_seed}",
        f"Recursion lim : {sys.getrecursionlimit()}",
        "",
        f"--- Effective bin counts ({len(BIN_LEVELS)}) ---",
        f"  {BIN_LEVELS}",
        f"  (Persist receives bins+1 internally; SAX uses fixed w={SAX_W})",
        "",
        "--- Methods ---",
        ]
    for method in METHOD_NAMES:
        lines.append(f"  {method:8s}: params built from (method, effective_bins)")

    lines += [
        "",
        "--- Training data (regenerated per seed) ---",
        f"  n_train : {n_train} traces (CONFIG['clean'] profile)",
        "",
        "--- Test sets (fixed across sweep) ---",
        f"  test positives : {n_test_pos} traces (clean_test on disk)",
        f"  negatives      : {n_neg} traces (from disk)",
        "",
        "--- Negative modes ---",
    ]
    mode_counts = Counter(neg_modes)
    for mode_int, count in sorted(mode_counts.items()):
        lines.append(f"  {NEG_MODE_NAMES[mode_int]:10s}: {count} traces")

    total = len(BIN_LEVELS) * len(METHOD_NAMES) * N_SEEDS
    lines += [
        "",
        f"--- Total runs ---",
        f"  {len(BIN_LEVELS)} bins × {len(METHOD_NAMES)} methods × {N_SEEDS} seeds = {total}",
        "",
        "--- Output folder ---",
        f"  {out_dir}",
        "",
        "=" * 60,
        ]
    (out_dir / "config.txt").write_text("\n".join(lines))
    print(f"  Saved: {out_dir / 'config.txt'}")


# =============================================================================
# SINGLE SEED RUN (crash-protected)
# =============================================================================

def _run_one_seed(method, params, train_traces, test_pos, test_neg,
                  neg_modes, tag_k, seed):
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
            "seed":     seed,
            "overall":  ov_clean,
            "per_mode": result["per_mode"],
            "n_states": result["n_states"],
            "n_edges":  result["n_edges"],
        }
    except Exception as e:
        return {
            "status":     "failed",
            "seed":       seed,
            "error_type": type(e).__name__,
            "error_msg":  str(e),
        }


# =============================================================================
# AGGREGATION
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
# MAIN
# =============================================================================

if __name__ == "__main__":
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    out_dir = ROOT / "Data" / "Graphs" / "Metrics_bins" / timestamp
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output folder: {out_dir}")
    print(f"Recursion limit: {sys.getrecursionlimit()}\n")

    data       = load_all_data()
    test_pos   = data["clean_test"]
    neg_traces = data["neg_traces"]
    neg_modes  = data["neg_modes"]
    base_seed  = CONFIG["seed_clean"]
    n_train    = CONFIG["n_train"]

    _save_config(out_dir, n_train, len(test_pos), len(neg_traces),
                 neg_modes, base_seed)

    total_runs = len(BIN_LEVELS) * len(METHOD_NAMES) * N_SEEDS
    print(f"\nTotal pipeline runs: {total_runs}\n")

    log = {
        "timestamp":  timestamp,
        "git_hash":   _git_hash(),
        "tag_k":      TAG_K,
        "sax_w":      SAX_W,
        "bin_levels": BIN_LEVELS,
        "n_seeds":    N_SEEDS,
        "n_train":    n_train,
        "n_test":     len(test_pos),
        "n_neg":      len(neg_traces),
        "methods":    [{"method": m} for m in METHOD_NAMES],
        "results":    [],
    }

    run_counter = 0
    for bins in BIN_LEVELS:
        print(f"--- bins = {bins}  "
              f"({BIN_LEVELS.index(bins) + 1}/{len(BIN_LEVELS)}) ---")
        bin_entry = {"bins": bins, "methods": {}}

        for method in METHOD_NAMES:
            params = make_params(method, bins)

            per_seed_results = []
            per_seed_overall_ok = []
            per_seed_per_mode_ok = []
            per_seed_n_states_ok = []
            per_seed_n_edges_ok = []

            for seed_offset in range(N_SEEDS):
                run_counter += 1
                seed = base_seed + seed_offset

                # Regenerate training data per seed (CONFIG['clean'] profile).
                train_traces = generate_trace_set(
                    n_traces      = n_train,
                    seed          = seed,
                    base_temp     = CONFIG["clean"]["base_temp"],
                    amplitude     = CONFIG["clean"]["amplitude"],
                    base_temp_std = CONFIG["clean"]["base_temp_std"],
                    amplitude_std = CONFIG["clean"]["amplitude_std"],
                    phase_std_h   = CONFIG["clean"]["phase_std_h"],
                    noise_std     = CONFIG["clean"]["noise_std"],
                )

                r = _run_one_seed(method, params, train_traces, test_pos,
                                  neg_traces, neg_modes, TAG_K, seed)
                per_seed_results.append(r)

                if r["status"] == "ok":
                    per_seed_overall_ok.append(r["overall"])
                    per_seed_per_mode_ok.append(r["per_mode"])
                    per_seed_n_states_ok.append(r["n_states"])
                    per_seed_n_edges_ok.append(r["n_edges"])
                    ov = r["overall"]
                    print(
                        f"  [{run_counter:3d}/{total_runs}] {method:8s} "
                        f"bins={bins} seed={seed}  "
                        f"P={ov['precision']:.3f} R={ov['recall']:.3f} F1={ov['f1']:.3f}"
                    )
                else:
                    print(
                        f"  [{run_counter:3d}/{total_runs}] {method:8s} "
                        f"bins={bins} seed={seed}  "
                        f"FAILED ({r['error_type']}): {r['error_msg'][:100]}"
                    )

            # Aggregate this cell
            n_ok     = len(per_seed_overall_ok)
            n_failed = N_SEEDS - n_ok
            if n_ok == 0:
                cell_status = "failed"
            elif n_failed == 0:
                cell_status = "ok"
            else:
                cell_status = "partial"

            cell = {
                "params":           params,
                "effective_bins":   bins,        # X-axis value for the plotter
                "status":           cell_status,
                "n_seeds_ok":       n_ok,
                "n_seeds_failed":   n_failed,
                "per_seed_results": per_seed_results,
            }
            if n_ok > 0:
                cell["overall"]  = _aggregate_overall(per_seed_overall_ok)
                cell["per_mode"] = _aggregate_per_mode(per_seed_per_mode_ok)
                cell["n_states_median"] = float(np.median(per_seed_n_states_ok))
                cell["n_states_min"]    = int(np.min(per_seed_n_states_ok))
                cell["n_states_max"]    = int(np.max(per_seed_n_states_ok))
                cell["n_edges_median"]  = float(np.median(per_seed_n_edges_ok))
                cell["n_edges_min"]     = int(np.min(per_seed_n_edges_ok))
                cell["n_edges_max"]     = int(np.max(per_seed_n_edges_ok))

            bin_entry["methods"][method] = cell

            if cell_status == "failed":
                print(f"  {method:8s} ALL {N_SEEDS} SEEDS FAILED at bins={bins}")
            else:
                f1 = cell["overall"]["f1"]
                partial = f" [{n_failed}/{N_SEEDS} seeds failed]" if n_failed > 0 else ""
                print(f"  {method:8s} aggregated: F1 median={f1['median']:.3f} "
                      f"(range {f1['min']:.3f}-{f1['max']:.3f}){partial}")

        log["results"].append(bin_entry)

        # Incremental save after every bins level.
        with open(out_dir / "results.json", "w") as f:
            json.dump(log, f, indent=2)
        print()

    print(f"  Saved: {out_dir / 'results.json'}")

    # End-of-run failure summary
    n_total_failed = sum(
        cell.get("n_seeds_failed", 0)
        for entry in log["results"]
        for cell in entry["methods"].values()
    )
    if n_total_failed > 0:
        print(f"\n{n_total_failed} seed(s) failed across the sweep.")

    print(f"\nRun complete. To generate plots:")
    print(f"  python Metrics_bins_plot.py")
    print(f"  python Metrics_bins_plot.py --log {out_dir / 'results.json'}")