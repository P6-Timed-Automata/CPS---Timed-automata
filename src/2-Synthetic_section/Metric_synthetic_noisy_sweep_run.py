"""
exp_53_run.py
=============
Experiment 5.3 — Noise tolerance threshold (multi-seed).

For each noise level in NOISE_LEVELS, for each method, generates training
data N_SEEDS times with different random seeds, runs the pipeline on each,
and records F1 / precision / recall per seed.

This addresses the "single-measurement-per-point" weakness of the original
sweep — with multiple seeds we can report median + range and the
threshold-crossing point becomes more reliable.

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

NOISE_LEVELS = [0.02, 0.05, 0.1, 0.15, 0.2, 0.25]

# Number of seeds per (method, noise_level). Multiplies runtime by this factor.
# 3 is a reasonable thesis-grade default. 5+ is more rigorous but slower.
N_SEEDS = 3

F1_THRESHOLD = 0.7

TAG_K = 4

METHODS = [
    ("naive",   {"bins": 5}),
    ("sax",     {"w": 288, "bins": 5}),
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


def _save_config(out_dir, data, base_seed):
    lines = [
        "=" * 55,
        "Run configuration -- Noise Sweep (multi-seed)",
        "=" * 55,
        "",
        f"Timestamp      : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"Git hash       : {_git_hash()}",
        f"TAG k-future   : {TAG_K}",
        f"F1 threshold   : {F1_THRESHOLD}",
        f"Seeds per pt   : {N_SEEDS}",
        f"Base seed      : {base_seed}  (per-seed offsets are 0..{N_SEEDS-1})",
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
# AGGREGATION
# =============================================================================

def _aggregate_overall(per_seed_overalls):
    """
    Aggregate a list of overall-metric dicts across seeds.
    Returns a dict with median/min/max/mean/std for each metric.
    """
    if not per_seed_overalls:
        return {}
    metrics = per_seed_overalls[0].keys()
    out = {}
    for m in metrics:
        vals = [s[m] for s in per_seed_overalls if isinstance(s.get(m), (int, float))]
        if not vals:
            continue
        out[m] = {
            "median":     float(np.median(vals)),
            "min":        float(np.min(vals)),
            "max":        float(np.max(vals)),
            "mean":       float(np.mean(vals)),
            "std":        float(np.std(vals)),
            "per_seed":   vals,
        }
    return out


def _aggregate_per_mode(per_seed_per_modes):
    """Aggregate per-mode metrics across seeds."""
    if not per_seed_per_modes:
        return {}
    mode_names = set()
    for pm in per_seed_per_modes:
        mode_names.update(pm.keys())

    out = {}
    for mode in mode_names:
        rejections = []
        for pm in per_seed_per_modes:
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
    print(f"Output folder: {out_dir}\n")

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
        # results[noise_level_idx][method_name] = aggregate_dict
        "results":      [],
    }

    run_counter = 0
    for noise_idx, noise in enumerate(NOISE_LEVELS):
        print(f"--- noise_std = {noise:.3f}  ({noise_idx + 1}/{len(NOISE_LEVELS)}) ---")
        noise_entry = {"noise_std": noise, "methods": {}}

        for method, params in METHODS:
            per_seed_overall = []
            per_seed_per_mode = []
            per_seed_n_states = []
            per_seed_n_edges = []
            per_seed_seeds = []

            for seed_offset in range(N_SEEDS):
                run_counter += 1
                seed = base_seed + seed_offset

                # Regenerate training data with this seed and this noise level
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

                result = run_pipeline(
                    method=method,
                    params=params,
                    train_traces=train_traces,
                    test_pos_traces=test_pos,
                    test_neg_traces=neg_traces,
                    tag_k=TAG_K,
                    neg_modes=neg_modes,
                )
                ov = result["overall"]
                ov_clean = {k: v for k, v in ov.items()
                            if not isinstance(v, list) and k not in ("save_path", "run_id")}

                per_seed_overall.append(ov_clean)
                per_seed_per_mode.append(result["per_mode"])
                per_seed_n_states.append(result["n_states"])
                per_seed_n_edges.append(result["n_edges"])
                per_seed_seeds.append(seed)

                print(
                    f"  [{run_counter:3d}/{total_runs}] {method:8s} seed={seed} "
                    f"P={ov['precision']:.3f} R={ov['recall']:.3f} F1={ov['f1']:.3f}"
                )

            # Aggregate across seeds for this (noise, method) cell
            agg_overall = _aggregate_overall(per_seed_overall)
            agg_per_mode = _aggregate_per_mode(per_seed_per_mode)

            noise_entry["methods"][method] = {
                "params":           params,
                "overall":          agg_overall,
                "per_mode":         agg_per_mode,
                "n_states_median":  float(np.median(per_seed_n_states)),
                "n_states_min":     int(np.min(per_seed_n_states)),
                "n_states_max":     int(np.max(per_seed_n_states)),
                "n_edges_median":   float(np.median(per_seed_n_edges)),
                "n_edges_min":      int(np.min(per_seed_n_edges)),
                "n_edges_max":      int(np.max(per_seed_n_edges)),
                "seeds":            per_seed_seeds,
                "per_seed_overall": per_seed_overall,
                "per_seed_per_mode": per_seed_per_mode,
            }

            # Brief per-(noise, method) summary
            f1_med = agg_overall["f1"]["median"]
            f1_lo = agg_overall["f1"]["min"]
            f1_hi = agg_overall["f1"]["max"]
            print(f"  {method:8s} aggregated: F1 median={f1_med:.3f} (range {f1_lo:.3f}-{f1_hi:.3f})")

        log["results"].append(noise_entry)

        # Save incrementally
        with open(out_dir / "results.json", "w") as f:
            json.dump(log, f, indent=2)
        print()

    print(f"  Saved: {out_dir / 'results.json'}")
    print(f"\nRun complete. To generate plots:")
    print(f"  python exp_53_plot.py")
    print(f"or:")
    print(f"  python exp_53_plot.py --log {out_dir / 'results.json'}")