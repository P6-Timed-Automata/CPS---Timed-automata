"""
exp_55_ecg.py / Metrics_ecg_run.py
==================================
Experiment 5.5 — ECG data.

Same pipeline as exp 5.4 (real temperature) ported to ECG traces.
ECG signals are highly regular, so this should be your strongest and
cleanest end-to-end pipeline demonstration.

Before running:
  1. Set DATA_FOLDER to your ECG positive-trace CSV directory.
  2. ECG CSVs must be semicolon-delimited, columns: time;value
     (matches Generators.load_traces).
  3. Set NEGATIVE_MODE to "folder" (load real anomalies) or "synthetic".
     Note: synthetic negatives use generate_negative_set, which produces
     TEMPERATURE-shaped negatives — known domain mismatch for ECG. Prefer
     "folder" with real labeled anomalies (e.g. MIT-BIH) when possible.
  4. Tune SAX w for ECG cycle length (≪ 24h temperature cycles).

Companion plotter:
  python Metrics_ecg_plot.py

Output (timestamped folder under Graphs/exp_55/):
  config.txt
  results.json
"""

import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path

# Raise recursion limit before TAG-related imports.
sys.setrecursionlimit(50000)

import numpy as np

# Project root is three levels up.
ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from Generators import (
    load_traces, load_traces_by_mode, NEG_MODE_NAMES, generate_negative_set,
)
from Pipeline import run_pipeline

# =============================================================================
# CONFIG
# =============================================================================

TAG_K = 2

# ---- Data paths -------------------------------------------------------------
# Mirrors the temperature runner: separate train and test folders on disk,
# negatives organised into mode subfolders (spikes / shifted / stuck / offset).
TRAIN_FOLDER    = ROOT / "Data" / "3-ExtractInterval" / "ecg" / "1beat-experiment1" / "1beat-train"
TEST_POS_FOLDER = ROOT / "Data" / "3-ExtractInterval" / "ecg" / "1beat-experiment1" / "1beat-test" / "positive"
NEG_DATA_FOLDER = ROOT / "Data" / "3-ExtractInterval" / "ecg" / "1beat-experiment1" / "1beat-test" / "negative"

TRAIN_SPLIT = 0.8   # only used if TEST_POS_FOLDER doesn't exist (fallback)

# ---- Negative mode ----------------------------------------------------------
# "folder"    : load real labeled anomalies from NEG_DATA_FOLDER
# "synthetic" : generate_negative_set (4 temp-shape modes; domain mismatch)
NEGATIVE_MODE   = "folder"
N_SYNTHETIC_NEG = 60

# ---- Methods ----------------------------------------------------------------
# SAX w should be roughly n_samples_per_cycle for ECG (much smaller than the
# 24h-temperature default of 144). E.g. 360 Hz × 1 beat ≈ 360 samples/beat,
# trace = 3 beats → w around 50–100 is a sane starting point.
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


def _clean_trace(trace):
    """Drop samples where either time or value is NaN."""
    times = np.asarray(trace[0], dtype=float)
    vals  = np.asarray(trace[1], dtype=float)
    mask  = ~(np.isnan(times) | np.isnan(vals))
    return (times[mask], vals[mask])


def _shuffle_traces(traces, seed=0):
    """Deterministic shuffle for reproducible splits."""
    rng = np.random.default_rng(seed)
    indices = list(range(len(traces)))
    rng.shuffle(indices)
    return [traces[i] for i in indices]


# =============================================================================
# DATA LOADING
# =============================================================================

def load_ecg_data():
    """
    Load training and test ECG traces from separate folders, mirroring the
    temperature runner. Falls back to splitting TRAIN_FOLDER 80/20 if
    TEST_POS_FOLDER doesn't exist. Negatives are loaded by mode subfolder so
    the per-mode rejection metrics in results.json are populated correctly.
    """
    print("Loading ECG traces...")

    if not TRAIN_FOLDER.exists():
        raise FileNotFoundError(
            f"TRAIN_FOLDER not found: {TRAIN_FOLDER}\n"
            f"Run split_dataset on your ECG positives first."
        )

    # --- Positives ----------------------------------------------------------
    if TEST_POS_FOLDER.exists():
        train_traces = load_traces(TRAIN_FOLDER)
        test_pos     = load_traces(TEST_POS_FOLDER)
        train_traces = [_clean_trace(t) for t in train_traces]
        test_pos     = [_clean_trace(t) for t in test_pos]
        train_traces = [t for t in train_traces if len(t[0]) > 0]
        test_pos     = [t for t in test_pos     if len(t[0]) > 0]
        print(f"  TRAIN_FOLDER    : {len(train_traces)} traces")
        print(f"  TEST_POS_FOLDER : {len(test_pos)} traces")
    else:
        print(f"  TEST_POS_FOLDER missing; splitting {TRAIN_FOLDER} "
              f"{TRAIN_SPLIT * 100:.0f}/{(1 - TRAIN_SPLIT) * 100:.0f}")
        all_traces = load_traces(TRAIN_FOLDER)
        all_traces = [_clean_trace(t) for t in all_traces]
        all_traces = [t for t in all_traces if len(t[0]) > 0]
        all_traces = _shuffle_traces(all_traces, seed=0)
        split = int(len(all_traces) * TRAIN_SPLIT)
        train_traces = all_traces[:split]
        test_pos     = all_traces[split:]
        print(f"  Train : {len(train_traces)} traces")
        print(f"  Test+ : {len(test_pos)} traces")

    if not train_traces or not test_pos:
        raise SystemExit("ERROR: empty train or test set after cleaning.")

    # --- Negatives (per-mode subfolders, like the synthetic exp 5.1/5.2) ----
    use_folder = (
            NEGATIVE_MODE == "folder"
            and NEG_DATA_FOLDER is not None
            and NEG_DATA_FOLDER.exists()
    )

    if use_folder:
        neg_traces, neg_modes = load_traces_by_mode(NEG_DATA_FOLDER)
        # Clean each trace, drop empties, keep modes aligned.
        cleaned = [(_clean_trace(t), m) for t, m in zip(neg_traces, neg_modes)]
        cleaned = [(t, m) for t, m in cleaned if len(t[0]) > 0]
        neg_traces     = [t for t, _ in cleaned]
        neg_modes      = [m for _, m in cleaned]
        neg_mode_names = list(NEG_MODE_NAMES.values())
        print(f"  Negatives: {len(neg_traces)} loaded from {NEG_DATA_FOLDER} "
              f"(by mode subfolder)")
    else:
        if NEGATIVE_MODE == "folder":
            print(f"  Warning: NEG_DATA_FOLDER missing ({NEG_DATA_FOLDER}); "
                  f"falling back to synthetic.")
        print("  Generating synthetic negatives (4 modes — temperature-shaped)...")
        neg_traces, neg_modes = generate_negative_set(
            n_traces=N_SYNTHETIC_NEG, seed=99,
        )
        neg_mode_names = list(NEG_MODE_NAMES.values())
        print(f"  Negatives: {len(neg_traces)}")

    return train_traces, test_pos, neg_traces, neg_modes, neg_mode_names

# =============================================================================
# PIPELINE RUN WITH CRASH PROTECTION
# =============================================================================

def _run_one_method(method, params, train_traces, test_pos, neg_traces,
                    neg_modes, tag_k, out_dir):
    """Run one pipeline call; return result dict with status=ok|failed."""
    ta_folder = str(out_dir / "ta_images" / method)
    try:
        result = run_pipeline(
            method=method,
            params=params,
            train_traces=train_traces,
            test_pos_traces=test_pos,
            test_neg_traces=neg_traces,
            tag_k=tag_k,
            neg_modes=neg_modes,
            save_ta_path=ta_folder,
            ta_title=f"exp55_{method}",
        )
        ov = result["overall"]
        print(f"  P={ov['precision']:.3f}  R={ov['recall']:.3f}  "
              f"F1={ov['f1']:.3f}  states={result['n_states']}")
        return {
            "method":   method,
            "params":   params,
            "status":   "ok",
            "n_states": result["n_states"],
            "n_edges":  result["n_edges"],
            "overall":  {k: v for k, v in ov.items()
                         if not isinstance(v, list)
                         and k not in ("save_path", "run_id")},
            "per_mode": result["per_mode"],
        }
    except Exception as e:
        error_type = type(e).__name__
        error_msg  = str(e)
        print(f"  FAILED ({error_type}): {error_msg[:200]}", flush=True)
        return {
            "method":     method,
            "params":     params,
            "status":     "failed",
            "error_type": error_type,
            "error_msg":  error_msg,
        }


# =============================================================================
# CONFIG FILE
# =============================================================================

def save_config(out_dir, train_traces, test_pos, neg_traces, neg_mode_names,
                actual_neg_mode):
    lines = [
        "=" * 55,
        "Run configuration — Exp 5.5 (ECG data)",
        "=" * 55,
        "",
        f"Timestamp        : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"Git hash         : {_git_hash()}",
        f"TAG k-future     : {TAG_K}",
        f"Negative mode    : {actual_neg_mode}",
        f"Recursion limit  : {sys.getrecursionlimit()}",
        "",
        "--- Methods ---",
        ]
    for method, params in METHODS:
        param_str = ", ".join(f"{k}={v}" for k, v in params.items())
        lines.append(f"  {method:8s}: {param_str}")

    lines += [
        "",
        "--- Data paths ---",
        f"  TRAIN_FOLDER    : {TRAIN_FOLDER}",
        f"  TEST_POS_FOLDER : {TEST_POS_FOLDER}",
        f"  NEG_DATA_FOLDER : {NEG_DATA_FOLDER}",
        "",
        "--- Dataset sizes ---",
        f"  train  : {len(train_traces)} traces",
        f"  test+  : {len(test_pos)} traces",
        f"  neg    : {len(neg_traces)} traces "
        f"({len(neg_mode_names)} mode{'s' if len(neg_mode_names) != 1 else ''})",
        "",
        "--- Negative modes ---",
    ]
    for name in neg_mode_names:
        lines.append(f"  {name}")

    lines += ["", "--- Output folder ---", f"  {out_dir}", "", "=" * 55]
    (out_dir / "config.txt").write_text("\n".join(lines))
    print(f"  Saved: {out_dir / 'config.txt'}")


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    out_dir = ROOT / "Data" / "Graphs" / "Metrics ECG" / timestamp
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output folder: {out_dir}")
    print(f"Recursion limit: {sys.getrecursionlimit()}\n")

    train_traces, test_pos, neg_traces, neg_modes, neg_mode_names = load_ecg_data()
    if not train_traces or not test_pos:
        raise SystemExit("ERROR: empty train or test set; nothing to run.")

    # The plotter uses negative_mode for display; tag it as actually-used here,
    # since "folder" can silently fall back to "synthetic" if the folder is missing.
    actual_neg_mode = (
        "folder" if (NEGATIVE_MODE == "folder"
                     and NEG_DATA_FOLDER is not None
                     and NEG_DATA_FOLDER.exists())
        else "synthetic"
    )

    save_config(out_dir, train_traces, test_pos, neg_traces,
                neg_mode_names, actual_neg_mode)

    log = {
        "timestamp":      timestamp,
        "git_hash":       _git_hash(),
        "tag_k":          TAG_K,
        "negative_mode":  actual_neg_mode,
        "train_folder":   str(TRAIN_FOLDER),
        "test_folder":    str(TEST_POS_FOLDER),
        "neg_folder":     str(NEG_DATA_FOLDER) if NEG_DATA_FOLDER else None,
        "n_train":        len(train_traces),
        "n_test_pos":     len(test_pos),
        "n_neg":          len(neg_traces),
        "neg_mode_names": neg_mode_names,
        "results":        [],
    }

    print()
    for method, params in METHODS:
        print(f"[{method}] {params} ...", flush=True)
        result = _run_one_method(
            method, params, train_traces, test_pos, neg_traces, neg_modes,
            TAG_K, out_dir,
        )
        log["results"].append(result)

        # Incremental save so a mid-run crash preserves prior methods.
        with open(out_dir / "results.json", "w") as f:
            json.dump(log, f, indent=2)

    # End-of-run summary
    failed = [r for r in log["results"] if r["status"] == "failed"]
    if failed:
        print(f"\n{len(failed)} method(s) failed:")
        for r in failed:
            print(f"  {r['method']:8s} : {r['error_type']} — {r['error_msg'][:100]}")
    else:
        print("\nAll methods completed successfully.")

    print(f"\nRun complete. To generate plots:")
    print(f"  python Metrics_ecg_plot.py")
    print(f"  python Metrics_ecg_plot.py --log {out_dir / 'results.json'}")