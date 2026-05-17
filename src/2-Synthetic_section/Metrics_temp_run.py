"""
exp_54_run.py
=============
Experiment 5.4 — Real temperature data.

Trains on real temperature traces, evaluates against either:
  - synthetic negatives (legacy mode), OR
  - injected negatives: real test traces with controlled anomaly perturbations
    applied directly to them.

Injected negatives are recommended: they preserve real-data baseline noise
characteristics while introducing controlled, known anomalies. This isolates
"can the TA detect anomalies?" from "does the TA generalize to real data?".

Crash-resilient like exp_51_52: each method runs in try/except, results.json
saved incrementally so SLURM SIGKILL preserves prior methods.

Output (timestamped folder under Graphs/exp_54/):
  config.txt
  results.json
"""



import sys
sys.setrecursionlimit(50000)
print("immediately after:", sys.getrecursionlimit())


import json
import subprocess
from collections import Counter
from datetime import datetime
from pathlib import Path


import numpy as np

# Project root is three levels up (script lives in src/2-Synthetic_section/).
ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from Generators import load_traces, NEG_MODE_NAMES, generate_negative_set
from Pipeline import run_pipeline


# =============================================================================
# CONFIG
# =============================================================================

TAG_K = 2

# Single method config per method (Path A). For full robustness sweep,
# extend with METHOD_VARIANTS like exp_51_52_run.py.
METHODS = [
    ("naive",   {"bins": 15}),
    ("sax",     {"w": 48,  "bins": 15}),
    ("persist", {"bins": 16}),
]

# ---- Data paths -------------------------------------------------------------
# Folder layout produced by helper.split_dataset(prefix="roomA-1day"):
#   <output_folder>/roomA-1day-train/        <- training CSVs
#   <output_folder>/roomA-1day-test/positive/<- positive test CSVs
TRAIN_FOLDER    = ROOT / "Data" / "3-ExtractInterval" / "1day-experiment" / "A-train"
TEST_POS_FOLDER = ROOT / "Data" / "3-ExtractInterval" / "1day-experiment" / "A-test" / "positive"
TRAIN_SPLIT     = 0.8   # used only if TEST_POS_FOLDER doesn't exist

# ---- Negative-trace mode ----------------------------------------------------
# "inject"    : copy real test traces and apply controlled perturbations
#               (recommended — keeps real-data noise characteristics)
# "synthetic" : use generate_negative_set (purely synthetic, used in 5.1/5.2)
NEGATIVE_MODE = "inject"

N_NEG_PER_MODE = 21

# Anomaly perturbation parameters — same magnitudes as synthetic 5.1/5.2
OFFSET_MAGNITUDE_C    = 15.0   # °C
SPIKE_MAGNITUDE_C     = 10.0   # °C
SPIKE_DURATION_S      = 600    # 10 minutes
STUCK_DURATION_S      = 3600   # 1 hour
PHASE_SHIFT_HOURS     = 6      # half-period for 12h cycle (24h day)


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


def _shuffle_traces(traces, seed=0):
    """Deterministic shuffle for reproducible splits."""
    rng = np.random.default_rng(seed)
    indices = list(range(len(traces)))
    rng.shuffle(indices)
    return [traces[i] for i in indices]

def _clean_trace(trace):
    """Drop samples where either time or value is NaN."""
    times = np.asarray(trace[0], dtype=float)
    temps = np.asarray(trace[1], dtype=float)
    mask  = ~(np.isnan(times) | np.isnan(temps))
    return (times[mask], temps[mask])


# =============================================================================
# ANOMALY INJECTION
# =============================================================================

def _inject_offset(trace, magnitude_c):
    """Add a constant offset to all temperatures in the trace."""
    times, temps = trace
    return (times.copy(), temps + magnitude_c)


def _inject_spike(trace, magnitude_c, duration_s):
    """Add a temperature spike of duration_s in the middle of the trace."""
    times, temps = trace
    times = np.asarray(times)
    temps = temps.copy().astype(float)
    midpoint = times[len(times) // 2]
    in_spike = (times >= midpoint) & (times < midpoint + duration_s)
    temps[in_spike] += magnitude_c
    return (times, temps)


def _inject_stuck(trace, duration_s):
    """Freeze temperature at a single value for duration_s in the middle."""
    times, temps = trace
    times = np.asarray(times)
    temps = temps.copy().astype(float)
    midpoint_idx  = len(times) // 2
    midpoint_time = times[midpoint_idx]
    in_stuck      = (times >= midpoint_time) & (times < midpoint_time + duration_s)
    temps[in_stuck] = float(temps[midpoint_idx])
    return (times, temps)


def _inject_phase_shift(trace, shift_hours):
    """
    Rotate the trace in time. Equivalent to shifting the time-of-day labels:
    daytime-shape now happens at night, etc. Detects whether the TA learned
    timed-dependent transitions or just symbol-sequence transitions.
    """
    times = np.asarray(trace[0])
    temps = np.asarray(trace[1]).copy()
    shift_seconds = shift_hours * 3600
    shift_idx = int(np.searchsorted(times - times[0], shift_seconds)) % len(temps)
    if shift_idx == 0:
        return (times, temps)
    rotated_temps = np.concatenate([temps[shift_idx:], temps[:shift_idx]])
    return (times, rotated_temps)


def build_injected_negatives(test_pos, n_per_mode, seed=42):
    """
    Take real test traces and apply each of the four anomaly modes to a
    sampled subset (with replacement). Returns (neg_traces, neg_modes)
    matching the interface generate_negative_set() uses.
    """
    rng = np.random.default_rng(seed)
    if len(test_pos) == 0:
        raise ValueError("No test positives to inject into.")

    neg_traces = []
    neg_modes  = []

    # Mode 0: spikes
    for _ in range(n_per_mode):
        idx = int(rng.integers(0, len(test_pos)))
        neg_traces.append(_inject_spike(test_pos[idx],
                                        SPIKE_MAGNITUDE_C, SPIKE_DURATION_S))
        neg_modes.append(0)

    # Mode 1: shifted (phase shift)
    for _ in range(n_per_mode):
        idx = int(rng.integers(0, len(test_pos)))
        neg_traces.append(_inject_phase_shift(test_pos[idx], PHASE_SHIFT_HOURS))
        neg_modes.append(1)

    # Mode 2: stuck
    for _ in range(n_per_mode):
        idx = int(rng.integers(0, len(test_pos)))
        neg_traces.append(_inject_stuck(test_pos[idx], STUCK_DURATION_S))
        neg_modes.append(2)

    # Mode 3: offset
    for _ in range(n_per_mode):
        idx = int(rng.integers(0, len(test_pos)))
        neg_traces.append(_inject_offset(test_pos[idx], OFFSET_MAGNITUDE_C))
        neg_modes.append(3)

    return neg_traces, neg_modes


# =============================================================================
# CONFIG FILE
# =============================================================================

def save_config(out_dir, train_traces, test_pos, neg_traces, neg_modes):
    lines = [
        "=" * 55,
        "Run configuration — Exp 5.4 (real temperature)",
        "=" * 55,
        "",
        f"Timestamp        : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"Git hash         : {_git_hash()}",
        f"TAG k-future     : {TAG_K}",
        f"Negative mode    : {NEGATIVE_MODE}",
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
        "",
        "--- Dataset sizes ---",
        f"  train  : {len(train_traces)} traces",
        f"  test+  : {len(test_pos)} traces",
        f"  neg    : {len(neg_traces)} traces ({len(set(neg_modes))} modes)",
        "",
        "--- Negative breakdown ---",
    ]
    mode_counts = Counter(neg_modes)
    for mode_int, count in sorted(mode_counts.items()):
        lines.append(f"  {NEG_MODE_NAMES[mode_int]:10s}: {count} traces")

    if NEGATIVE_MODE == "inject":
        lines += [
            "",
            "--- Injection parameters ---",
            f"  Spike magnitude  : +{SPIKE_MAGNITUDE_C} °C for {SPIKE_DURATION_S}s",
            f"  Phase shift      : {PHASE_SHIFT_HOURS}h rotation",
            f"  Stuck duration   : {STUCK_DURATION_S}s",
            f"  Offset magnitude : +{OFFSET_MAGNITUDE_C} °C",
        ]

    lines += ["", "--- Output folder ---", f"  {out_dir}", "", "=" * 55]
    (out_dir / "config.txt").write_text("\n".join(lines))
    print(f"  Saved: {out_dir / 'config.txt'}")


# =============================================================================
# DATA LOADING (with helpful error if folders missing)
# =============================================================================

def load_real_data():
    """
    Load training and test positive traces from disk. Falls back to splitting
    TRAIN_FOLDER if TEST_POS_FOLDER doesn't exist. Raises a descriptive
    FileNotFoundError if neither is present.
    """
    print("Loading real temperature traces...")

    if TEST_POS_FOLDER.exists():
        if not TRAIN_FOLDER.exists():
            raise FileNotFoundError(
                f"TEST_POS_FOLDER exists but TRAIN_FOLDER missing:\n"
                f"  TRAIN_FOLDER    : {TRAIN_FOLDER}\n"
                f"  TEST_POS_FOLDER : {TEST_POS_FOLDER}\n"
                f"Did split_dataset run successfully?"
            )
        train_traces = load_traces(TRAIN_FOLDER)
        test_pos     = load_traces(TEST_POS_FOLDER)
        train_traces = [_clean_trace(t) for t in train_traces]
        test_pos     = [_clean_trace(t) for t in test_pos]
        train_traces = [t for t in train_traces if len(t[0]) > 0]
        test_pos     = [t for t in test_pos     if len(t[0]) > 0]
        print(f"  TRAIN_FOLDER    : {len(train_traces)} traces")
        print(f"  TEST_POS_FOLDER : {len(test_pos)} traces")
        return train_traces, test_pos

    # Fallback: split TRAIN_FOLDER 80/20
    if not TRAIN_FOLDER.exists():
        raise FileNotFoundError(
            f"Neither TRAIN_FOLDER nor TEST_POS_FOLDER exists:\n"
            f"  TRAIN_FOLDER    : {TRAIN_FOLDER}\n"
            f"  TEST_POS_FOLDER : {TEST_POS_FOLDER}\n"
            f"Run processData.extract_time_intervals and split_dataset first."
        )

    print(f"  TEST_POS_FOLDER missing; splitting {TRAIN_FOLDER} "
          f"{TRAIN_SPLIT * 100:.0f}/{(1 - TRAIN_SPLIT) * 100:.0f}")
    all_traces = _shuffle_traces(load_traces(TRAIN_FOLDER), seed=0)
    split = int(len(all_traces) * TRAIN_SPLIT)
    train_traces = all_traces[:split]
    test_pos     = all_traces[split:]
    print(f"  Train : {len(train_traces)} traces")
    print(f"  Test+ : {len(test_pos)} traces")
    return train_traces, test_pos


# =============================================================================
# PIPELINE RUN WITH CRASH PROTECTION
# =============================================================================

def _run_one_method(method, params, train_traces, test_pos, neg_traces,
                    neg_modes, tag_k, out_dir):
    """Run one pipeline call; return result dict with status=ok|failed. Never raises."""
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
            ta_title=f"exp54_{method}",
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
# MAIN
# =============================================================================

if __name__ == "__main__":
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    out_dir = ROOT / "Data" / "Graphs" / "exp_54" / timestamp
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output folder: {out_dir}")
    print(f"Recursion limit: {sys.getrecursionlimit()}\n")

    train_traces, test_pos = load_real_data()
    if not train_traces or not test_pos:
        raise SystemExit("ERROR: empty train or test set; nothing to run.")

    # ---- Build negatives ----
    if NEGATIVE_MODE == "inject":
        neg_traces, neg_modes = build_injected_negatives(
            test_pos, N_NEG_PER_MODE, seed=42,
        )
        print(f"  Injected negatives: {len(neg_traces)} "
              f"({N_NEG_PER_MODE} per mode × 4 modes)")
    elif NEGATIVE_MODE == "synthetic":
        neg_traces, neg_modes = generate_negative_set(
            n_traces=N_NEG_PER_MODE * 4, seed=99,
        )
        print(f"  Synthetic negatives: {len(neg_traces)}")
    else:
        raise ValueError(f"Unknown NEGATIVE_MODE: {NEGATIVE_MODE}")

    save_config(out_dir, train_traces, test_pos, neg_traces, neg_modes)

    log = {
        "timestamp":     timestamp,
        "git_hash":      _git_hash(),
        "tag_k":         TAG_K,
        "negative_mode": NEGATIVE_MODE,
        "train_folder":  str(TRAIN_FOLDER),
        "test_folder":   str(TEST_POS_FOLDER),
        "n_train":       len(train_traces),
        "n_test_pos":    len(test_pos),
        "n_neg":         len(neg_traces),
        "injection_params": {
            "spike_magnitude_c":  SPIKE_MAGNITUDE_C,
            "spike_duration_s":   SPIKE_DURATION_S,
            "phase_shift_hours":  PHASE_SHIFT_HOURS,
            "stuck_duration_s":   STUCK_DURATION_S,
            "offset_magnitude_c": OFFSET_MAGNITUDE_C,
        } if NEGATIVE_MODE == "inject" else None,
        "results":       [],
    }

    print()
    for method, params in METHODS:
        print(f"[{method}] {params} ...", flush=True)
        result = _run_one_method(
            method, params, train_traces, test_pos, neg_traces, neg_modes,
            TAG_K, out_dir,
        )
        log["results"].append(result)

        # Incremental save so a crash mid-experiment preserves prior methods.
        with open(out_dir / "results.json", "w") as f:
            json.dump(log, f, indent=2)

    # ---- End-of-run failure summary ----
    failed = [r for r in log["results"] if r["status"] == "failed"]
    if failed:
        print(f"\n{len(failed)} method(s) failed:")
        for r in failed:
            print(f"  {r['method']:8s} : {r['error_type']} — {r['error_msg'][:100]}")
    else:
        print("\nAll methods completed successfully.")

    print(f"\nRun complete. To generate plots:")
    print(f"  python exp_54_plot.py")
    print(f"  python exp_54_plot.py --log {out_dir / 'results.json'}")