"""
Data_graphs.py
==============
Generates data-section figures for the thesis.

Script location: src/2-Synthetic_section/Data_graphs.py
  parent          = src/2-Synthetic_section/
  parent.parent   = src/               <- SRC  (for Generators import)
  parent.parent.parent = project root  <- ROOT (for Data/ paths)

Produces (in ROOT/Data/Graphs/data_overview/):
  fig1_single_traces.png     — one clean trace + one noisy trace
  fig2_training_overview.png — all clean vs all noisy traces overlaid
  fig3_test_cases.png        — positive trace vs all four negative modes

Negative traces for fig3 are generated via shared.generators.generate_negative_set
so that the modes are exactly what your experiments use.
Falls back to inline generation if the import fails.

CSV format assumed: semicolon-delimited, one header row, columns: time;value
Time unit auto-detected: values covering a full day (~86400) are treated as
seconds and converted to hours for display.
"""

import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

# ============================================================
# PATHS
# ============================================================

# src/2-Synthetic_section/Data_graphs.py
_HERE = Path(__file__).resolve()
SRC   = _HERE.parent.parent          # src/
ROOT  = _HERE.parent.parent.parent   # CPS---Timed-automata/

# Add src/ to path so shared.generators is importable
sys.path.insert(0, str(SRC))

CLEAN_FOLDER = ROOT / "Data" / "synthetic_data" / "clean_train"
NOISY_FOLDER = ROOT / "Data" / "synthetic_data" / "noisy_train"
NEG_FOLDER   = ROOT / "Data" / "synthetic_data" / "negative"
OUT_DIR      = ROOT / "Data" / "Graphs" / "data_overview"

# ============================================================
# SHARED GENERATORS IMPORT
# ============================================================

try:
    from Generators import generate_negative_set, NEG_MODE_NAMES
    _HAS_GENERATORS = True
    print("shared.generators loaded successfully.")
except ImportError as _e:
    _HAS_GENERATORS = False
    print(f"Warning: could not import shared.generators ({_e}).")
    print("Falling back to inline negative generation.")

# ============================================================
# STYLE
# ============================================================

COLOR_CLEAN    = "#2E7D32"
COLOR_NOISY    = "#1565C0"
COLOR_POSITIVE = "#2E7D32"

# One entry per mode 0-3
NEG_STYLES = [
    ("#E53935", "-"),    # Spikes     - red solid
    ("#F9A825", "-"),    # Shifted    - amber solid
    ("#7B1FA2", "--"),   # Stuck      - purple dashed
    ("#BF360C", "-"),    # Off-set    - deep orange solid
]

# ============================================================
# CSV LOADER
# ============================================================

def load_csv(path: Path):
    """Returns list of (value, time) tuples. Tries ; then , delimiter."""
    for delim in [";", ","]:
        try:
            data = np.genfromtxt(path, delimiter=delim,
                                 dtype=str, skip_header=1)
            if data.ndim == 1:
                data = data.reshape(1, -1)
            if data.shape[1] < 2:
                continue
            times = data[:, 0].astype(float).astype(int)
            vals  = data[:, 1].astype(float)
            return [(float(v), int(t)) for v, t in zip(vals, times)]
        except Exception:
            continue
    raise ValueError(f"Could not parse {path}")


def load_folder(folder: Path):
    traces = []
    for p in sorted(folder.glob("*.csv")):
        try:
            traces.append(load_csv(p))
        except Exception as e:
            print(f"  Warning: skipping {p.name} - {e}")
    return traces


# ============================================================
# TIME CONVERSION
#
# A full day in seconds = 86400.
# Thresholds:
#   span > 7200  -> seconds  (divide by 3600)
#   span > 120   -> minutes  (divide by 60)
#   else         -> hours    (use as-is)
#
# This correctly handles:
#   86400 s  -> 86400/3600 = 24 h  (your synthetic data)
#   1440 min -> 1440/60   = 24 h
#   24 h     -> 24
# ============================================================

def to_hours(times: np.ndarray) -> np.ndarray:
    span = float(times[-1] - times[0])
    if span > 7200:
        return (times - times[0]) / 3600.0   # seconds -> hours
    elif span > 120:
        return (times - times[0]) / 60.0     # minutes -> hours
    else:
        return (times - times[0]).astype(float)


def extract(trace):
    """
    Handles two trace formats:
      A) (times_array, temps_array)  — from shared.generators
      B) [(value, time), ...]        — from load_csv
    """
    if isinstance(trace[0], np.ndarray):
        # Format A: tuple of two arrays
        times, vals = trace
    else:
        # Format B: list of (value, time) tuples
        vals  = np.array([v for v, _ in trace])
        times = np.array([t for _, t in trace], dtype=float)
    return to_hours(np.array(times, dtype=float)), np.array(vals)


# ============================================================
# NEGATIVE GENERATION
# ============================================================

# --- Via shared.generators (preferred) ---

def get_negatives_from_generators(n_per_mode: int = 1, seed: int = 42):
    """
    Calls generate_negative_set to produce n_per_mode traces per mode.
    Returns list of (trace, label_str) tuples.
    """
    n_total = n_per_mode * len(NEG_MODE_NAMES)
    neg_traces, neg_modes = generate_negative_set(n_traces=n_total, seed=seed)

    result = []
    for trace, mode_int in zip(neg_traces, neg_modes):
        label = NEG_MODE_NAMES[mode_int]
        result.append((trace, label))
    return result


# --- Inline fallback ---

_FALLBACK_MODE_LABELS = {
    0: "Spikes (Mode 0)",
    1: "Shifted (Mode 1)",
    2: "Stuck (Mode 2)",
    3: "Off-set (Mode 3)",
}


def _apply_mode_inline(vals: np.ndarray, mode: int, seed: int = 0) -> np.ndarray:
    s   = vals.copy()
    n   = len(s)
    rng = np.random.default_rng(seed + mode * 17)

    if mode == 0:   # Spikes
        for _ in range(rng.integers(2, 5)):
            idx   = rng.integers(int(n * 0.05), int(n * 0.5))
            s[idx:idx + rng.integers(1, 4)] += rng.uniform(6.0, 10.0)
    elif mode == 1: # Shifted
        s += rng.uniform(1.2, 2.0)
    elif mode == 2: # Stuck
        s[:int(n * 0.35)] = s[0]
    elif mode == 3: # Off-set
        s += rng.uniform(12.0, 16.0)

    return s


def get_negatives_inline(clean_trace, seed: int = 42):
    """Generates one example of each mode from a clean trace."""
    vals  = np.array([v for v, _ in clean_trace])
    times = [t for _, t in clean_trace]
    result = []
    for mode in range(4):
        neg_vals  = _apply_mode_inline(vals, mode, seed=seed)
        neg_trace = [(float(v), t) for v, t in zip(neg_vals, times)]
        result.append((neg_trace, _FALLBACK_MODE_LABELS[mode]))
    return result


# ============================================================
# FIGURE 1 - Single clean + single noisy trace
# ============================================================

def fig1_single_traces(clean_traces, noisy_traces, out_path):
    fig, axes = plt.subplots(1, 2, figsize=(14, 4))

    for ax, trace, color, title in [
        (axes[0], clean_traces[0], COLOR_CLEAN, "Clean synthetic trace - example"),
        (axes[1], noisy_traces[0], COLOR_NOISY, "Noisy synthetic trace - example"),
    ]:
        t, v = extract(trace)
        ax.plot(t, v, color=color, linewidth=0.9)
        ax.set_xlabel("Time (hours)")
        ax.set_ylabel("Temperature (°C)")
        ax.set_title(title)
        ax.set_xlim(0, 24)
        ax.grid(True, linestyle="--", alpha=0.4)

    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


# ============================================================
# FIGURE 2 - All clean vs all noisy overlaid
# ============================================================

def fig2_training_overview(clean_traces, noisy_traces, out_path):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Training data: clean vs noisy", fontsize=13)

    panels = [
        (axes[0], clean_traces, plt.cm.YlGn,  f"Clean (n={len(clean_traces)})"),
        (axes[1], noisy_traces, plt.cm.Blues,  f"Noisy (n={len(noisy_traces)})"),
    ]

    for ax, traces, cmap, title in panels:
        colors   = cmap(np.linspace(0.4, 0.85, len(traces)))
        all_vals = []
        ref_t    = None

        for trace, c in zip(traces, colors):
            t, v = extract(trace)
            ax.plot(t, v, color=c, alpha=0.35, linewidth=0.5)
            all_vals.append(v)
            if ref_t is None:
                ref_t = t

        min_len = min(len(v) for v in all_vals)
        mean_v  = np.mean([v[:min_len] for v in all_vals], axis=0)
        ax.plot(ref_t[:min_len], mean_v, color="black", linewidth=1.8,
                linestyle="--", label="Mean", zorder=5)

        ax.set_xlabel("Time (hours)")
        ax.set_ylabel("Temperature (°C)")
        ax.set_title(title)
        ax.set_xlim(0, 24)
        ax.legend(loc="upper right")
        ax.grid(True, linestyle="--", alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


# ============================================================
# FIGURE 3 - Positive trace vs all four negative modes
# ============================================================

def fig3_test_cases(clean_traces, neg_traces_with_labels, out_path):
    """
    Plots one clean trace as positive and one example per mode.
    De-duplicates by label so each mode appears exactly once.
    """
    seen = {}
    for trace, label in neg_traces_with_labels:
        if label not in seen:
            seen[label] = trace

    fig, ax = plt.subplots(figsize=(13, 5))

    t, v = extract(clean_traces[0])
    ax.plot(t, v, color=COLOR_POSITIVE, linewidth=2.0,
            label="Positive (Normal)", zorder=5)

    for i, (label, trace) in enumerate(seen.items()):
        color, ls = NEG_STYLES[i % len(NEG_STYLES)]
        t, v = extract(trace)
        ax.plot(t, v, color=color, linestyle=ls, linewidth=1.3,
                label=f"Neg: {label}")

    ax.set_xlabel("Time (Hours)")
    ax.set_ylabel("Temperature (°C)")
    ax.set_title("Classifier Validation: Positive vs. All Negative Modes")
    ax.set_xlim(0, 24)
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(True, linestyle="--", alpha=0.35)

    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Output folder: {OUT_DIR}\n")

    missing = [f for f in [CLEAN_FOLDER, NOISY_FOLDER]
               if not f.exists()]
    if missing:
        print("ERROR: folders not found:")
        for m in missing:
            print(f"  {m}")
        sys.exit(1)

    print("Loading clean traces...")
    clean_traces = load_folder(CLEAN_FOLDER)
    print(f"  Loaded {len(clean_traces)} clean traces")

    # Quick sanity check — print the raw time span so you can verify unit detection
    if clean_traces:
        sample_times = np.array([t for _, t in clean_traces[0]])
        span = float(sample_times[-1] - sample_times[0])
        print(f"  Time span of first trace: {span:.0f} raw units "
              f"-> {span/3600:.1f} h (if seconds) | "
              f"{span/60:.1f} h (if minutes)")

    print("Loading noisy traces...")
    noisy_traces = load_folder(NOISY_FOLDER)
    print(f"  Loaded {len(noisy_traces)} noisy traces")

    print("Preparing negative traces for fig3...")
    if _HAS_GENERATORS:
        # Use your actual generator — 4 traces total, one per mode
        neg_traces = get_negatives_from_generators(n_per_mode=1, seed=42)
        print(f"  Generated via shared.generators: "
              f"{[label for _, label in neg_traces]}")
    else:
        # Fallback: apply modes inline to first clean trace
        neg_traces = get_negatives_inline(clean_traces[0], seed=42)
        print(f"  Generated inline (fallback): "
              f"{[label for _, label in neg_traces]}")

    if not clean_traces or not noisy_traces:
        print("ERROR: clean or noisy dataset is empty.")
        sys.exit(1)

    print("\nGenerating figures...")

    fig1_single_traces(
        clean_traces, noisy_traces,
        OUT_DIR / "fig1_single_traces.png"
    )
    fig2_training_overview(
        clean_traces, noisy_traces,
        OUT_DIR / "fig2_training_overview.png"
    )
    fig3_test_cases(
        clean_traces, neg_traces,
        OUT_DIR / "fig3_test_cases.png"
    )

    print(f"\nDone. All figures saved to {OUT_DIR}")