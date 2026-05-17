"""
Data_graphs_realdata.py
=======================
Generates positive-vs-negative-modes figures for BOTH the real-temperature
and ECG datasets, using one example trace per mode. Same figure layout for
both: an overlay and a 2x2 per-mode subplot grid.

Script location: src/2-Synthetic_section/

Output structure (under ROOT/Data/Graphs/):
  ecg_data_overview/
    fig_ecg_overlay.png
    fig_ecg_per_mode.png
  temp_data_overview/
    fig_temp_overlay.png
    fig_temp_per_mode.png

To add a third dataset, add another entry in DATASETS.
"""

import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import re

def _tid_from_filename(path):
    """Extract numeric tid (e.g. 42) from 'neg-foo-tid42.csv'. Returns 0 if absent."""
    match = re.search(r"tid(\d+)", path.stem)
    return int(match.group(1)) if match else 0
# ============================================================
# PATHS
# ============================================================

_HERE = Path(__file__).resolve()
ROOT  = _HERE.parent.parent.parent
OUT_ROOT = ROOT / "Data" / "Graphs"

def _inject_spike_temp(trace, magnitude, duration_samples, rng):
    values = np.array([float(v) for v, _ in trace], dtype=float)
    times  = np.array([float(t) for _, t in trace], dtype=float)
    n = len(values)
    if n < duration_samples + 2:
        return list(zip(values.tolist(), times.tolist()))
    start = int(rng.integers(0, n - duration_samples))
    sign  = int(rng.choice([-1, 1]))
    values[start:start + duration_samples] += sign * magnitude
    return list(zip(values.tolist(), times.tolist()))


def _inject_shifted_temp(trace, shift_fraction):
    values = np.array([float(v) for v, _ in trace], dtype=float)
    times  = np.array([float(t) for _, t in trace], dtype=float)
    n = len(values)
    shift_idx = int(n * shift_fraction)
    if shift_idx <= 0 or shift_idx >= n:
        return list(zip(values.tolist(), times.tolist()))
    rotated = np.concatenate([values[shift_idx:], values[:shift_idx]])
    return list(zip(rotated.tolist(), times.tolist()))


def _inject_stuck_temp(trace, duration_samples):
    values = np.array([float(v) for v, _ in trace], dtype=float)
    times  = np.array([float(t) for _, t in trace], dtype=float)
    n = len(values)
    if n < duration_samples + 2:
        return list(zip(values.tolist(), times.tolist()))
    start = max(0, n // 3 - duration_samples // 2)
    end   = min(n, start + duration_samples)
    values[start:end] = values[start]
    return list(zip(values.tolist(), times.tolist()))


def _inject_offset_temp(trace, magnitude):
    values = np.array([float(v) for v, _ in trace], dtype=float) + magnitude
    times  = np.array([float(t) for _, t in trace], dtype=float)
    return list(zip(values.tolist(), times.tolist()))

def make_inmemory_negatives_temp(pos_path, crop_samples, time_to_hours_flag,
                                 *, spike_magnitude=5.0,
                                 spike_duration_samples=2,
                                 phase_shift_fraction=0.25,
                                 stuck_duration_samples=60,
                                 offset_magnitude=3.0,
                                 seed=42):
    """
    Apply all four temperature injections to a single positive trace.
    Returns {mode: (times, values)} aligned with the same base positive,
    so the per-mode figure shows what each mode does to THIS trace.
    """
    t_raw, v_raw = load_csv(pos_path)
    trace = list(zip(v_raw.tolist(), t_raw.tolist()))
    rng = np.random.default_rng(seed)

    perturbed = {
        "spikes":  _inject_spike_temp(trace, spike_magnitude, spike_duration_samples, rng),
        "shifted": _inject_shifted_temp(trace, phase_shift_fraction),
        #"stuck":   _inject_stuck_temp(trace, stuck_duration_samples),
        "offset":  _inject_offset_temp(trace, offset_magnitude),
    }

    out = {}
    for mode, p_trace in perturbed.items():
        v = np.array([float(val) for val, _ in p_trace])
        t = np.array([float(tm)  for _, tm  in p_trace])
        t, v = crop_to_middle(t, v, crop_samples)
        if time_to_hours_flag:
            t = to_hours(t)
        out[mode] = (t, v)
    return out


# ============================================================
# STYLE (shared across datasets)
# ============================================================

COLOR_POSITIVE = "#2E7D32"
NEG_STYLES = {
    "spikes":  ("#E53935", "-"),    # red solid
    "shifted": ("#F9A825", "-"),    # amber solid
    "stuck":   ("#7B1FA2", "--"),   # purple dashed
    "offset":  ("#BF360C", "-"),    # deep orange solid
}
MODE_ORDER = ["spikes", "shifted", "stuck", "offset"]


# ============================================================
# DATASETS
# Per-dataset configuration. To swap real temperature for synthetic,
# change pos_folder / neg_folder paths only.
# ============================================================

DATASETS = {
    "ecg": {
        "pos_folder":     ROOT / "Data" / "3-ExtractInterval" / "ecg" / "1beat-experiment1" / "1beat-test" / "positive",
        "neg_folder":     ROOT / "Data" / "3-ExtractInterval" / "ecg" / "1beat-experiment1" / "1beat-test" / "negative",
        "out_subdir":     "ecg_data_overview",
        "crop_samples":   250,                # ≈ 700 ms at 360 Hz — one beat
        "time_to_hours":  False,              # ECG: leave time axis as-is
        "x_label":        "Sample index",
        "y_label":        "ECG amplitude",
        "title_prefix":   "ECG",
    },
    "temperature": {
        "pos_folder":     ROOT / "Data" / "3-ExtractInterval" / "1day-experiment" / "A-test" / "positive",
        "neg_folder":     ROOT / "Data" / "3-ExtractInterval" / "1day-experiment" / "A-test" / "negative",
        "out_subdir":     "temp_data_overview",
        "crop_samples":   None,               # show full 24h trace
        "time_to_hours":  True,               # convert seconds → hours
        "x_label":        "Time (hours)",
        "y_label":        "Temperature (°C)",
        "title_prefix":   "Temperature",
        "inmemory_negatives": True,
    },
}

# ============================================================
# Helpers
# ============================================================

def _ordered_modes(neg_by_mode):
    """MODE_ORDER first, then any extras (e.g., generic 'anomaly')."""
    ordered = [m for m in MODE_ORDER if m in neg_by_mode]
    extras  = [m for m in neg_by_mode if m not in MODE_ORDER]
    return ordered + extras


def load_negatives(neg_folder, crop_samples, time_to_hours_flag):
    """
    Returns {mode: (times, values)} with one example per mode.

    Tries two layouts:
      1. Subfolder layout: neg_folder/<mode>/*.csv  (ECG convention)
      2. Flat layout:      filenames contain mode name, all in neg_folder/
    Fallback (no mode info in filenames): {"anomaly": <first CSV>}.
    """
    out = {}

    # --- Layout 1: per-mode subfolders ---
    if any((neg_folder / m).is_dir() for m in MODE_ORDER):
        for mode in MODE_ORDER:
            mode_folder = neg_folder / mode
            if not mode_folder.exists():
                print(f"  Warning: {mode}/ subfolder missing")
                continue
            t_v = load_first(mode_folder, crop_samples, time_to_hours_flag)
            if t_v is None:
                print(f"  Warning: no CSVs in {mode}/")
                continue
            out[mode] = t_v
            print(f"  {mode}: {len(t_v[0])} samples")
        return out

    # --- Flat folder: sort by numeric tid to avoid lex-sort scrambling ---
    all_csvs = list(neg_folder.glob("*.csv"))
    ...
    # --- Flat folder: sort by numeric tid to avoid lex-sort scrambling ---
    all_csvs = list(neg_folder.glob("*.csv"))
    if not all_csvs:
        print(f"  Warning: no CSVs in {neg_folder}")
        return out
    all_csvs.sort(key=_tid_from_filename)
    print(f"  Flat layout: {len(all_csvs)} CSV files (sorted by tid)")

    # --- Layout 2: mode name in filename ---
    for mode in MODE_ORDER:
        matches = [p for p in all_csvs if mode in p.stem.lower()]
        if matches:
            t, v = load_csv(matches[0])
            t, v = crop_to_middle(t, v, crop_samples)
            if time_to_hours_flag:
                t = to_hours(t)
            out[mode] = (t, v)
            print(f"  {mode}: {len(t)} samples ({matches[0].name})")
    if out:
        return out

    # --- Layout 3: positional bucketing ---
    # No mode info in filenames; assume the generator wrote modes sequentially:
    # files [0, n_per_mode) = mode 0, [n_per_mode, 2*n_per_mode) = mode 1, ...
    n = len(all_csvs)
    n_per_mode = n // len(MODE_ORDER)
    if n_per_mode == 0:
        print(f"  Too few files for positional split; using first as 'anomaly'")
        t, v = load_csv(all_csvs[0])
        t, v = crop_to_middle(t, v, crop_samples)
        if time_to_hours_flag:
            t = to_hours(t)
        out["anomaly"] = (t, v)
        return out

    print(f"  Positional bucketing: ~{n_per_mode} files per mode "
          f"({n} total / {len(MODE_ORDER)} modes)")
    for i, mode in enumerate(MODE_ORDER):
        idx = i * n_per_mode
        t, v = load_csv(all_csvs[idx])
        t, v = crop_to_middle(t, v, crop_samples)
        if time_to_hours_flag:
            t = to_hours(t)
        out[mode] = (t, v)
        print(f"  {mode}: {len(t)} samples ({all_csvs[idx].name})")

    return out

# ============================================================
# LOADING / PREPROCESSING
# ============================================================

def load_csv(path):
    """Semicolon-delimited CSV with header. Returns (times, values)."""
    data = np.genfromtxt(path, delimiter=";", skip_header=1)
    if data.ndim == 1:
        data = data.reshape(1, -1)
    return data[:, 0], data[:, 1]


def to_hours(times):
    """Convert seconds → hours anchored at the first sample."""
    return (times - times[0]) / 3600.0


def crop_to_middle(times, values, n_samples):
    """Center a fixed-width window on the trace midpoint."""
    if n_samples is None or len(values) <= n_samples:
        return times, values
    mid   = len(values) // 2
    half  = n_samples // 2
    start = max(0, mid - half)
    end   = min(len(values), start + n_samples)
    return times[start:end], values[start:end]


def load_first(folder, crop_samples=None, time_to_hours_flag=False):
    """Load the first CSV alphabetically; crop and convert time as configured."""
    csvs = sorted(folder.glob("*.csv"))
    if not csvs:
        return None
    t, v = load_csv(csvs[0])
    t, v = crop_to_middle(t, v, crop_samples)
    if time_to_hours_flag:
        t = to_hours(t)
    return t, v


# ============================================================
# FIGURES (generic over dataset)
# ============================================================

def fig_overlay(pos, neg_by_mode, out_path, *, title, xlabel, ylabel):
    """Single panel: positive + one example per mode, all overlaid."""
    fig, ax = plt.subplots(figsize=(13, 5))

    pos_t, pos_v = pos
    ax.plot(pos_t, pos_v, color=COLOR_POSITIVE, linewidth=2.0,
            label="Positive (Normal)", zorder=5)

    for mode in _ordered_modes(neg_by_mode):
        t, v = neg_by_mode[mode]
        color, ls = NEG_STYLES.get(mode, ("#444444", "-"))
        ax.plot(t, v, color=color, linestyle=ls, linewidth=1.3,
                label=f"Neg: {mode}")

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(True, linestyle="--", alpha=0.35)

    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")

def fig_examples_grid(pos_folder, neg_folder, out_path, *, title, xlabel, ylabel,
                      crop_samples, time_to_hours_flag, n_examples=10):
    """
    n_examples × 4 grid for verification. Rows = different example pairs,
    cols = anomaly modes. Each cell shows positive (gray) + one negative (color).

    Uses sorted file order — example N pairs the Nth positive with the Nth
    negative from each mode subfolder. If a mode has fewer files than
    n_examples, the missing cells are left empty.
    """
    pos_csvs = sorted(pos_folder.glob("*.csv"))
    if not pos_csvs:
        print(f"  No positive CSVs for grid figure")
        return

    # Available files per mode
    mode_csvs_by_name = {}
    for mode in MODE_ORDER:
        mode_folder = neg_folder / mode
        if mode_folder.is_dir():
            mode_csvs_by_name[mode] = sorted(mode_folder.glob("*.csv"))
        else:
            mode_csvs_by_name[mode] = []

    n_rows = min(n_examples, len(pos_csvs))
    if n_rows == 0:
        return

    fig, axes = plt.subplots(
        n_rows, len(MODE_ORDER),
        figsize=(4 * len(MODE_ORDER), 1.8 * n_rows),
        sharex=True, sharey=True, squeeze=False,
    )
    fig.suptitle(title, fontsize=13)

    for row in range(n_rows):
        # Positive at this row
        pos_t, pos_v = load_csv(pos_csvs[row])
        pos_t, pos_v = crop_to_middle(pos_t, pos_v, crop_samples)
        if time_to_hours_flag:
            pos_t = to_hours(pos_t)

        for col, mode in enumerate(MODE_ORDER):
            ax = axes[row, col]
            ax.plot(pos_t, pos_v, color="gray", linewidth=0.9, alpha=0.6, zorder=1)

            mode_files = mode_csvs_by_name[mode]
            if row < len(mode_files):
                t, v = load_csv(mode_files[row])
                t, v = crop_to_middle(t, v, crop_samples)
                if time_to_hours_flag:
                    t = to_hours(t)
                color, ls = NEG_STYLES.get(mode, ("#444444", "-"))
                ax.plot(t, v, color=color, linestyle=ls, linewidth=1.1, zorder=2)
            else:
                ax.text(0.5, 0.5, "(no file)", ha="center", va="center",
                        transform=ax.transAxes, fontsize=8, color="#888888")

            ax.grid(True, linestyle="--", alpha=0.3)
            if row == 0:
                ax.set_title(mode.capitalize(), fontsize=10)
            if row == n_rows - 1:
                ax.set_xlabel(xlabel, fontsize=8)
            if col == 0:
                ax.set_ylabel(f"#{row}", fontsize=8)

    # Outer ylabel for the whole grid
    fig.supylabel(ylabel, fontsize=10)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")

def fig_per_mode_subplots(pos, neg_by_mode, out_path, *, title, xlabel, ylabel):
    """2x2 grid: each panel shows positive (gray) + one mode (color)."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 8), sharex=True, sharey=True)
    fig.suptitle(title, fontsize=13)

    pos_t, pos_v = pos
    modes = _ordered_modes(neg_by_mode)

    for ax, mode in zip(axes.flat, modes):
        ax.plot(pos_t, pos_v, color="gray", linewidth=1.0, alpha=0.6,
                label="Positive", zorder=1)
        t, v = neg_by_mode[mode]
        color, ls = NEG_STYLES.get(mode, ("#444444", "-"))
        ax.plot(t, v, color=color, linestyle=ls, linewidth=1.5,
                label=f"Neg: {mode}", zorder=2)
        ax.set_title(mode.capitalize())
        ax.grid(True, linestyle="--", alpha=0.35)
        ax.legend(loc="upper right", fontsize=8)

    # Hide unused panels if fewer than 4 modes were loaded.
    for ax in list(axes.flat)[len(modes):]:
        ax.set_visible(False)

    for ax in axes[1, :]:
        ax.set_xlabel(xlabel)
    for ax in axes[:, 0]:
        ax.set_ylabel(ylabel)

    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")

# ============================================================
# DATASET PROCESSING
# ============================================================

def process_dataset(name, cfg, out_root):
    out_dir = out_root / cfg["out_subdir"]
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n{'=' * 50}\nDataset: {name}\nOutput:  {out_dir}\n{'=' * 50}")

    if not cfg["pos_folder"].exists():
        print(f"  SKIP: positive folder not found: {cfg['pos_folder']}")
        return

    # Pick the first positive as the display + injection base.
    pos_csvs = sorted(cfg["pos_folder"].glob("*.csv"))
    if not pos_csvs:
        print(f"  ERROR: no positive CSVs in {cfg['pos_folder']}")
        return
    pos_path = pos_csvs[0]
    pos = load_first(cfg["pos_folder"], cfg["crop_samples"], cfg["time_to_hours"])
    print(f"  Positive: {len(pos[0])} samples ({pos_path.name})")

    # In-memory injection (cleaner figure: all four modes share the same base).
    if cfg.get("inmemory_negatives"):
        print(f"  Using in-memory injection (base: {pos_path.name})")
        neg_by_mode = make_inmemory_negatives_temp(
            pos_path, cfg["crop_samples"], cfg["time_to_hours"]
        )
    else:
        if not cfg["neg_folder"].exists():
            print(f"  SKIP: negative folder not found: {cfg['neg_folder']}")
            return
        neg_by_mode = load_negatives(
            cfg["neg_folder"], cfg["crop_samples"], cfg["time_to_hours"]
        )

    if not neg_by_mode:
        print(f"  ERROR: no negative traces for {name}")
        return

    title_prefix = cfg["title_prefix"]
    fig_overlay(pos, neg_by_mode, out_dir / f"fig_{name}_overlay.png",
                title=f"{title_prefix}: Positive vs. All Negative Modes",
                xlabel=cfg["x_label"], ylabel=cfg["y_label"])
    fig_per_mode_subplots(pos, neg_by_mode, out_dir / f"fig_{name}_per_mode.png",
                          title=f"{title_prefix}: Positive vs. Each Negative Mode",
                          xlabel=cfg["x_label"], ylabel=cfg["y_label"])


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    print(f"Output root: {OUT_ROOT}")

    for name, cfg in DATASETS.items():
        process_dataset(name, cfg, OUT_ROOT)

    print(f"\nDone. Output: {OUT_ROOT}")