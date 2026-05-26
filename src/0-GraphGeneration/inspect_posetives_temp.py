"""
Inspect_positives.py
====================
Sanity-check script: renders ALL positive traces from each dataset's
positive folder. One small panel per file plus a single overlay of all
traces. Useful when downstream figures look wrong and you want to verify
the input data trace by trace.

Script location: src/2-Synthetic_section/Inspect_positives.py

Produces (in ROOT/Data/Graphs/positive_inspection/):
  <dataset>_grid.png      — N-row × M-col grid, one panel per file
  <dataset>_overlay.png   — every trace on one axis, low alpha

To inspect a different folder, add an entry to DATASETS.
"""

import math
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

# ============================================================
# PATHS
# ============================================================

_HERE = Path(__file__).resolve()
ROOT  = _HERE.parent.parent.parent
OUT_DIR = ROOT / "Data" / "Graphs" / "positive_inspection"


# ============================================================
# DATASETS
# ============================================================

DATASETS = {
    "temperature": {
        "pos_folder":    ROOT / "Data" / "3-ExtractInterval" / "1day-experiment" / "A-test" / "positive",
        "crop_samples":  None,
        "time_to_hours": True,
        "x_label":       "Time (hours)",
        "y_label":       "Temperature (°C)",
    },
    "ecg": {
        "pos_folder":    ROOT / "Data" / "3-ExtractInterval" / "ecg" / "1beat-experiment1" / "1beat-test" / "positive",
        "crop_samples":  250,
        "time_to_hours": False,
        "x_label":       "Sample index",
        "y_label":       "ECG amplitude",
    },
}

# Also inspect train folders — bad training data would explain weird negatives.
DATASETS["temperature_train"] = {
    "pos_folder":    ROOT / "Data" / "3-ExtractInterval" / "1day-experiment" / "A-train",
    "crop_samples":  None,
    "time_to_hours": True,
    "x_label":       "Time (hours)",
    "y_label":       "Temperature (°C)",
}
def fig_overlay_raw(folder, out_path, *, title, xlabel, ylabel):
    """
    Overlay every CSV in folder using its raw time column verbatim —
    no cropping, no unit conversion, no normalization. Files whose length
    differs from the modal length are drawn in red so any time-axis
    inconsistency between files becomes visually obvious.
    """
    from collections import Counter

    csvs = sorted(folder.glob("*.csv"))
    if not csvs:
        return

    raw_traces = []
    for path in csvs:
        try:
            t, v = load_csv(path)
            raw_traces.append((path.name, t, v))
        except Exception as e:
            print(f"  Warning: failed to load {path.name}: {e}")

    mode_len   = Counter(len(t) for _, t, _ in raw_traces).most_common(1)[0][0]
    n_outliers = 0

    fig, ax = plt.subplots(figsize=(14, 5))
    for name, t, v in raw_traces:
        if len(t) == mode_len:
            ax.plot(t, v, color="#1565C0", linewidth=0.5, alpha=0.25)
        else:
            ax.plot(t, v, color="#E53935", linewidth=1.2, alpha=0.9,
                    label=f"{name} (n={len(t)})")
            n_outliers += 1

    suffix = f", {n_outliers} length-outliers in red" if n_outliers else ""
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(f"{title}  (n={len(raw_traces)} traces, "
                 f"mode length={mode_len}{suffix})")
    ax.grid(True, linestyle="--", alpha=0.35)
    if n_outliers:
        ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")

def fig_individual(traces, out_dir, prefix, *, title, xlabel, ylabel):
    """
    Save each trace as its own SVG. Per-trace inspection at full
    resolution, with a shared y-range so amplitudes are visually
    comparable across files.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Shared y-range for fair visual comparison across files.
    all_v = np.concatenate([v for _, _, v in traces])
    y_lo, y_hi = float(np.nanmin(all_v)), float(np.nanmax(all_v))
    y_pad = 0.05 * (y_hi - y_lo) if y_hi > y_lo else 1.0

    for i, (name, t, v) in enumerate(traces):
        fig, ax = plt.subplots(figsize=(8, 3))
        ax.plot(t, v, color="#1565C0", linewidth=0.8)
        ax.set_ylim(y_lo - y_pad, y_hi + y_pad)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(f"{title} — #{i}: {name}", fontsize=10)
        ax.grid(True, linestyle="--", alpha=0.3)
        fig.tight_layout()

        out_path = out_dir / f"{prefix}_{i:03d}_{Path(name).stem}.svg"
        fig.savefig(out_path, format="svg", bbox_inches="tight")
        plt.close(fig)

    print(f"  Saved: {len(traces)} individual SVGs to {out_dir}")


# ============================================================
# LOADING
# ============================================================

def load_csv(path):
    """Semicolon-delimited CSV with header. Returns (times, values)."""
    data = np.genfromtxt(path, delimiter=";", skip_header=1)
    if data.ndim == 1:
        data = data.reshape(1, -1)
    return data[:, 0], data[:, 1]


def to_hours(times):
    return (times - times[0]) / 3600.0


def crop_to_middle(times, values, n_samples):
    if n_samples is None or len(values) <= n_samples:
        return times, values
    mid   = len(values) // 2
    half  = n_samples // 2
    start = max(0, mid - half)
    end   = min(len(values), start + n_samples)
    return times[start:end], values[start:end]


def load_traces_in_folder(folder, crop_samples, time_to_hours_flag):
    """Return list of (filename, times, values) for every CSV in folder."""
    csvs = sorted(folder.glob("*.csv"))
    traces = []
    for path in csvs:
        try:
            t, v = load_csv(path)
            t, v = crop_to_middle(t, v, crop_samples)
            if time_to_hours_flag:
                t = to_hours(t)
            traces.append((path.name, t, v))
        except Exception as e:
            print(f"  Warning: failed to load {path.name}: {e}")
    return traces


# ============================================================
# FIGURES
# ============================================================

def fig_grid_all(traces, out_path, *, title, xlabel, ylabel, n_cols=8):
    """One small panel per trace in a fixed-width grid."""
    n = len(traces)
    n_rows = math.ceil(n / n_cols)
    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(2.0 * n_cols, 1.4 * n_rows),
        sharex=True, sharey=True, squeeze=False,
    )
    fig.suptitle(f"{title}  (n={n})", fontsize=13)

    # Compute shared y-range across all traces for fair comparison.
    all_v = np.concatenate([v for _, _, v in traces])
    y_lo, y_hi = float(np.nanmin(all_v)), float(np.nanmax(all_v))
    y_pad = 0.05 * (y_hi - y_lo) if y_hi > y_lo else 1.0

    for i, (name, t, v) in enumerate(traces):
        row, col = divmod(i, n_cols)
        ax = axes[row, col]
        ax.plot(t, v, color="#1565C0", linewidth=0.6, alpha=0.9)
        ax.set_ylim(y_lo - y_pad, y_hi + y_pad)
        ax.grid(True, linestyle="--", alpha=0.25)
        # File index in the top-left of the panel — readable, doesn't waste space
        ax.text(0.02, 0.92, f"#{i}", transform=ax.transAxes,
                fontsize=7, color="#444444", va="top")
        # Tick labels only on outer panels
        if col != 0:
            ax.tick_params(labelleft=False)
        if row != n_rows - 1:
            ax.tick_params(labelbottom=False)

    # Hide unused panels at the end of the last row.
    for j in range(n, n_rows * n_cols):
        row, col = divmod(j, n_cols)
        axes[row, col].set_visible(False)

    fig.supxlabel(xlabel, fontsize=10)
    fig.supylabel(ylabel, fontsize=10)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


def fig_overlay_all(traces, out_path, *, title, xlabel, ylabel):
    """All traces overlaid on one axis (low alpha) + mean trace in black."""
    fig, ax = plt.subplots(figsize=(14, 5))

    # Plot every trace with low alpha to see the shape envelope.
    all_v_aligned = []
    ref_t = None
    for name, t, v in traces:
        ax.plot(t, v, color="#1565C0", linewidth=0.5, alpha=0.25)
        if ref_t is None:
            ref_t = t
        all_v_aligned.append(v)

    # Mean trace, computed over the shortest common prefix.
    min_len = min(len(v) for v in all_v_aligned)
    mean_v  = np.mean([v[:min_len] for v in all_v_aligned], axis=0)
    ax.plot(ref_t[:min_len], mean_v, color="black", linewidth=1.8,
            linestyle="--", label="Mean", zorder=5)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(f"{title}  (n={len(traces)} traces overlaid)")
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(True, linestyle="--", alpha=0.35)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


# ============================================================
# DATASET PROCESSING
# ============================================================

def process_dataset(name, cfg, out_dir):
    print(f"\n{'=' * 50}")
    print(f"Dataset: {name}")
    print(f"Source:  {cfg['pos_folder']}")
    print(f"{'=' * 50}")

    if not cfg["pos_folder"].exists():
        print(f"  SKIP: folder not found")
        return

    traces = load_traces_in_folder(
        cfg["pos_folder"], cfg["crop_samples"], cfg["time_to_hours"]
    )
    if not traces:
        print(f"  ERROR: no traces loaded")
        return
    print(f"  Loaded: {len(traces)} traces")

    # One SVG per trace, in a per-dataset subfolder so the output stays tidy.
    fig_individual(
        traces, out_dir / name, prefix=name,
        title=f"{name}: positive trace",
        xlabel=cfg["x_label"], ylabel=cfg["y_label"],
                )
    fig_overlay_all(
        traces, out_dir / f"{name}_overlay.svg",
        title=f"{name}: all positive traces (overlay)",
        xlabel=cfg["x_label"], ylabel=cfg["y_label"],
                )
    fig_overlay_raw(
        cfg["pos_folder"], out_dir / f"{name}_overlay_raw.svg",
        title=f"{name}: all positive traces (raw x-axis)",
        xlabel="Raw time (from CSV)", ylabel=cfg["y_label"],
                           )


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Output folder: {OUT_DIR}")

    for name, cfg in DATASETS.items():
        process_dataset(name, cfg, OUT_DIR)

    print(f"\nDone. Output: {OUT_DIR}")