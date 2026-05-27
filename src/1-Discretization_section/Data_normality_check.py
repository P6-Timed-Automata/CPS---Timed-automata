"""
qq_plots_all_data.py
====================
Q-Q plots and D'Agostino K² normality tests for the raw values of the three
datasets used in the thesis: synthetic temperature, real room temperature,
and ECG. Used to back the SAX normality discussion in §VII.A.

Output:
    Data/Graphs/Normality/<timestamp>/
        config.txt
        results.json
        qq_<dataset>.png/.svg      (one per dataset)
        qq_combined.png/.svg       (side-by-side comparison)

Switch the synthetic source between clean_train and noisy_train by editing
the DATASETS list below.
"""

import csv
import json
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats


# =============================================================================
# CONFIG
# =============================================================================

ROOT = Path(__file__).resolve().parent.parent.parent

# (display_name, folder_path) — edit paths if your tree differs.
DATASETS = [
    ("synthetic_clean",
     ROOT / "Data" / "synthetic_data" / "clean_train"),
    ("synthetic_noisy",
     ROOT / "Data" / "synthetic_data" / "noisy_train"),
    ("temperature",
     ROOT / "Data" / "3-ExtractInterval" / "1day-experiment" / "A-train"),
    ("ecg",
     ROOT / "Data" / "3-ExtractInterval" / "ecg" / "1beat"),
]


# =============================================================================
# DATA LOADING
# =============================================================================

def load_raw_values(folder_path):
    """
    Read every *.csv in folder_path and return (values, n_traces) where
    values is a flat numpy array of the second-column values and n_traces
    is the number of CSV files loaded. Files use ';' as delimiter with a
    header line; non-numeric rows (the header) are skipped silently.
    """
    values = []
    n_traces = 0
    for file_path in sorted(folder_path.glob("*.csv")):
        file_had_data = False
        with open(file_path, "r") as f:
            reader = csv.reader(f, delimiter=";")
            for row in reader:
                if not row or len(row) < 2:
                    continue
                try:
                    values.append(float(row[1]))
                    file_had_data = True
                except ValueError:
                    continue  # header
        if file_had_data:
            n_traces += 1
    return np.array(values), n_traces


# =============================================================================
# PLOTTING
# =============================================================================

def _draw_qq(ax, dataset_name, raw_values, n_traces, p_value, is_normal):
    """Render one Q-Q plot onto the given axes, with an inset test verdict."""
    stats.probplot(raw_values, dist="norm", plot=ax)
    ax.set_title(f"{dataset_name}  ({n_traces:,} traces)")
    ax.grid(True, linestyle="--", alpha=0.5)

    verdict = "Normal" if is_normal else "NOT Normal"
    textstr = (
        f"D'Agostino K²\n"
        f"p = {p_value:.2e}\n"
        f"→ {verdict}"
    )
    ax.text(
        0.05, 0.95, textstr,
        transform=ax.transAxes,
        fontsize=9,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.85),
    )


def save_single_qq(dataset_name, raw_values, n_traces, p_value, is_normal, out_path_base):
    """One dataset → one figure."""
    fig, ax = plt.subplots(figsize=(6, 6))
    _draw_qq(ax, dataset_name, raw_values, n_traces, p_value, is_normal)
    fig.suptitle("Q-Q Plot: Testing for Normal Distribution", fontsize=11)
    fig.tight_layout()
    fig.savefig(out_path_base.with_suffix(".png"), dpi=150, bbox_inches="tight")
    fig.savefig(out_path_base.with_suffix(".svg"), bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path_base.name}.png / .svg")


def save_combined_qq(results, out_path_base):
    """All datasets → 2-column grid of panels (auto rows)."""
    import math
    n = len(results)
    ncols = 2
    nrows = math.ceil(n / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 6 * nrows))
    # Normalize axes to a flat list regardless of nrows/ncols
    axes = np.atleast_1d(axes).flatten()

    for ax, r in zip(axes, results):
        _draw_qq(ax, r["dataset"], r["raw_values"], r["n_traces"],
                 r["p_value"], r["is_normal"])

    # Hide any unused axes (e.g. 3 datasets in a 2x2 grid)
    for ax in axes[len(results):]:
        ax.set_visible(False)

    fig.suptitle("Q-Q Plots: Raw-Data Normality Across Datasets",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out_path_base.with_suffix(".png"), dpi=150, bbox_inches="tight")
    fig.savefig(out_path_base.with_suffix(".svg"), bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path_base.name}.png / .svg")


# =============================================================================
# CONFIG FILE
# =============================================================================

def _save_config(out_dir):
    lines = [
        "=" * 55,
        "Q-Q Normality Analysis — Raw Data",
        "=" * 55,
        "",
        f"Timestamp : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "--- Datasets ---",
        ]
    for name, folder in DATASETS:
        lines.append(f"  {name:12s}: {folder}")
    lines += ["", f"Output folder: {out_dir}", "", "=" * 55]
    (out_dir / "config.txt").write_text("\n".join(lines))
    print(f"  Saved: config.txt")


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    out_dir = ROOT / "Data" / "Graphs" / "Normality" / timestamp
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output folder: {out_dir}\n")

    _save_config(out_dir)
    print()

    log = {
        "timestamp": timestamp,
        "datasets": [],
    }
    results = []

    for dataset_name, folder in DATASETS:
        print(f"--- {dataset_name} ---")
        print(f"  Loading from: {folder}")

        if not folder.exists():
            print(f"  SKIPPING: folder does not exist\n")
            log["datasets"].append({
                "dataset": dataset_name,
                "folder": str(folder),
                "status": "missing",
            })
            continue

        raw_values, n_traces = load_raw_values(folder)
        if raw_values.size == 0:
            print(f"  SKIPPING: no numeric data loaded\n")
            log["datasets"].append({
                "dataset": dataset_name,
                "folder": str(folder),
                "status": "empty",
            })
            continue

        k2, p_val = stats.normaltest(raw_values)
        is_normal = bool(p_val >= 0.05)

        print(f"  Traces:  {n_traces:,}")
        print(f"  Samples: {raw_values.size:,}")
        print(f"  Mean:    {raw_values.mean():.4f}")
        print(f"  Std:     {raw_values.std():.4f}")
        print(f"  K²:      {k2:.2f}")
        print(f"  p-value: {p_val:.2e}  →  "
              f"{'Normal' if is_normal else 'NOT Normal'}")

        save_single_qq(
            dataset_name, raw_values, n_traces, p_val, is_normal,
            out_dir / f"qq_{dataset_name}",
            )

        results.append({
            "dataset":    dataset_name,
            "raw_values": raw_values,
            "n_traces":   n_traces,
            "p_value":    p_val,
            "is_normal":  is_normal,
        })

        log["datasets"].append({
            "dataset":   dataset_name,
            "folder":    str(folder),
            "status":    "ok",
            "n_traces":  int(n_traces),
            "n_samples": int(raw_values.size),
            "mean":      float(raw_values.mean()),
            "std":       float(raw_values.std()),
            "k2":        float(k2),
            "p_value":   float(p_val),
            "is_normal": is_normal,
        })
        print()

    # Combined comparison figure
    if results:
        print("--- Combined comparison ---")
        save_combined_qq(results, out_dir / "qq_combined")
        print()

    # Persist log
    with open(out_dir / "results.json", "w") as f:
        json.dump(log, f, indent=2)
    print(f"  Saved: results.json")
    print(f"\nDone. Results → {out_dir}")