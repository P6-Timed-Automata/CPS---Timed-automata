"""
exp_55_ecg.py
=============
Experiment 5.5 — ECG data.

Same pipeline as Exp 5.4 (real data) but on ECG traces.
ECG signals are highly regular, so this should be your strongest
result and clearest demonstration of the pipeline end-to-end.

Before running:
  1. Set DATA_FOLDER to your ECG CSV directory.
  2. ECG CSVs must match the semicolon-delimited format:
       time_s;value
  3. Set NEGATIVE_MODE to choose how negatives are generated —
     either load from a folder of known anomalous ECG or generate
     synthetic negatives with the modes below.
  4. Adjust METHODS / params if ECG dynamics need different settings
     (e.g. smaller w for SAX since ECG cycles are much shorter than
     24h temperature cycles).

Output (timestamped folder under Graphs/exp_55/):
  results.json
  comparison.png
  per_mode_heatmap.png
"""

import json
import sys
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from shared.generators import load_traces, NEG_MODE_NAMES
from shared.pipeline import run_pipeline


# =============================================================================
# CONFIG — adjust before running
# =============================================================================

TAG_K = 2

# ---- Data paths -------------------------------------------------------------
DATA_FOLDER      = ROOT / "Data" / "ecg"          # folder with positive CSVs
NEG_DATA_FOLDER  = ROOT / "Data" / "ecg_negative" # set to None to use synthetic

TRAIN_SPLIT = 0.8   # fraction of DATA_FOLDER used for training

# ---- Negative mode ----------------------------------------------------------
# "folder"    : load negatives from NEG_DATA_FOLDER
# "synthetic" : generate synthetic negatives (same 4 modes as temperature)
NEGATIVE_MODE = "folder"
N_SYNTHETIC_NEG = 60

# ---- Methods ----------------------------------------------------------------
# SAX w should be roughly n_samples_per_cycle for ECG
# e.g. if ECG sampled at 360 Hz and one heartbeat = 1s → w ≈ 360
# Adjust based on your actual ECG sample rate and epoch length
METHODS = [
    ("naive",   {"bins": 5}),
    ("sax",     {"w": 50, "bins": 5}),   # <-- tune w for ECG
    ("persist", {"bins": 10}),
]

METHOD_COLORS = {
    "naive":   "steelblue",
    "sax":     "darkorange",
    "persist": "seagreen",
}


# =============================================================================
# PLOTTING  (identical helpers to exp 5.4)
# =============================================================================

def plot_comparison(results, out_path):
    methods = [r["method"] for r in results]
    metrics = [("precision", "Precision"), ("recall", "Recall"), ("f1", "F1")]
    x, width = np.arange(len(methods)), 0.5

    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    for ax, (key, title) in zip(axes, metrics):
        vals = [r["overall"][key] for r in results]
        bars = ax.bar(x, vals, width,
                      color=[METHOD_COLORS.get(m, "gray") for m in methods],
                      alpha=0.85)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.01, f"{v:.2f}",
                    ha="center", va="bottom", fontsize=9)
        ax.set_xticks(x)
        ax.set_xticklabels(methods)
        ax.set_ylim(0, 1.2)
        ax.set_title(title)
        ax.grid(True, linestyle="--", alpha=0.4, axis="y")
    fig.suptitle("Exp 5.5 — ECG data", fontsize=12)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


def plot_per_mode_heatmap(results, neg_mode_names, out_path):
    methods = [r["method"] for r in results]
    data    = np.zeros((len(methods), len(neg_mode_names)))

    for i, r in enumerate(results):
        for j, mode in enumerate(neg_mode_names):
            data[i, j] = r["per_mode"].get(mode, {}).get("rejection", 0.0)

    fig, ax = plt.subplots(figsize=(max(8, len(neg_mode_names) * 1.5), 4))
    im = ax.imshow(data, vmin=0, vmax=100, cmap="RdYlGn", aspect="auto")
    ax.set_xticks(range(len(neg_mode_names)))
    ax.set_xticklabels(neg_mode_names)
    ax.set_yticks(range(len(methods)))
    ax.set_yticklabels(methods)
    ax.set_title("Rejection rate (%) per anomaly mode — ECG")
    for i in range(len(methods)):
        for j in range(len(neg_mode_names)):
            ax.text(j, i, f"{data[i, j]:.0f}",
                    ha="center", va="center", fontsize=11,
                    color="black" if 20 < data[i, j] < 80 else "white")
    plt.colorbar(im, ax=ax, label="%")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    BASE_DIR  = ROOT
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    out_dir   = BASE_DIR / "Data" / "Graphs" / "exp_55" / timestamp
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output folder: {out_dir}\n")

    # --- Load ECG traces -----------------------------------------------------
    print("Loading ECG traces...")
    if not DATA_FOLDER.exists():
        print(f"ERROR: DATA_FOLDER not found: {DATA_FOLDER}")
        print("Set DATA_FOLDER at the top of this script and re-run.")
        sys.exit(1)

    all_traces = load_traces(DATA_FOLDER)
    split      = int(len(all_traces) * TRAIN_SPLIT)
    train_traces = all_traces[:split]
    test_pos     = all_traces[split:]
    print(f"  Train: {len(train_traces)} | Test positive: {len(test_pos)}")

    # --- Negatives -----------------------------------------------------------
    if NEGATIVE_MODE == "folder" and NEG_DATA_FOLDER is not None \
            and NEG_DATA_FOLDER.exists():
        neg_traces = load_traces(NEG_DATA_FOLDER)
        # Assign all to a single "anomaly" mode for heatmap compatibility
        neg_modes     = [0] * len(neg_traces)
        neg_mode_names = ["anomaly"]
        print(f"  Negatives: {len(neg_traces)} loaded from {NEG_DATA_FOLDER}")
    else:
        print("  Generating synthetic negatives (4 modes)...")
        from Generators import generate_negative_set
        neg_traces, neg_modes = generate_negative_set(
            n_traces=N_SYNTHETIC_NEG, seed=99
        )
        neg_mode_names = list(NEG_MODE_NAMES.values())
        print(f"  Negatives: {len(neg_traces)}")

    if not train_traces:
        print("ERROR: No training traces loaded.")
        sys.exit(1)

    # --- Run pipeline --------------------------------------------------------
    log = {
        "timestamp":   timestamp,
        "tag_k":       TAG_K,
        "data_folder": str(DATA_FOLDER),
        "n_train":     len(train_traces),
        "n_test_pos":  len(test_pos),
        "n_neg":       len(neg_traces),
        "results":     [],
    }

    all_results = []
    for method, params in METHODS:
        print(f"\n[{method}] {params} ...", flush=True)
        ta_folder = str(out_dir / "ta_images" / method)
        result = run_pipeline(
            method=method, params=params,
            train_traces=train_traces,
            test_pos_traces=test_pos,
            test_neg_traces=neg_traces,
            tag_k=TAG_K,
            neg_modes=neg_modes,
            save_ta_path=ta_folder,
            ta_title=f"exp55_{method}",
        )
        ov = result["overall"]
        print(f"  P={ov['precision']:.3f}  R={ov['recall']:.3f}  "
              f"F1={ov['f1']:.3f}  states={result['n_states']}")

        entry = {
            "method":   method,
            "params":   params,
            "n_states": result["n_states"],
            "n_edges":  result["n_edges"],
            "overall":  {k: v for k, v in ov.items()
                         if not isinstance(v, list)},
            "per_mode": result["per_mode"],
        }
        all_results.append(entry)
        log["results"].append(entry)

        with open(out_dir / "results.json", "w") as f:
            json.dump(log, f, indent=2)

    # --- Plots ---------------------------------------------------------------
    plot_comparison(all_results, out_dir / "comparison.png")
    plot_per_mode_heatmap(all_results, neg_mode_names,
                          out_dir / "per_mode_heatmap.png")

    print(f"\nDone. Results → {out_dir}")