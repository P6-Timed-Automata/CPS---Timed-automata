"""
exp_54_real_temp.py
===================
Experiment 5.4 — Real temperature data.

Trains on normalised real temperature traces, evaluates against the same
four synthetic negative modes used in 5.1/5.2. Comparing these results
to 5.1/5.2 tells you whether synthetic performance transfers to real data.

Configure the paths at the top of MAIN before running.

Output (timestamped folder under Graphs/exp_54/):
  results.json
  comparison.png        — P / R / F1 per method
  per_mode_heatmap.png  — rejection rate per method × negative mode
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

from Generators import generate_negative_set, load_traces, NEG_MODE_NAMES
from Pipeline import run_pipeline


# =============================================================================
# CONFIG
# =============================================================================

TAG_K = 2
N_NEG = 60   # should be divisible by 4 for equal mode representation

METHODS = [
    ("naive",   {"bins": 5}),
    ("sax",     {"w": 288, "bins": 5}),
    ("persist", {"bins": 10}),
]

METHOD_COLORS = {
    "naive":   "steelblue",
    "sax":     "darkorange",
    "persist": "seagreen",
}


# =============================================================================
# PLOTTING  (same helpers as 5.1/5.2)
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
    fig.suptitle("Exp 5.4 — Real temperature data", fontsize=12)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


def plot_per_mode_heatmap(results, out_path):
    mode_names = list(NEG_MODE_NAMES.values())
    methods    = [r["method"] for r in results]
    data       = np.zeros((len(methods), len(mode_names)))

    for i, r in enumerate(results):
        for j, mode in enumerate(mode_names):
            data[i, j] = r["per_mode"].get(mode, {}).get("rejection", 0.0)

    fig, ax = plt.subplots(figsize=(8, 4))
    im = ax.imshow(data, vmin=0, vmax=100, cmap="RdYlGn", aspect="auto")
    ax.set_xticks(range(len(mode_names)))
    ax.set_xticklabels(mode_names)
    ax.set_yticks(range(len(methods)))
    ax.set_yticklabels(methods)
    ax.set_title("Rejection rate (%) per anomaly mode — real temperature data")
    for i in range(len(methods)):
        for j in range(len(mode_names)):
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
    out_dir   = BASE_DIR / "Data" / "Graphs" / "exp_54" / timestamp
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output folder: {out_dir}\n")

    # -------------------------------------------------------------------------
    # Configure these paths before running
    # -------------------------------------------------------------------------
    TRAIN_FOLDER    = BASE_DIR / "Data" / "3-ExtractInterval" / "1day-experiment" / "roomA-train"
    TEST_POS_FOLDER = BASE_DIR / "Data" / "3-ExtractInterval" / "1day-experiment" / "roomA-test" / "positive"
    # If you don't have a separate positive test folder, use a held-out split:
    # set TEST_POS_FOLDER = None and configure TRAIN_SPLIT below
    TRAIN_SPLIT = 0.8   # fraction used for training if no separate test folder
    # -------------------------------------------------------------------------

    # Load training traces
    print("Loading real temperature traces...")
    if TEST_POS_FOLDER is not None and TEST_POS_FOLDER.exists():
        train_traces = load_traces(TRAIN_FOLDER)
        test_pos     = load_traces(TEST_POS_FOLDER)
    else:
        all_traces   = load_traces(TRAIN_FOLDER)
        split        = int(len(all_traces) * TRAIN_SPLIT)
        train_traces = all_traces[:split]
        test_pos     = all_traces[split:]

    print(f"  Train: {len(train_traces)} traces")
    print(f"  Test positive: {len(test_pos)} traces")

    if not train_traces:
        print("ERROR: No training traces found. Check TRAIN_FOLDER path.")
        sys.exit(1)

    # Generate negative traces (synthetic, same as 5.1/5.2)
    neg_traces, neg_modes = generate_negative_set(n_traces=N_NEG, seed=99)
    print(f"  Negatives: {len(neg_traces)} ({len(set(neg_modes))} modes)\n")

    log = {
        "timestamp":    timestamp,
        "tag_k":        TAG_K,
        "train_folder": str(TRAIN_FOLDER),
        "n_train":      len(train_traces),
        "n_test_pos":   len(test_pos),
        "n_neg":        len(neg_traces),
        "results":      [],
    }

    all_results = []
    for method, params in METHODS:
        print(f"[{method}] {params} ...", flush=True)
        ta_folder = str(out_dir / "ta_images" / method)
        result = run_pipeline(
            method=method, params=params,
            train_traces=train_traces,
            test_pos_traces=test_pos,
            test_neg_traces=neg_traces,
            tag_k=TAG_K,
            neg_modes=neg_modes,
            save_ta_path=ta_folder,
            ta_title=f"exp54_{method}",
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

    plot_comparison(all_results, out_dir / "comparison.png")
    plot_per_mode_heatmap(all_results, out_dir / "per_mode_heatmap.png")

    print(f"\nDone. Results → {out_dir}")