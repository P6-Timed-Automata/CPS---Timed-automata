"""
exp_51_52_synthetic.py
======================
Experiment 5.1 — Train on CLEAN synthetic data, evaluate on all four
                  negative modes separately.
Experiment 5.2 — Same pipeline but train on NOISY synthetic data.

Requires: run generate_all_data.py first.

Output (timestamped folder under Graphs/exp_51_52/):
  results.json               — all metrics
  comparison.png             — precision / recall / F1 per method per condition
  per_mode_heatmap.png       — rejection rate per method x negative mode
  table_overall.csv          — overall P / R / F1 / states / edges
  table_per_mode_clean.csv   — per-mode rejection rates, clean training
  table_per_mode_noisy.csv   — per-mode rejection rates, noisy training
  tables.txt                 — all tables formatted for easy copy-paste
"""

import csv
import json
import sys
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from Generate_data import load_all_data
from Generators import NEG_MODE_NAMES
from Pipeline import run_pipeline


# =============================================================================
# CONFIG
# =============================================================================

TAG_K = 2

METHODS = [
    ("naive",   {"bins": 15}),
    ("sax",     {"w": 288, "bins": 15}),
    ("persist", {"bins": 15}),
]


# =============================================================================
# HELPERS
# =============================================================================

def _run_condition(label, train_traces, test_pos, test_neg, neg_modes,
                   out_dir, methods, tag_k):
    results = []
    for method, params in methods:
        print(f"  [{label}] {method} {params} ...", flush=True)
        ta_folder = str(out_dir / "ta_images" / label / method)
        result = run_pipeline(
            method=method, params=params,
            train_traces=train_traces,
            test_pos_traces=test_pos,
            test_neg_traces=test_neg,
            tag_k=tag_k,
            neg_modes=neg_modes,
            save_ta_path=ta_folder,
            ta_title=f"{label}_{method}",
        )
        ov = result["overall"]
        print(f"    P={ov['precision']:.3f} R={ov['recall']:.3f} "
              f"F1={ov['f1']:.3f} states={result['n_states']}")
        results.append({
            "condition": label,
            "method":    method,
            "params":    params,
            "n_states":  result["n_states"],
            "n_edges":   result["n_edges"],
            "overall":   {k: v for k, v in ov.items()
                          if k not in ("save_path", "run_id")},
            "per_mode":  result["per_mode"],
        })
    return results


# =============================================================================
# TABLE GENERATION
# =============================================================================

def _fmt_row(values, widths):
    """Format a single table row with fixed column widths."""
    return "  ".join(str(v).ljust(w) for v, w in zip(values, widths))


def _make_table(headers, rows):
    """
    Build a plain-text table string.
    headers : list of str
    rows    : list of lists
    Returns : (table_str, col_widths)
    """
    all_rows  = [headers] + [[str(c) for c in r] for r in rows]
    widths    = [max(len(r[i]) for r in all_rows)
                 for i in range(len(headers))]
    separator = "  ".join("-" * w for w in widths)
    lines     = [_fmt_row(headers, widths), separator]
    for row in rows:
        lines.append(_fmt_row([str(c) for c in row], widths))
    return "\n".join(lines), widths


def build_tables(all_results):
    """
    Build three tables from the results list.
    Returns a dict with keys: overall, per_mode_clean, per_mode_noisy.
    Each value is a dict with keys: headers, rows, title.
    """
    mode_names = list(NEG_MODE_NAMES.values())

    # ------------------------------------------------------------------
    # Table 1 — Overall metrics
    # ------------------------------------------------------------------
    t1_headers = ["Condition", "Method", "Precision", "Recall",
                  "F1", "States", "Edges"]
    t1_rows = []
    for r in all_results:
        ov = r["overall"]
        t1_rows.append([
            r["condition"],
            r["method"],
            f"{ov['precision']:.3f}",
            f"{ov['recall']:.3f}",
            f"{ov['f1']:.3f}",
            r["n_states"],
            r["n_edges"],
        ])

    # ------------------------------------------------------------------
    # Tables 2 & 3 — Per-mode rejection rate (%) per condition
    # ------------------------------------------------------------------
    def per_mode_rows(condition):
        rows = []
        for r in all_results:
            if r["condition"] != condition:
                continue
            row = [r["method"]]
            for mode in mode_names:
                pct = r["per_mode"].get(mode, {}).get("rejection", 0.0)
                row.append(f"{pct:.1f}%")
            rows.append(row)
        return rows

    t2_headers = ["Method"] + [m.capitalize() for m in mode_names]
    t3_headers = t2_headers

    return {
        "overall": {
            "title":   "Table 1 — Overall metrics (Exp 5.1 clean / Exp 5.2 noisy)",
            "headers": t1_headers,
            "rows":    t1_rows,
        },
        "per_mode_clean": {
            "title":   "Table 2 — Per-mode rejection rate (%) — clean training (Exp 5.1)",
            "headers": t2_headers,
            "rows":    per_mode_rows("clean"),
        },
        "per_mode_noisy": {
            "title":   "Table 3 — Per-mode rejection rate (%) — noisy training (Exp 5.2)",
            "headers": t3_headers,
            "rows":    per_mode_rows("noisy"),
        },
    }


def print_tables(tables):
    """Print all tables to stdout."""
    for key, tbl in tables.items():
        print(f"\n{tbl['title']}")
        table_str, _ = _make_table(tbl["headers"], tbl["rows"])
        print(table_str)


def save_tables(tables, out_dir):
    """
    Save each table as a CSV and all three as a combined tables.txt.
    """
    filename_map = {
        "overall":         "table_overall.csv",
        "per_mode_clean":  "table_per_mode_clean.csv",
        "per_mode_noisy":  "table_per_mode_noisy.csv",
    }
    txt_lines = []

    for key, tbl in tables.items():
        # CSV
        csv_path = out_dir / filename_map[key]
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(tbl["headers"])
            writer.writerows(tbl["rows"])
        print(f"  Saved: {csv_path}")

        # Collect for combined txt
        table_str, _ = _make_table(tbl["headers"], tbl["rows"])
        txt_lines.append(tbl["title"])
        txt_lines.append(table_str)
        txt_lines.append("")

    # Combined plain-text file
    txt_path = out_dir / "tables.txt"
    with open(txt_path, "w") as f:
        f.write("\n".join(txt_lines))
    print(f"  Saved: {txt_path}")


# =============================================================================
# PLOTTING
# =============================================================================

def plot_comparison(all_results, out_path):
    methods    = [r["method"] for r in all_results if r["condition"] == "clean"]
    conditions = ["clean", "noisy"]
    colors     = {"clean": "steelblue", "noisy": "darkorange"}
    metrics    = [("precision", "Precision"), ("recall", "Recall"), ("f1", "F1")]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    x     = np.arange(len(methods))
    width = 0.35

    for ax, (key, title) in zip(axes, metrics):
        for ci, cond in enumerate(conditions):
            vals = [r["overall"][key] for r in all_results
                    if r["condition"] == cond]
            bars = ax.bar(x + (ci - 0.5) * width, vals, width,
                          label=cond, color=colors[cond], alpha=0.85)
            for bar, v in zip(bars, vals):
                ax.text(bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + 0.01, f"{v:.2f}",
                        ha="center", va="bottom", fontsize=8)
        ax.set_xticks(x)
        ax.set_xticklabels(methods)
        ax.set_ylim(0, 1.2)
        ax.set_title(title)
        ax.legend(fontsize=9)
        ax.grid(True, linestyle="--", alpha=0.4, axis="y")

    fig.suptitle("Exp 5.1 / 5.2 — Classifier metrics: clean vs noisy training",
                 fontsize=12)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


def plot_per_mode_heatmap(all_results, out_path):
    mode_names = list(NEG_MODE_NAMES.values())
    methods    = [r["method"] for r in all_results if r["condition"] == "clean"]
    conditions = ["clean", "noisy"]

    fig, axes = plt.subplots(1, 2, figsize=(13, 4))

    for ax, cond in zip(axes, conditions):
        data = np.zeros((len(methods), len(mode_names)))
        for i, method in enumerate(methods):
            r = next(x for x in all_results
                     if x["condition"] == cond and x["method"] == method)
            for j, mode in enumerate(mode_names):
                data[i, j] = r["per_mode"].get(mode, {}).get("rejection", 0.0)

        im = ax.imshow(data, vmin=0, vmax=100, cmap="RdYlGn", aspect="auto")
        ax.set_xticks(range(len(mode_names)))
        ax.set_xticklabels(mode_names)
        ax.set_yticks(range(len(methods)))
        ax.set_yticklabels(methods)
        ax.set_title(f"Rejection rate (%) — {cond} training")

        for i in range(len(methods)):
            for j in range(len(mode_names)):
                ax.text(j, i, f"{data[i, j]:.0f}",
                        ha="center", va="center", fontsize=10,
                        color="black" if 20 < data[i, j] < 80 else "white")
        plt.colorbar(im, ax=ax, label="%")

    fig.suptitle("Anomaly rejection rate per mode (higher = better)", fontsize=11)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    out_dir   = ROOT / "Data" / "Graphs" / "Metrics_clean" / timestamp
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output folder: {out_dir}\n")

    data = load_all_data()

    log = {
        "timestamp": timestamp,
        "tag_k":     TAG_K,
        "n_train":   len(data["clean_train"]),
        "n_test":    len(data["clean_test"]),
        "n_neg":     len(data["neg_traces"]),
        "results":   [],
    }

    # --- Exp 5.1 — clean training --------------------------------------------
    print("\n=== Experiment 5.1 — Clean training ===")
    log["results"] += _run_condition(
        "clean",
        data["clean_train"], data["clean_test"],
        data["neg_traces"],  data["neg_modes"],
        out_dir, METHODS, TAG_K,
    )

    # --- Exp 5.2 — noisy training --------------------------------------------
    print("\n=== Experiment 5.2 — Noisy training ===")
    log["results"] += _run_condition(
        "noisy",
        data["noisy_train"], data["noisy_test"],
        data["neg_traces"],  data["neg_modes"],
        out_dir, METHODS, TAG_K,
    )

    # --- Tables --------------------------------------------------------------
    print("\n=== Tables ===")
    tables = build_tables(log["results"])
    print_tables(tables)
    print()
    save_tables(tables, out_dir)

    # --- Plots ---------------------------------------------------------------
    print("\n=== Plots ===")
    with open(out_dir / "results.json", "w") as f:
        json.dump(log, f, indent=2)
    print(f"  Saved: {out_dir / 'results.json'}")

    plot_comparison(log["results"], out_dir / "comparison.png")
    plot_per_mode_heatmap(log["results"], out_dir / "per_mode_heatmap.png")

    print(f"\nDone. Results -> {out_dir}")