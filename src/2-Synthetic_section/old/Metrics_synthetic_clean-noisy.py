"""
exp_51_52_synthetic.py
======================
Experiment 5.1 — Train on CLEAN synthetic data, evaluate on all four
                  negative modes separately.
Experiment 5.2 — Same pipeline but train on NOISY synthetic data.

Now sweeps multiple parameter settings per method to show robustness across
the parameter range, not just a single configuration.

Requires: run generate_all_data.py first.

Output (timestamped folder under Graphs/Metrics_5_1_5_2/):
  results.json                  — all metrics for all variants
  config.txt                    — full configuration used
  comparison_<metric>.png       — one figure per metric (P, R, F1)
  per_mode_heatmap.png          — rejection rate per variant x negative mode
  robustness.png                — F1 spread per method across variants
  table_overall.csv             — overall P / R / F1 / states per variant
  table_per_mode_clean.csv      — per-mode rejection rates, clean training
  table_per_mode_noisy.csv      — per-mode rejection rates, noisy training
  table_robustness.csv          — per-method median F1 + (min, max) across variants
  tables.txt                    — all tables formatted for easy copy-paste
  table_*.png                   — rendered table images
"""

import csv
import json
import subprocess
import sys
from collections import defaultdict
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

# Parameter sweep per method. Pick the variants you actually want to compare —
# typically a few values around what the benchmark identified as best.
METHOD_VARIANTS = {
    "naive": [
        {"bins": 5},
        {"bins": 10},
        {"bins": 15},
    ],
    "sax": [
        {"w": 24,  "bins": 5},
        {"w": 24,  "bins": 10},
        {"w": 24,  "bins": 15},
        {"w": 48,  "bins": 5},
        {"w": 48,  "bins": 10},
        {"w": 48,  "bins": 15},
        {"w": 144,  "bins": 5},
        {"w": 144,  "bins": 10},
        {"w": 144,  "bins": 15},
        
    ],
    "persist": [
        {"bins": 5},
        {"bins": 10},
        {"bins": 15},
    ],
}


# =============================================================================
# HELPERS
# =============================================================================

def _variant_label(method, params):
    """A short, filesystem-safe label like 'naive_bins=10' or 'sax_w=48_bins=15'."""
    param_str = "_".join(f"{k}={v}" for k, v in params.items())
    return f"{method}_{param_str}"


def _git_hash():
    """Return current git HEAD hash for reproducibility, or 'unknown'."""
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT
        ).decode().strip()
    except Exception:
        return "unknown"


def _run_condition(label, train_traces, test_pos, test_neg, neg_modes,
                   out_dir, method_variants, tag_k):
    """Run all (method, params) variants for one training condition."""
    results = []
    for method, variants in method_variants.items():
        for params in variants:
            variant_label = _variant_label(method, params)
            print(f"  [{label}] {variant_label} ...", flush=True)
            ta_folder = str(out_dir / "ta_images" / label / variant_label)

            result = run_pipeline(
                method=method, params=params,
                train_traces=train_traces,
                test_pos_traces=test_pos,
                test_neg_traces=test_neg,
                tag_k=tag_k,
                neg_modes=neg_modes,
                save_ta_path=ta_folder,
                ta_title=f"{label}_{variant_label}",
            )
            ov = result["overall"]
            print(f"    P={ov['precision']:.3f} R={ov['recall']:.3f} "
                  f"F1={ov['f1']:.3f} states={result['n_states']}")
            results.append({
                "condition":     label,
                "method":        method,
                "params":        params,
                "variant_label": variant_label,
                "n_states":      result["n_states"],
                "n_edges":       result["n_edges"],
                "overall":       {k: v for k, v in ov.items()
                                  if k not in ("save_path", "run_id")},
                "per_mode":      result["per_mode"],
            })
    return results


# =============================================================================
# TABLE GENERATION
# =============================================================================

def _fmt_row(values, widths):
    return "  ".join(str(v).ljust(w) for v, w in zip(values, widths))


def _make_table(headers, rows):
    all_rows = [headers] + [[str(c) for c in r] for r in rows]
    widths = [max(len(r[i]) for r in all_rows) for i in range(len(headers))]
    separator = "  ".join("-" * w for w in widths)
    lines = [_fmt_row(headers, widths), separator]
    for row in rows:
        lines.append(_fmt_row([str(c) for c in row], widths))
    return "\n".join(lines), widths


def build_tables(all_results):
    """Build all tables. Returns dict keyed by table name."""
    mode_names = list(NEG_MODE_NAMES.values())

    # ------------------------------------------------------------------
    # Table 1 — Overall metrics, one row per variant per condition
    # ------------------------------------------------------------------
    t1_headers = ["Condition", "Method", "Params", "Precision", "Recall",
                  "F1", "States", "Edges"]
    t1_rows = []
    for r in all_results:
        ov = r["overall"]
        param_str = ", ".join(f"{k}={v}" for k, v in r["params"].items())
        t1_rows.append([
            r["condition"],
            r["method"],
            param_str,
            f"{ov['precision']:.3f}",
            f"{ov['recall']:.3f}",
            f"{ov['f1']:.3f}",
            r["n_states"],
            r["n_edges"],
        ])

    # ------------------------------------------------------------------
    # Tables 2 & 3 — Per-mode rejection rate per variant per condition
    # ------------------------------------------------------------------
    def per_mode_rows(condition):
        rows = []
        for r in all_results:
            if r["condition"] != condition:
                continue
            row = [r["variant_label"]]
            for mode in mode_names:
                pct = r["per_mode"].get(mode, {}).get("rejection", 0.0)
                row.append(f"{pct:.1f}%")
            rows.append(row)
        return rows

    t23_headers = ["Variant"] + [m.capitalize() for m in mode_names]

    # ------------------------------------------------------------------
    # Table 4 — Robustness summary: per method × condition, median F1 +
    #          (min, max) across variants
    # ------------------------------------------------------------------
    t4_headers = ["Condition", "Method", "F1 median", "F1 range",
                  "Best variant", "Worst variant"]
    t4_rows = []
    # Group by (condition, method)
    grouped = defaultdict(list)
    for r in all_results:
        grouped[(r["condition"], r["method"])].append(r)

    for (cond, method), variants in sorted(grouped.items()):
        f1s = [v["overall"]["f1"] for v in variants]
        labels = [v["variant_label"] for v in variants]
        if not f1s:
            continue
        f1_median = float(np.median(f1s))
        f1_min, f1_max = float(min(f1s)), float(max(f1s))
        best_idx = int(np.argmax(f1s))
        worst_idx = int(np.argmin(f1s))
        t4_rows.append([
            cond,
            method,
            f"{f1_median:.3f}",
            f"{f1_min:.3f}-{f1_max:.3f}",
            labels[best_idx],
            labels[worst_idx],
        ])

    return {
        "overall": {
            "title":   "Overall metrics (one row per variant per condition)",
            "headers": t1_headers,
            "rows":    t1_rows,
        },
        "per_mode_clean": {
            "title":   "Per-mode rejection rate (%) — clean training",
            "headers": t23_headers,
            "rows":    per_mode_rows("clean"),
        },
        "per_mode_noisy": {
            "title":   "Per-mode rejection rate (%) — noisy training",
            "headers": t23_headers,
            "rows":    per_mode_rows("noisy"),
        },
        "robustness": {
            "title":   "Robustness: F1 spread per method across variants",
            "headers": t4_headers,
            "rows":    t4_rows,
        },
    }


def print_tables(tables):
    for tbl in tables.values():
        print(f"\n{tbl['title']}")
        table_str, _ = _make_table(tbl["headers"], tbl["rows"])
        print(table_str)


def save_tables(tables, out_dir):
    filename_map = {
        "overall":        "table_overall.csv",
        "per_mode_clean": "table_per_mode_clean.csv",
        "per_mode_noisy": "table_per_mode_noisy.csv",
        "robustness":     "table_robustness.csv",
    }
    txt_lines = []
    for key, tbl in tables.items():
        csv_path = out_dir / filename_map[key]
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(tbl["headers"])
            writer.writerows(tbl["rows"])
        print(f"  Saved: {csv_path}")

        table_str, _ = _make_table(tbl["headers"], tbl["rows"])
        txt_lines += [tbl["title"], table_str, ""]

    txt_path = out_dir / "tables.txt"
    txt_path.write_text("\n".join(txt_lines))
    print(f"  Saved: {txt_path}")


def save_config(out_dir, tag_k, method_variants, data):
    lines = [
        "=" * 55,
        "Run configuration",
        "=" * 55,
        "",
        f"Timestamp   : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"Git hash    : {_git_hash()}",
        f"TAG k-future: {tag_k}",
        "",
        "--- Method variants ---",
        ]
    total_variants = 0
    for method, variants in method_variants.items():
        lines.append(f"  {method}:")
        for params in variants:
            param_str = ", ".join(f"{k}={v}" for k, v in params.items())
            lines.append(f"    - {param_str}")
            total_variants += 1
    lines.append(f"  Total variants per condition: {total_variants}")

    lines += [
        "",
        "--- Dataset sizes ---",
        f"  clean_train : {len(data['clean_train'])} traces",
        f"  clean_test  : {len(data['clean_test'])} traces",
        f"  noisy_train : {len(data['noisy_train'])} traces",
        f"  noisy_test  : {len(data['noisy_test'])} traces",
        f"  negatives   : {len(data['neg_traces'])} traces "
        f"({len(set(data['neg_modes']))} modes)",
        "",
        "--- Negative modes ---",
    ]
    from collections import Counter
    mode_counts = Counter(data["neg_modes"])
    for mode_int, count in sorted(mode_counts.items()):
        lines.append(f"  {NEG_MODE_NAMES[mode_int]:10s}: {count} traces")

    lines += ["", "--- Output folder ---", f"  {out_dir}", "", "=" * 55]

    config_path = out_dir / "config.txt"
    config_path.write_text("\n".join(lines))
    print(f"  Saved: {config_path}")


def save_tables_as_images(tables, out_dir):
    """Render each table as a PNG. Per-mode tables get color coding."""
    filename_map = {
        "overall":        "table_overall.png",
        "per_mode_clean": "table_per_mode_clean.png",
        "per_mode_noisy": "table_per_mode_noisy.png",
        "robustness":     "table_robustness.png",
    }

    HEADER_COLOR = "#2c3e50"
    ROW_EVEN = "#f2f4f6"
    ROW_ODD = "#ffffff"
    COLORED_BACKGROUNDS = {"#27ae60", "#e74c3c", "#f39c12", HEADER_COLOR}

    def _cell_color(value_str, is_header, row_idx, is_mode_table):
        if is_header:
            return HEADER_COLOR
        if is_mode_table:
            try:
                pct = float(value_str.strip("%"))
                if pct >= 80:
                    return "#27ae60"
                elif pct >= 50:
                    return "#f39c12"
                else:
                    return "#e74c3c"
            except ValueError:
                pass
        return ROW_EVEN if row_idx % 2 == 0 else ROW_ODD

    for key, tbl in tables.items():
        headers = tbl["headers"]
        rows = tbl["rows"]
        title = tbl["title"]
        is_mode_table = "per_mode" in key

        n_cols = len(headers)
        n_rows = len(rows) + 1

        fig_w = max(8, n_cols * 1.6)
        fig_h = max(2, n_rows * 0.45 + 0.8)

        fig, ax = plt.subplots(figsize=(fig_w, fig_h))
        ax.axis("off")

        cell_text = [headers] + [[str(c) for c in r] for r in rows]
        cell_colors = []
        for ri, row in enumerate(cell_text):
            is_header = (ri == 0)
            row_colors = [
                _cell_color(val, is_header, ri - 1, is_mode_table)
                for val in row
            ]
            cell_colors.append(row_colors)

        tbl_obj = ax.table(
            cellText=cell_text, cellColours=cell_colors,
            cellLoc="center", loc="center",
        )
        tbl_obj.auto_set_font_size(False)
        tbl_obj.set_fontsize(9)
        tbl_obj.scale(1, 1.4)

        for (ri, ci), cell in tbl_obj.get_celld().items():
            cell.set_edgecolor("white")
            cell.set_linewidth(1.5)
            bg = cell_colors[ri][ci]
            if bg in COLORED_BACKGROUNDS:
                cell.set_text_props(
                    color="white",
                    fontweight="bold" if ri == 0 else "normal",
                )
            else:
                cell.set_text_props(color="#2c3e50")

        fig.suptitle(title, fontsize=11, fontweight="bold",
                     y=0.97, color="#2c3e50")

        out_path = out_dir / filename_map[key]
        fig.savefig(out_path, dpi=200, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        print(f"  Saved: {out_path}")


# =============================================================================
# PLOTTING
# =============================================================================

def plot_metric_comparison(all_results, metric, metric_title, out_path):
    """
    Bar chart of one metric (precision/recall/f1) per variant per condition.
    One subplot per method; one bar per (variant, condition).
    """
    methods = sorted(set(r["method"] for r in all_results))
    conditions = ["clean", "noisy"]
    colors = {"clean": "steelblue", "noisy": "darkorange"}

    fig, axes = plt.subplots(1, len(methods), figsize=(5 * len(methods), 5),
                             squeeze=False)
    axes = axes[0]

    for ax, method in zip(axes, methods):
        method_results = [r for r in all_results if r["method"] == method]
        # Order variants consistently within method
        variant_labels = sorted(set(r["variant_label"] for r in method_results))
        x = np.arange(len(variant_labels))
        width = 0.35

        for ci, cond in enumerate(conditions):
            vals = []
            for vlabel in variant_labels:
                matching = [r for r in method_results
                            if r["variant_label"] == vlabel
                            and r["condition"] == cond]
                vals.append(matching[0]["overall"][metric] if matching else 0)

            bars = ax.bar(x + (ci - 0.5) * width, vals, width,
                          label=cond, color=colors[cond], alpha=0.85)
            for bar, v in zip(bars, vals):
                ax.text(bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + 0.01, f"{v:.2f}",
                        ha="center", va="bottom", fontsize=7)

        ax.set_xticks(x)
        # Strip the method prefix from variant labels for readability
        short_labels = [v.replace(f"{method}_", "") for v in variant_labels]
        ax.set_xticklabels(short_labels, rotation=30, ha="right", fontsize=8)
        ax.set_ylim(0, 1.2)
        ax.set_title(method)
        ax.legend(fontsize=9)
        ax.grid(True, linestyle="--", alpha=0.4, axis="y")

    fig.suptitle(f"{metric_title} per variant — clean vs. noisy training",
                 fontsize=12)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


def plot_robustness(all_results, out_path):
    """
    Strip plot of F1 per variant, grouped by (method, condition).
    Visualizes how much F1 varies across parameter choices.
    """
    methods = sorted(set(r["method"] for r in all_results))
    conditions = ["clean", "noisy"]
    colors = {"clean": "steelblue", "noisy": "darkorange"}

    fig, ax = plt.subplots(figsize=(10, 6))

    x_pos = 0
    x_ticks = []
    x_labels = []

    for method in methods:
        for cond in conditions:
            f1s = [r["overall"]["f1"] for r in all_results
                   if r["method"] == method and r["condition"] == cond]
            if not f1s:
                continue
            # Jitter horizontally to avoid overlap
            jitter = np.random.RandomState(0).uniform(-0.15, 0.15, len(f1s))
            ax.scatter(
                [x_pos + j for j in jitter], f1s,
                color=colors[cond], s=80, alpha=0.7,
                edgecolors="black", linewidth=0.5,
            )
            # Draw median and range
            f1_median = float(np.median(f1s))
            ax.hlines(f1_median, x_pos - 0.25, x_pos + 0.25,
                      color="black", linewidth=2)
            ax.vlines(x_pos, min(f1s), max(f1s),
                      color="gray", linewidth=1, alpha=0.5)

            x_ticks.append(x_pos)
            x_labels.append(f"{method}\n{cond}")
            x_pos += 1
        x_pos += 0.5   # gap between methods

    ax.set_xticks(x_ticks)
    ax.set_xticklabels(x_labels, fontsize=9)
    ax.set_ylabel("F1")
    ax.set_ylim(0, 1.05)
    ax.set_title("F1 across parameter variants per method per condition\n"
                 "(black bar = median, dots = individual variants)")
    ax.grid(True, linestyle="--", alpha=0.4, axis="y")

    # Custom legend for the two conditions
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor=colors[c],
               markersize=10, label=c)
        for c in conditions
    ]
    ax.legend(handles=legend_elements, loc="lower right")

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


def plot_per_mode_heatmap(all_results, out_path):
    """One row per variant, one column per anomaly mode. One heatmap per condition."""
    mode_names = list(NEG_MODE_NAMES.values())
    conditions = ["clean", "noisy"]

    fig, axes = plt.subplots(1, 2, figsize=(13, 7))

    for ax, cond in zip(axes, conditions):
        variants = [r for r in all_results if r["condition"] == cond]
        labels = [r["variant_label"] for r in variants]
        data = np.zeros((len(variants), len(mode_names)))
        for i, r in enumerate(variants):
            for j, mode in enumerate(mode_names):
                data[i, j] = r["per_mode"].get(mode, {}).get("rejection", 0.0)

        im = ax.imshow(data, vmin=0, vmax=100, cmap="RdYlGn", aspect="auto")
        ax.set_xticks(range(len(mode_names)))
        ax.set_xticklabels(mode_names, rotation=20, ha="right")
        ax.set_yticks(range(len(variants)))
        ax.set_yticklabels(labels, fontsize=8)
        ax.set_title(f"Rejection rate (%) — {cond} training")

        for i in range(len(variants)):
            for j in range(len(mode_names)):
                ax.text(j, i, f"{data[i, j]:.0f}",
                        ha="center", va="center", fontsize=8,
                        color="black" if 20 < data[i, j] < 80 else "white")
        plt.colorbar(im, ax=ax, label="%")

    fig.suptitle("Anomaly rejection rate per variant per mode (higher = better)",
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    out_dir = ROOT / "Data" / "Graphs" / "Metrics_clean_noisy" / timestamp
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output folder: {out_dir}\n")

    data = load_all_data()

    log = {
        "timestamp":       timestamp,
        "tag_k":           TAG_K,
        "git_hash":        _git_hash(),
        "method_variants": {m: v for m, v in METHOD_VARIANTS.items()},
        "n_train":         len(data["clean_train"]),
        "n_test":          len(data["clean_test"]),
        "n_neg":           len(data["neg_traces"]),
        "results":         [],
    }

    save_config(out_dir, TAG_K, METHOD_VARIANTS, data)

    n_variants = sum(len(v) for v in METHOD_VARIANTS.values())
    print(f"\nTotal variants per condition: {n_variants}")
    print(f"Total runs: {n_variants * 2} (clean + noisy)\n")

    # --- Exp 5.1 — clean training -------------------------------------------
    print("=== Experiment 5.1 — Clean training ===")
    log["results"] += _run_condition(
        "clean",
        data["clean_train"], data["clean_test"],
        data["neg_traces"],  data["neg_modes"],
        out_dir, METHOD_VARIANTS, TAG_K,
    )

    # --- Exp 5.2 — noisy training -------------------------------------------
    print("\n=== Experiment 5.2 — Noisy training ===")
    log["results"] += _run_condition(
        "noisy",
        data["noisy_train"], data["noisy_test"],
        data["neg_traces"],  data["neg_modes"],
        out_dir, METHOD_VARIANTS, TAG_K,
    )

    # --- Tables -------------------------------------------------------------
    print("\n=== Tables ===")
    tables = build_tables(log["results"])
    print_tables(tables)
    print()
    save_tables(tables, out_dir)
    save_tables_as_images(tables, out_dir)

    # --- Plots --------------------------------------------------------------
    print("\n=== Plots ===")
    with open(out_dir / "results.json", "w") as f:
        json.dump(log, f, indent=2)
    print(f"  Saved: {out_dir / 'results.json'}")

    plot_metric_comparison(log["results"], "precision", "Precision",
                           out_dir / "comparison_precision.png")
    plot_metric_comparison(log["results"], "recall", "Recall",
                           out_dir / "comparison_recall.png")
    plot_metric_comparison(log["results"], "f1", "F1",
                           out_dir / "comparison_f1.png")
    plot_per_mode_heatmap(log["results"], out_dir / "per_mode_heatmap.png")
    plot_robustness(log["results"], out_dir / "robustness.png")

    print(f"\nDone. Results -> {out_dir}")