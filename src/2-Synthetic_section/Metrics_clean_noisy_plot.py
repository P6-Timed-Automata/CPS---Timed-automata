"""
exp_51_52_plot.py
=================
Reads results.json produced by exp_51_52_run.py and regenerates all tables
and figures. Output goes to the same folder as the results file by default.

Run directly from PyCharm (no arguments) — auto-selects the most recent run.
To plot a specific run:
    python exp_51_52_plot.py --log path/to/results.json
"""

import argparse
import csv
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from Generators import NEG_MODE_NAMES

# =============================================================================
# STATUS FILTERING
# =============================================================================

def _ok(r):
    """True if this variant completed successfully."""
    return r.get("status", "ok") == "ok"


def _ok_results(results):
    """Filter a list of results to only those that completed successfully."""
    return [r for r in results if _ok(r)]


def _failed_results(results):
    """Filter a list of results to only failed ones."""
    return [r for r in results if not _ok(r)]

# =============================================================================
# LOG LOADING
# =============================================================================

def load_log(log_path):
    with open(log_path) as f:
        return json.load(f)


def find_latest_log():
    """Find the most recent results.json under Metrics_clean_noisy/."""
    base = ROOT / "Data" / "Graphs" / "Metrics_clean_noisy"
    if not base.is_dir():
        return None

    candidates = []
    for entry in os.scandir(base):
        if entry.is_dir():
            log = Path(entry.path) / "results.json"
            if log.is_file():
                candidates.append((entry.name, str(log)))
    if not candidates:
        return None
    candidates.sort(key=lambda x: x[0])
    return candidates[-1][1]


# =============================================================================
# TABLE BUILDERS
# =============================================================================

def _fmt_row(values, widths):
    return "  ".join(str(v).ljust(w) for v, w in zip(values, widths))


def _make_table(headers, rows):
    all_rows = [headers] + [[str(c) for c in r] for r in rows]
    # Added + 2 to provide extra padding between text columns
    widths = [max(len(r[i]) for r in all_rows) + 2 for i in range(len(headers))]
    separator = "  ".join("-" * w for w in widths)
    lines = [_fmt_row(headers, widths), separator]
    for row in rows:
        lines.append(_fmt_row([str(c) for c in row], widths))
    return "\n".join(lines), widths


def build_tables(all_results):
    mode_names = list(NEG_MODE_NAMES.values())
    ok_results = _ok_results(all_results)
    failed_results = _failed_results(all_results)

    # ----- Table 1: Overall metrics (ok variants only, failures listed at end)
    t1_headers = ["Condition", "Method", "Params", "Status", "Precision",
                  "Recall", "F1", "States", "Edges"]
    t1_rows = []
    for r in ok_results:
        ov = r["overall"]
        param_str = ", ".join(f"{k}={v}" for k, v in r["params"].items())
        t1_rows.append([
            r["condition"], r["method"], param_str, "ok",
            f"{ov['precision']:.3f}", f"{ov['recall']:.3f}",
            f"{ov['f1']:.3f}", r["n_states"], r["n_edges"],
        ])
    # Append failed variants
    for r in failed_results:
        param_str = ", ".join(f"{k}={v}" for k, v in r["params"].items())
        err_type = r.get("error_type", "failed")
        t1_rows.append([
            r["condition"], r["method"], param_str, err_type,
            "-", "-", "-", "-", "-",
        ])

    # ----- Tables 2 & 3: Per-mode rejection rate (ok variants only) -----
    def per_mode_rows(condition):
        rows = []
        for r in all_results:
            if r["condition"] != condition:
                continue
            row = [r["variant_label"]]
            if not _ok(r):
                err = r.get("error_type", "failed")
                row.extend([err] * len(mode_names))
            else:
                for mode in mode_names:
                    pct = r["per_mode"].get(mode, {}).get("rejection", 0.0)
                    row.append(f"{pct:.1f}%")
            rows.append(row)
        return rows

    t23_headers = ["Variant"] + [m.capitalize() for m in mode_names]

    # ----- Table 4: Robustness (computed on ok variants only) -----
    t4_headers = ["Condition", "Method", "F1 median", "F1 range",
                  "Best variant", "Worst variant", "N ok / total"]
    t4_rows = []
    grouped = defaultdict(list)
    grouped_all = defaultdict(list)
    for r in all_results:
        grouped_all[(r["condition"], r["method"])].append(r)
        if _ok(r):
            grouped[(r["condition"], r["method"])].append(r)

    for (cond, method), all_variants in sorted(grouped_all.items()):
        variants = grouped[(cond, method)]
        n_total = len(all_variants)
        n_ok = len(variants)
        if not variants:
            t4_rows.append([
                cond, method, "-", "-", "all failed", "-",
                f"0/{n_total}",
            ])
            continue
        f1s = [v["overall"]["f1"] for v in variants]
        labels = [v["variant_label"] for v in variants]
        f1_median = float(np.median(f1s))
        f1_min, f1_max = float(min(f1s)), float(max(f1s))
        best = labels[int(np.argmax(f1s))]
        worst = labels[int(np.argmin(f1s))]
        t4_rows.append([
            cond, method,
            f"{f1_median:.3f}", f"{f1_min:.3f}-{f1_max:.3f}",
            best, worst,
            f"{n_ok}/{n_total}",
        ])

    return {
        "overall": {
            "title":   "Overall metrics (one row per variant per condition)",
            "headers": t1_headers, "rows": t1_rows,
        },
        "per_mode_clean": {
            "title":   "Per-mode rejection rate (%) — clean training",
            "headers": t23_headers, "rows": per_mode_rows("clean"),
        },
        "per_mode_noisy": {
            "title":   "Per-mode rejection rate (%) — noisy training",
            "headers": t23_headers, "rows": per_mode_rows("noisy"),
        },
        "robustness": {
            "title":   "Robustness: F1 spread per method across variants",
            "headers": t4_headers, "rows": t4_rows,
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


def save_tables_as_images(tables, out_dir):
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
        is_mode_table = "per_mode" in key

        n_cols = len(headers)
        n_rows = len(rows) + 1

        # 1. Scale down base figure dimensions for a compact fit
        fig_w = n_cols * 1.1
        fig_h = n_rows * 0.35 + 0.2

        fig, ax = plt.subplots(figsize=(fig_w, fig_h))
        ax.axis("off")

        cell_text = [headers] + [[str(c) for c in r] for r in rows]
        cell_colors = []
        for ri, row in enumerate(cell_text):
            is_header = (ri == 0)
            row_colors = [_cell_color(v, is_header, ri - 1, is_mode_table)
                          for v in row]
            cell_colors.append(row_colors)

        tbl_obj = ax.table(
            cellText=cell_text, cellColours=cell_colors,
            cellLoc="center", loc="center",
        )
        tbl_obj.auto_set_font_size(False)
        tbl_obj.set_fontsize(9)
        tbl_obj.scale(1, 1.4)

        # Keep auto-width to prevent text overlap
        tbl_obj.auto_set_column_width(col=list(range(n_cols)))

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

        # 2. Anchor title to the axes (table) instead of the figure frame
        ax.set_title(tbl["title"], fontsize=11, fontweight="bold",
                     color="#2c3e50", pad=12)

        out_path_png = out_dir / filename_map[key]
        out_path_svg = out_path_png.with_suffix(".svg")

        # 3. Save as both PNG and SVG, trimming the exact bounding box
        fig.savefig(out_path_png, dpi=200, bbox_inches="tight", facecolor="white")
        fig.savefig(out_path_svg, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        print(f"  Saved: {out_path_png.name} & .svg")
# =============================================================================
# PLOTS
# =============================================================================

def plot_metric_comparison(all_results, metric, metric_title, out_path):
    all_results = _ok_results(all_results)   # filter at the top
    if not all_results:
        print(f"  Skipping {metric} plot — no successful variants")
        return

    methods = sorted(set(r["method"] for r in all_results))
    conditions = ["clean", "noisy"]
    colors = {"clean": "steelblue", "noisy": "darkorange"}

    fig, axes = plt.subplots(1, len(methods), figsize=(5 * len(methods), 5),
                             squeeze=False)
    axes = axes[0]

    for ax, method in zip(axes, methods):
        method_results = [r for r in all_results if r["method"] == method]
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
        short_labels = [v.replace(f"{method}_", "") for v in variant_labels]
        ax.set_xticklabels(short_labels, rotation=30, ha="right", fontsize=8)
        ax.set_ylim(0, 1.2)
        ax.set_title(method)
        ax.legend(fontsize=9)
        ax.grid(True, linestyle="--", alpha=0.4, axis="y")

    fig.suptitle(f"{metric_title} per variant — clean vs. noisy training",
                 fontsize=12)
    fig.tight_layout()

    # Save both PNG and SVG
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    fig.savefig(out_path.with_suffix(".svg"), bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path.name} & .svg")


def plot_robustness(all_results, out_path):
    all_results = _ok_results(all_results)
    if not all_results:
        print("  Skipping robustness plot — no successful variants")
        return

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
            jitter = np.random.RandomState(0).uniform(-0.15, 0.15, len(f1s))
            ax.scatter(
                [x_pos + j for j in jitter], f1s,
                color=colors[cond], s=80, alpha=0.7,
                edgecolors="black", linewidth=0.5,
            )
            f1_median = float(np.median(f1s))
            ax.hlines(f1_median, x_pos - 0.25, x_pos + 0.25,
                      color="black", linewidth=2)
            ax.vlines(x_pos, min(f1s), max(f1s),
                      color="gray", linewidth=1, alpha=0.5)

            x_ticks.append(x_pos)
            x_labels.append(f"{method}\n{cond}")
            x_pos += 1
        x_pos += 0.5

    ax.set_xticks(x_ticks)
    ax.set_xticklabels(x_labels, fontsize=9)
    ax.set_ylabel("F1")
    ax.set_ylim(0, 1.05)
    ax.set_title("F1 across parameter variants per method per condition\n"
                 "(black bar = median, dots = individual variants)")
    ax.grid(True, linestyle="--", alpha=0.4, axis="y")

    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor=colors[c],
               markersize=10, label=c)
        for c in conditions
    ]
    ax.legend(handles=legend_elements, loc="lower right")

    fig.tight_layout()

    # Save both PNG and SVG
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    fig.savefig(out_path.with_suffix(".svg"), bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path.name} & .svg")


def plot_per_mode_heatmap(all_results, out_path):
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

    # Save both PNG and SVG
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    fig.savefig(out_path.with_suffix(".svg"), bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path.name} & .svg")


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--log", default=None,
        help="Path to results.json. Defaults to most recent run.",
    )
    parser.add_argument(
        "--out", default=None,
        help="Output folder. Defaults to the same folder as the log file.",
    )
    args = parser.parse_args()

    if args.log is None:
        args.log = find_latest_log()
        if args.log is None:
            print("No results.json found. Run exp_51_52_run.py first.")
            return
        print(f"Auto-selected log: {args.log}")

    out_dir = Path(args.out) if args.out else Path(args.log).parent
    out_dir.mkdir(parents=True, exist_ok=True)

    log = load_log(args.log)
    all_results = log["results"]

    # --- ADJUST BIN LABELS FOR PERSIST ONLY ---
    for r in all_results:
        if r.get("method") == "persist":
            params = r.get("params", {})
            for k in ["bin", "bins"]:
                if k in params:
                    orig_val = params[k]
                    try:
                        new_val = int(orig_val) - 1
                        params[k] = new_val
                        if "variant_label" in r:
                            r["variant_label"] = r["variant_label"].replace(
                                f"{k}={orig_val}", f"{k}={new_val}"
                            )
                    except ValueError:
                        pass
    # ------------------------------------------

    print(f"Plotting run from: {log.get('timestamp', 'unknown')}")
    print(f"Total results:     {len(all_results)}")
    print(f"Output folder:     {out_dir}\n")

    # ----- Tables -----
    print("=== Tables ===")
    tables = build_tables(all_results)
    print_tables(tables)
    print()
    save_tables(tables, out_dir)
    save_tables_as_images(tables, out_dir)

    # ----- Plots -----
    print("\n=== Plots ===")
    plot_metric_comparison(all_results, "precision", "Precision",
                           out_dir / "comparison_precision.png")
    plot_metric_comparison(all_results, "recall", "Recall",
                           out_dir / "comparison_recall.png")
    plot_metric_comparison(all_results, "f1", "F1",
                           out_dir / "comparison_f1.png")
    plot_per_mode_heatmap(all_results, out_dir / "per_mode_heatmap.png")
    plot_robustness(all_results, out_dir / "robustness.png")

    print(f"\nDone. Output: {out_dir}")


if __name__ == "__main__":
    main()