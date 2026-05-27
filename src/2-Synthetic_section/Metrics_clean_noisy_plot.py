"""
exp_51_52_plot.py
=================
Reads results.json produced by exp_51_52_run.py and regenerates the three
figures used in the thesis:

  - comparison_f1.svg          aggregate F1 per variant, clean vs. noisy
  - table_per_mode_clean.svg   per-mode rejection table (clean training)
  - table_per_mode_noisy.svg   per-mode rejection table (noisy training)

(Each is saved as both .png and .svg.)

Run with no arguments to auto-select the most recent run, or
    python exp_51_52_plot.py --log path/to/results.json
to plot a specific one.
"""

import argparse
import json
import os
import sys
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
    return r.get("status", "ok") == "ok"


def _ok_results(results):
    return [r for r in results if _ok(r)]


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
# PERSIST BIN LABEL FIX (cosmetic)
# =============================================================================

def _adjust_persist_bin_labels(all_results):
    """Persist's `bins=N` parameter produces N-1 effective bins. Decrement
    the displayed value in params and variant_label so the figures match
    the effective alphabet size used elsewhere in the thesis."""
    for r in all_results:
        if r.get("method") != "persist":
            continue
        params = r.get("params", {})
        for k in ("bin", "bins"):
            if k not in params:
                continue
            try:
                orig_val = params[k]
                new_val = int(orig_val) - 1
                params[k] = new_val
                if "variant_label" in r:
                    r["variant_label"] = r["variant_label"].replace(
                        f"{k}={orig_val}", f"{k}={new_val}"
                    )
            except ValueError:
                pass


# =============================================================================
# PER-MODE TABLES
# =============================================================================

def build_per_mode_tables(all_results):
    """Build the two per-mode rejection tables (clean and noisy training).
    Returns {key: {title, headers, rows}}."""
    mode_names = list(NEG_MODE_NAMES.values())
    ok_results = _ok_results(all_results)
    headers = ["Variant"] + [m.capitalize() for m in mode_names]

    def rows_for(condition):
        rows = []
        for r in ok_results:
            if r["condition"] != condition:
                continue
            row = [r["variant_label"]]
            for mode in mode_names:
                pct = r["per_mode"].get(mode, {}).get("rejection", 0.0)
                row.append(f"{pct:.1f}%")
            rows.append(row)
        return rows

    return {
        "per_mode_clean": {
            "title":   "Per-mode rejection rate (%) — clean training",
            "headers": headers,
            "rows":    rows_for("clean"),
        },
        "per_mode_noisy": {
            "title":   "Per-mode rejection rate (%) — noisy training",
            "headers": headers,
            "rows":    rows_for("noisy"),
        },
    }


def save_tables_as_images(tables, out_dir):
    """Render each table as a coloured PNG+SVG. Cells are coloured by
    rejection-rate threshold: green ≥80%, orange ≥50%, red below."""
    filename_map = {
        "per_mode_clean": "table_per_mode_clean.png",
        "per_mode_noisy": "table_per_mode_noisy.png",
    }

    HEADER_COLOR = "#2c3e50"
    ROW_EVEN, ROW_ODD = "#f2f4f6", "#ffffff"
    COLORED_BACKGROUNDS = {"#27ae60", "#e74c3c", "#f39c12", HEADER_COLOR}

    def _cell_color(value_str, is_header, row_idx):
        if is_header:
            return HEADER_COLOR
        # First column is the variant label; everything else is a percentage.
        try:
            pct = float(value_str.strip("%"))
        except ValueError:
            return ROW_EVEN if row_idx % 2 == 0 else ROW_ODD
        if pct >= 80:
            return "#27ae60"
        if pct >= 50:
            return "#f39c12"
        return "#e74c3c"

    for key, tbl in tables.items():
        headers, rows = tbl["headers"], tbl["rows"]
        n_cols, n_rows = len(headers), len(rows) + 1

        fig_w = n_cols * 1.1
        fig_h = n_rows * 0.35 + 0.2
        fig, ax = plt.subplots(figsize=(fig_w, fig_h))
        ax.axis("off")

        cell_text = [headers] + [[str(c) for c in r] for r in rows]
        cell_colors = [
            [_cell_color(v, ri == 0, ri - 1) for v in row]
            for ri, row in enumerate(cell_text)
        ]

        tbl_obj = ax.table(
            cellText=cell_text, cellColours=cell_colors,
            cellLoc="center", loc="center",
        )
        tbl_obj.auto_set_font_size(False)
        tbl_obj.set_fontsize(9)
        tbl_obj.scale(1, 1.4)
        # Auto-width prevents long variant labels from overlapping neighbours.
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

        # Anchor title to the axes so the tight bbox includes it.
        ax.set_title(tbl["title"], fontsize=11, fontweight="bold",
                     color="#2c3e50", pad=12)

        out_png = out_dir / filename_map[key]
        out_svg = out_png.with_suffix(".svg")
        fig.savefig(out_png, dpi=200, bbox_inches="tight", facecolor="white")
        fig.savefig(out_svg, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        print(f"  Saved: {out_png.name} & .svg")


# =============================================================================
# F1 COMPARISON PLOT
# =============================================================================

def plot_f1_comparison(all_results, out_path):
    """Bar chart of F1 per variant per method, grouped by training condition.
    One subplot per method; each subplot has clean / noisy bars side by side."""
    all_results = _ok_results(all_results)
    if not all_results:
        print("  Skipping F1 plot — no successful variants")
        return

    methods = sorted({r["method"] for r in all_results})
    conditions = ["clean", "noisy"]
    colors = {"clean": "steelblue", "noisy": "darkorange"}

    fig, axes = plt.subplots(
        1, len(methods), figsize=(5 * len(methods), 5), squeeze=False,
    )
    axes = axes[0]

    for ax, method in zip(axes, methods):
        method_results = [r for r in all_results if r["method"] == method]
        variant_labels = sorted({r["variant_label"] for r in method_results})
        x = np.arange(len(variant_labels))
        width = 0.35

        for ci, cond in enumerate(conditions):
            vals = []
            for vlabel in variant_labels:
                matching = [
                    r for r in method_results
                    if r["variant_label"] == vlabel and r["condition"] == cond
                ]
                vals.append(matching[0]["overall"]["f1"] if matching else 0)

            bars = ax.bar(
                x + (ci - 0.5) * width, vals, width,
                label=cond, color=colors[cond], alpha=0.85,
            )
            for bar, v in zip(bars, vals):
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.01, f"{v:.2f}",
                    ha="center", va="bottom", fontsize=7,
                )

        ax.set_xticks(x)
        short_labels = [v.replace(f"{method}_", "") for v in variant_labels]
        ax.set_xticklabels(short_labels, rotation=30, ha="right", fontsize=8)
        ax.set_ylim(0, 1.2)
        ax.set_title(method)
        ax.legend(fontsize=9)
        ax.grid(True, linestyle="--", alpha=0.4, axis="y")

    fig.suptitle("F1 per variant — clean vs. noisy training", fontsize=12)
    fig.tight_layout()
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
    _adjust_persist_bin_labels(all_results)

    print(f"Plotting run from: {log.get('timestamp', 'unknown')}")
    print(f"Total results:     {len(all_results)}")
    print(f"Output folder:     {out_dir}\n")

    print("=== Tables ===")
    tables = build_per_mode_tables(all_results)
    save_tables_as_images(tables, out_dir)

    print("\n=== Plots ===")
    plot_f1_comparison(all_results, out_dir / "comparison_f1.png")

    print(f"\nDone. Output: {out_dir}")


if __name__ == "__main__":
    main()