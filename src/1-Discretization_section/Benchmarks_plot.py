"""
plot_benchmark.py
=================
Read benchmark_log.json produced by run_benchmark.py and regenerate all figures.
Output goes to the same timestamped folder as the log by default.

New log layout: <TA_Benchmark>/<timestamp>/k<n>/benchmark_log.json
Use --k to pick a TAG k value (defaults to 2).

Run directly from PyCharm (no arguments) — auto-selects the most recent run with k=2.
To plot a specific run:
    python plot_benchmark.py --log path/to/benchmark_log.json
"""

import argparse
import json
import os

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec


# ---------------------------------------------------------------------------
# Log loading
# ---------------------------------------------------------------------------

def load_log(log_path: str) -> dict:
    with open(log_path) as f:
        return json.load(f)


def find_latest_log(tag_k=2):
    """Find the most recent benchmark_log.json for a given TAG k value."""
    base = os.path.normpath(os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "..", "..", "Data", "Graphs", "TA_Benchmark"
    ))
    if not os.path.isdir(base):
        return None

    candidates = []
    for run_entry in os.scandir(base):
        if not run_entry.is_dir():
            continue
        log = os.path.join(run_entry.path, f"k{tag_k}", "benchmark_log.json")
        if os.path.isfile(log):
            candidates.append((run_entry.name, log))

    if not candidates:
        return None
    candidates.sort(key=lambda x: x[0])
    return candidates[-1][1]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _header_color(tab, n_cols, color="#FFD700"):
    for j in range(n_cols):
        tab[(0, j)].set_facecolor(color)


def _highlight_best(tab, n_cols, color="#CCFFCC"):
    for j in range(n_cols):
        tab[(1, j)].set_facecolor(color)


def _bins_display(r: dict) -> str:
    """Display the chosen bin count for a variant; '*' if it varied across traces."""
    bin_vals = [pt["actual_bins"] for pt in r["per_trace"]]
    dominant = max(set(bin_vals), key=bin_vals.count)
    suffix = "*" if len(set(bin_vals)) > 1 else ""
    return f"{dominant}{suffix}"


def _fmt_states(r: dict) -> str:
    """e.g. '12 (8-19)'"""
    return (f"{r['n_states_median']:.0f} "
            f"({r['n_states_min']}-{r['n_states_max']})")


def _fmt_edges(r: dict) -> str:
    """e.g. '18 (12-25)'"""
    return (f"{r['n_edges_median']:.0f} "
            f"({r['n_edges_min']}-{r['n_edges_max']})")


def _fmt_mae(r: dict) -> str:
    """e.g. '0.123 (0.098-0.187)'"""
    return f"{r['mae_median']:.3f} ({r['mae_min']:.3f}-{r['mae_max']:.3f})"


def _fmt_time(r: dict) -> str:
    """e.g. '0.234±0.045s'"""
    return f"{r['time_median']:.2f}±{r['time_std']:.2f}s"


def _consistency_marker(r: dict) -> str:
    """Inline marker shown after variant labels: ' ✗3/20' or '' if all consistent."""
    n_inc = r["n_total"] - r["n_consistent"]
    if n_inc == 0:
        return ""
    return f" ✗{n_inc}/{r['n_total']}"


def _consistency_cell(r: dict) -> str:
    """Table cell text: '20/20' or '17/20'."""
    return f"{r['n_consistent']}/{r['n_total']}"


def _load_raw_for_variant(r: dict):
    """Load raw (t, v) signal for the median-MAE trace selected as representative."""
    path = r["plot_trace_path"]
    data = np.genfromtxt(path, delimiter=";", skip_header=1)
    return data[:, 0], data[:, 1]


# ---------------------------------------------------------------------------
# Shared table renderer
# ---------------------------------------------------------------------------

def _save_table(method_name, rows, cols, output_folder, suffix,
                n_traces=None, footnote=None, highlight_best=True):
    n_cols = len(cols)
    n_data = len(rows)

    row_h    = 0.38
    title_h  = 0.30
    footer_h = 0.30
    fig_h    = (n_data + 1) * row_h + title_h + footer_h
    fig_w    = max(7, 1.8 * n_cols)

    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.axis("off")

    footer_frac = footer_h / fig_h
    title_frac  = title_h  / fig_h
    table_bottom = footer_frac
    table_height = 1.0 - footer_frac - title_frac
    bbox = [0, table_bottom, 1, table_height]

    tab = ax.table(cellText=rows, colLabels=cols, cellLoc="center", bbox=bbox)
    tab.auto_set_font_size(False)
    tab.set_fontsize(9.5)
    _header_color(tab, n_cols)
    if highlight_best:
        _highlight_best(tab, n_cols)

    title_y = 1.0 - title_frac / 2
    ax.text(0.5, title_y, f"{method_name} — TA structure",
            ha="center", va="center", fontsize=12, fontweight="bold",
            transform=ax.transAxes)

    footer_lines = []
    if n_traces is not None:
        footer_lines.append(f"Median across {n_traces} per-trace TAs")
    if footnote:
        footer_lines.append(footnote)

    if footer_lines:
        ax.text(0.5, footer_frac / 2, "  |  ".join(footer_lines),
                ha="center", va="center", fontsize=8, color="gray",
                transform=ax.transAxes)

    fig.subplots_adjust(top=1.0, bottom=0.0, left=0.01, right=0.99)

    out = os.path.join(output_folder, f"{method_name}{suffix}.png")
    plt.savefig(out, bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"  Saved: {out}")


# ---------------------------------------------------------------------------
# Plot 1: combined signal + table
# ---------------------------------------------------------------------------

def plot_combined(method_name, results, output_folder):
    is_persist = (method_name == "Persist")
    best = results[0]

    t_raw, v_raw = _load_raw_for_variant(best)
    t_d    = np.array(best["plot_t_d"])
    v_d    = np.array(best["plot_v_d"])
    resids = np.array(best["plot_resids"])

    fig = plt.figure(figsize=(18, 8))
    gs  = GridSpec(2, 2, width_ratios=[4.5, 2.8],
                   height_ratios=[3, 1], hspace=0.28, wspace=0.18)

    ax_top   = fig.add_subplot(gs[0, 0])
    ax_bot   = fig.add_subplot(gs[1, 0], sharex=ax_top)
    ax_table = fig.add_subplot(gs[:, 1])
    ax_table.axis("off")

    ax_top.plot(np.array(t_raw) / 3600, v_raw, alpha=0.5, label="Raw")
    ax_top.step(t_d / 3600, v_d, where="post", label="Discretized")
    trace_basename = os.path.basename(best["plot_trace_path"])
    ax_top.set_title(
        f"{method_name} — lowest median MAE: {best['label']}{_consistency_marker(best)}\n"
        f"Representative trace: {trace_basename} (median MAE)"
    )
    ax_top.set_ylabel("Temperature")
    ax_top.legend()

    ax_bot.axhline(0, color="black", lw=1, ls="--")
    ax_bot.plot(t_d / 3600, resids)
    ax_bot.set_xlabel("Time (hours)")
    ax_bot.set_ylabel("Residual")

    if is_persist:
        table_data = [
            [r["label"] + _consistency_marker(r),
             _bins_display(r), _fmt_mae(r), _fmt_time(r)]
            for r in results
        ]
        cols = ["Parameter", "Bins/trace", "MAE (median, min-max)", "Time"]
    else:
        table_data = [
            [r["label"] + _consistency_marker(r),
             _fmt_mae(r), _fmt_time(r)]
            for r in results
        ]
        cols = ["Parameter", "MAE (median, min-max)", "Time"]

    n_cols = len(cols)
    tab = ax_table.table(
        cellText=table_data, colLabels=cols, cellLoc="center", loc="center"
    )
    tab.auto_set_font_size(False)
    tab.set_fontsize(9.5)
    tab.scale(1.2, 2.3)
    _header_color(tab, n_cols)
    _highlight_best(tab, n_cols)

    if is_persist:
        ax_table.text(0.5, -0.03, "* = bin count varied across traces",
                      ha="center", fontsize=8, clip_on=False,
                      transform=ax_table.transAxes, color="gray")

    out = os.path.join(output_folder, f"{method_name}_TA_Benchmark.png")
    plt.savefig(out, bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"  Saved: {out}")


# ---------------------------------------------------------------------------
# Plot 2: signal only
# ---------------------------------------------------------------------------

def plot_signal(method_name, results, output_folder):
    best = results[0]
    t_raw, v_raw = _load_raw_for_variant(best)
    t_d    = np.array(best["plot_t_d"])
    v_d    = np.array(best["plot_v_d"])
    resids = np.array(best["plot_resids"])

    fig = plt.figure(figsize=(12, 6))
    gs  = GridSpec(2, 1, height_ratios=[3, 1], hspace=0.25)

    ax_top = fig.add_subplot(gs[0])
    ax_bot = fig.add_subplot(gs[1], sharex=ax_top)

    ax_top.plot(np.array(t_raw) / 3600, v_raw, alpha=0.5, label="Raw")
    ax_top.step(t_d / 3600, v_d, where="post",
                label=f"Discretized (lowest median MAE: {best['label']})")
    trace_basename = os.path.basename(best["plot_trace_path"])
    ax_top.set_title(
        f"{method_name} — lowest median MAE: {best['label']}{_consistency_marker(best)}\n"
        f"Representative trace: {trace_basename} (median MAE)"
    )
    ax_top.set_ylabel("Temperature")
    ax_top.legend()

    ax_bot.axhline(0, color="black", lw=1, ls="--")
    ax_bot.plot(t_d / 3600, resids)
    ax_bot.set_xlabel("Time (hours)")
    ax_bot.set_ylabel("Residual")

    out = os.path.join(output_folder, f"{method_name}_signal.png")
    plt.savefig(out, bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"  Saved: {out}")


# ---------------------------------------------------------------------------
# Plot 3, 4, 5: structure tables
# ---------------------------------------------------------------------------

def plot_structure_table_full(method_name, results, output_folder):
    is_persist = (method_name == "Persist")
    n_traces = results[0]["n_total"]

    rows = []
    for r in results:
        if is_persist:
            rows.append([r["label"], _bins_display(r), _fmt_mae(r),
                         _fmt_states(r), _fmt_edges(r), _fmt_time(r),
                         _consistency_cell(r)])
        else:
            rows.append([r["label"], _fmt_mae(r),
                         _fmt_states(r), _fmt_edges(r), _fmt_time(r),
                         _consistency_cell(r)])

    cols = (
        ["Parameter", "Bins/trace", "MAE (median, min-max)",
         "States (median, min-max)", "Edges (median, min-max)",
         "Time", "Consistent"]
        if is_persist else
        ["Parameter", "MAE (median, min-max)",
         "States (median, min-max)", "Edges (median, min-max)",
         "Time", "Consistent"]
    )

    _save_table(method_name, rows, cols, output_folder,
                suffix="_table_full", n_traces=n_traces,
                footnote="* = bin count varied across traces" if is_persist else None)


def plot_structure_table_compact(method_name, results, output_folder):
    is_persist = (method_name == "Persist")
    n_traces = results[0]["n_total"]

    rows = []
    for r in results:
        if is_persist:
            rows.append([r["label"], _bins_display(r),
                         _fmt_states(r), _fmt_edges(r), _consistency_cell(r)])
        else:
            rows.append([r["label"],
                         _fmt_states(r), _fmt_edges(r), _consistency_cell(r)])

    cols = (
        ["Parameter", "Bins/trace", "States (median, min-max)",
         "Edges (median, min-max)", "Consistent"]
        if is_persist else
        ["Parameter", "States (median, min-max)",
         "Edges (median, min-max)", "Consistent"]
    )

    _save_table(method_name, rows, cols, output_folder,
                suffix="_table_compact", n_traces=n_traces,
                footnote="* = bin count varied across traces" if is_persist else None,
                highlight_best=False)


def plot_structure_table_with_time(method_name, results, output_folder):
    is_persist = (method_name == "Persist")
    n_traces = results[0]["n_total"]

    rows = []
    for r in results:
        if is_persist:
            rows.append([r["label"], _bins_display(r),
                         _fmt_states(r), _fmt_edges(r),
                         _fmt_time(r), _consistency_cell(r)])
        else:
            rows.append([r["label"],
                         _fmt_states(r), _fmt_edges(r),
                         _fmt_time(r), _consistency_cell(r)])

    cols = (
        ["Parameter", "Bins/trace", "States (median, min-max)",
         "Edges (median, min-max)", "Time", "Consistent"]
        if is_persist else
        ["Parameter", "States (median, min-max)",
         "Edges (median, min-max)", "Time", "Consistent"]
    )

    _save_table(method_name, rows, cols, output_folder,
                suffix="_table_with_time", n_traces=n_traces,
                footnote="* = bin count varied across traces" if is_persist else None,
                highlight_best=False)


# ---------------------------------------------------------------------------
# Plot 6: summary table
# ---------------------------------------------------------------------------

def plot_summary_table(log, output_folder):
    cols = ["Method", "Parameter", "Bins/trace", "MAE (median, min-max)",
            "States (median, min-max)", "Edges (median, min-max)",
            "Time", "Consistent"]
    rows = []

    for method_name, results in log["methods"].items():
        best = results[0]
        is_persist = (method_name == "Persist")
        bins_col = _bins_display(best) if is_persist else "-"
        rows.append([
            method_name, best["label"], bins_col,
            _fmt_mae(best), _fmt_states(best), _fmt_edges(best),
            _fmt_time(best), _consistency_cell(best),
        ])

    n_traces = list(log["methods"].values())[0][0]["n_total"]
    tag_k = log.get("tag_k", "?")

    _save_table(f"All methods (k={tag_k})", rows, cols, output_folder,
                suffix="_summary", n_traces=n_traces,
                footnote="Best variant (lowest median MAE) per method shown. "
                         "* = bin count varied across traces.",
                highlight_best=False)


# ---------------------------------------------------------------------------
# Plot 7: trade-off scatter
# ---------------------------------------------------------------------------

def _pareto_front(states, maes):
    """Return boolean mask of Pareto-optimal points (minimise both axes)."""
    n = len(states)
    pareto = np.ones(n, dtype=bool)
    for i in range(n):
        for j in range(n):
            if i != j:
                if (states[j] <= states[i] and maes[j] <= maes[i]
                        and (states[j] < states[i] or maes[j] < maes[i])):
                    pareto[i] = False
                    break
    return pareto


def _knee_index(states, maes, all_states, all_maes):
    """Knee of the Pareto frontier in globally-normalised space."""
    mask = _pareto_front(states, maes)
    idx = np.where(mask)[0]
    s_norm = (states[idx] - all_states.min()) / (all_states.max() - all_states.min() + 1e-12)
    m_norm = (maes[idx]   - all_maes.min())   / (all_maes.max()   - all_maes.min()   + 1e-12)
    return int(idx[np.argmin(np.sqrt(s_norm ** 2 + m_norm ** 2))])


def _draw_tradeoff_ax(ax, states, maes, labels, colors, groups, n_traces,
                      title, all_states=None, all_maes=None):
    states = np.array(states)
    maes = np.array(maes)
    _all_s = all_states if all_states is not None else states
    _all_m = all_maes if all_maes is not None else maes

    for group_name, color in groups:
        mask = np.array(colors) == color
        ax.scatter(states[mask], maes[mask], color=color, label=group_name,
                   s=70, zorder=3, alpha=0.85)

    pareto_mask = _pareto_front(states, maes)
    p_idx = np.where(pareto_mask)[0]
    p_sorted = p_idx[np.argsort(states[p_idx])]
    ax.step(states[p_sorted], maes[p_sorted], where="post",
            color="black", linewidth=1.2, linestyle="--",
            label="Pareto frontier", zorder=2, alpha=0.6)
    ax.scatter(states[p_sorted], maes[p_sorted],
               color="black", s=30, zorder=4, alpha=0.5)

    ki = _knee_index(states, maes, _all_s, _all_m)
    ax.scatter(states[ki], maes[ki], marker="o", s=220, zorder=5,
               facecolors="none", edgecolors="red", linewidths=1.25,
               label=f"Knee: {labels[ki]}")
    ax.axvline(states[ki], color="red", linewidth=0.6, linestyle=":", alpha=0.4)
    ax.axhline(maes[ki], color="red", linewidth=0.6, linestyle=":", alpha=0.4)

    ax.set_xlabel("Number of States (TA complexity, median)")
    ax.set_ylabel("MAE (discretization error, median)")
    ax.set_title(title)
    ax.legend(fontsize=8, framealpha=0.9)
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.text(0.02, 0.02,
            f"Lower-left = better  |  median across {n_traces} per-trace TAs  "
            f"|  ○ = knee",
            transform=ax.transAxes, fontsize=7.5, color="gray", va="bottom")


def plot_tradeoff(log, output_folder):
    method_colors = {"Naive": "#1f77b4", "SAX": "#ff7f0e", "Persist": "#2ca02c"}
    n_traces = list(log["methods"].values())[0][0]["n_total"]

    method_data = {}
    for method_name, results in log["methods"].items():
        color = method_colors.get(method_name, "gray")
        method_data[method_name] = {
            "states": np.array([r["n_states_median"] for r in results]),
            "maes":   np.array([r["mae_median"]      for r in results]),
            "labels": [r["label"] for r in results],
            "colors": [color] * len(results),
            "color":  color,
        }

    all_states = np.concatenate([d["states"] for d in method_data.values()])
    all_maes   = np.concatenate([d["maes"]   for d in method_data.values()])
    all_labels = sum([d["labels"] for d in method_data.values()], [])
    all_colors = sum([d["colors"] for d in method_data.values()], [])
    groups = [(name, method_colors[name]) for name in method_data]

    fig, ax = plt.subplots(figsize=(10, 7))
    _draw_tradeoff_ax(ax, all_states, all_maes, all_labels, all_colors,
                      groups=groups, n_traces=n_traces,
                      title=f"MAE vs. TA Complexity — All Methods (k={log.get('tag_k', '?')})",
                      all_states=all_states, all_maes=all_maes)
    fig.tight_layout()
    out = os.path.join(output_folder, "tradeoff_combined.png")
    fig.savefig(out, bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"  Saved: {out}")

    for method_name, d in method_data.items():
        fig, ax = plt.subplots(figsize=(9, 6))
        _draw_tradeoff_ax(ax, d["states"], d["maes"], d["labels"],
                          colors=d["colors"], groups=[(method_name, d["color"])],
                          n_traces=n_traces,
                          title=f"MAE vs. TA Complexity — {method_name} (k={log.get('tag_k', '?')})",
                          all_states=all_states, all_maes=all_maes)
        fig.tight_layout()
        out = os.path.join(output_folder, f"tradeoff_{method_name}.png")
        fig.savefig(out, bbox_inches="tight", dpi=300)
        plt.close(fig)
        print(f"  Saved: {out}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--log", default=None,
        help="Path to benchmark_log.json. Defaults to most recent run for the given --k.",
    )
    parser.add_argument(
        "--k", type=int, default=2,
        help="TAG k value to plot (selects <run>/k<n>/benchmark_log.json). Default: 2.",
    )
    parser.add_argument(
        "--out", default=None,
        help="Output folder. Defaults to the same folder as the log file.",
    )
    args = parser.parse_args()

    if args.log is None:
        args.log = find_latest_log(tag_k=args.k)
        if args.log is None:
            print(f"No benchmark_log.json found for k={args.k}. Run run_benchmark.py first.")
            return
        print(f"Auto-selected log: {args.log}")

    out_dir = args.out if args.out else os.path.dirname(os.path.abspath(args.log))
    os.makedirs(out_dir, exist_ok=True)

    log = load_log(args.log)
    print(f"Plotting run from: {log.get('timestamp', 'unknown')} (k={log.get('tag_k', '?')})")
    print(f"Output folder:     {out_dir}")

    for method_name, results in log["methods"].items():
        print(f"\n=== {method_name} ===")
        plot_combined(method_name, results, out_dir)
        plot_signal(method_name, results, out_dir)
        plot_structure_table_full(method_name, results, out_dir)
        plot_structure_table_compact(method_name, results, out_dir)
        plot_structure_table_with_time(method_name, results, out_dir)

    print("\n=== Summary ===")
    plot_summary_table(log, out_dir)

    print("\n=== Trade-off ===")
    plot_tradeoff(log, out_dir)

    print("\nDone.")


if __name__ == "__main__":
    main()