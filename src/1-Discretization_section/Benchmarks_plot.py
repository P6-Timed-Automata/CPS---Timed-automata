"""
plot_benchmark.py
=================
Read benchmark_log.json produced by run_benchmark.py and regenerate all figures.
Output goes to the same timestamped folder as the log by default.

Run directly from PyCharm (no arguments) — auto-selects the most recent run.
To plot a specific run:
    python plot_benchmark.py --log path/to/TA_Benchmark/2026-05-12_14-30-00/benchmark_log.json
"""

import argparse
import json
import os

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_log(log_path: str) -> dict:
    with open(log_path) as f:
        return json.load(f)


def _header_color(tab, n_cols, color="#FFD700"):
    for j in range(n_cols):
        tab[(0, j)].set_facecolor(color)


def _highlight_best(tab, n_cols, color="#CCFFCC"):
    for j in range(n_cols):
        tab[(1, j)].set_facecolor(color)


def _bins_display(r: dict) -> str:
    bin_vals = [pt["actual_bins"] - 1 for pt in r["per_trace"]]
    dominant = max(set(bin_vals), key=bin_vals.count)
    return str(dominant) + ("*" if len(set(bin_vals)) > 1 else "")


def _fmt_states(r: dict) -> str:
    """e.g. '121 (81–130)'"""
    return (f"{r['n_states_mean']:.0f} "
            f"({r['n_states_min']:.0f}–{r['n_states_max']:.0f})")


def _fmt_edges(r: dict) -> str:
    """e.g. '204 (140–230)'"""
    return (f"{r['n_edges_mean']:.0f} "
            f"({r['n_edges_min']:.0f}–{r['n_edges_max']:.0f})")


def find_latest_log() -> str:
    base = os.path.normpath(os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "..", "..", "Data", "Graphs", "TA_Benchmark"
    ))
    if not os.path.isdir(base):
        return None
    candidates = []
    for entry in os.scandir(base):
        if entry.is_dir():
            log = os.path.join(entry.path, "benchmark_log.json")
            if os.path.isfile(log):
                candidates.append((entry.name, log))
    if not candidates:
        return None
    candidates.sort(key=lambda x: x[0])
    return candidates[-1][1]


# ---------------------------------------------------------------------------
# Shared table renderer
# ---------------------------------------------------------------------------

def _save_table(method_name, rows, cols, output_folder, suffix,
                n_traces=None, footnote=None, highlight_best=True):
    n_cols     = len(cols)
    n_data     = len(rows)

    # Tight figure: just enough height for title + rows + footer text
    row_h    = 0.38          # inches per data row (incl. header)
    title_h  = 0.30          # inches for title
    footer_h = 0.30          # inches for trace count + footnote
    fig_h    = (n_data + 1) * row_h + title_h + footer_h
    fig_w    = max(7, 1.8 * n_cols)

    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.axis("off")

    # How much of the axes height (0-1) the footer occupies
    footer_frac = footer_h / fig_h
    title_frac  = title_h  / fig_h

    # Table fills the middle: leave footer_frac at bottom, title_frac at top
    table_bottom = footer_frac
    table_height = 1.0 - footer_frac - title_frac
    bbox = [0, table_bottom, 1, table_height]

    tab = ax.table(cellText=rows, colLabels=cols, cellLoc="center", bbox=bbox)
    tab.auto_set_font_size(False)
    tab.set_fontsize(9.5)
    _header_color(tab, n_cols)
    if highlight_best:
        _highlight_best(tab, n_cols)

    # Title just above the table (small pad so it sits close)
    title_y = 1.0 - title_frac / 2
    ax.text(0.5, title_y, f"{method_name} — TA structure",
            ha="center", va="center", fontsize=12, fontweight="bold",
            transform=ax.transAxes)

    # Footer: trace count + optional footnote
    footer_lines = []
    if n_traces is not None:
        footer_lines.append(f"Average of {n_traces} traces")
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
# Plot 1: combined signal + table  (TA_Benchmark.png style)
# ---------------------------------------------------------------------------

def plot_combined(method_name: str, t_raw, v_raw, results: list, output_folder: str):
    is_persist = (method_name == "Persist")
    best = results[0]

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
    ax_top.set_title(f"{method_name} — lowest MAE: {best['label']}")
    ax_top.set_ylabel("Temperature")
    ax_top.legend()

    ax_bot.axhline(0, color="black", lw=1, ls="--")
    ax_bot.plot(t_d / 3600, resids)
    ax_bot.set_xlabel("Time (hours)")
    ax_bot.set_ylabel("Residual")

    if is_persist:
        table_data = [
            [r["label"], _bins_display(r),
             f"{r['mae_mean']:.3f}±{r['mae_std']:.3f}",
             f"{r['time_mean']:.2f}±{r['time_std']:.2f}s"]
            for r in results
        ]
        cols = ["Parameter", "Bins/trace", "MAE", "Time"]
    else:
        table_data = [
            [r["label"],
             f"{r['mae_mean']:.3f}±{r['mae_std']:.3f}",
             f"{r['time_mean']:.2f}±{r['time_std']:.2f}s"]
            for r in results
        ]
        cols = ["Parameter", "MAE", "Time"]

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

def plot_signal(method_name: str, t_raw, v_raw, results: list, output_folder: str):
    best   = results[0]
    t_d    = np.array(best["plot_t_d"])
    v_d    = np.array(best["plot_v_d"])
    resids = np.array(best["plot_resids"])

    fig = plt.figure(figsize=(12, 6))
    gs  = GridSpec(2, 1, height_ratios=[3, 1], hspace=0.25)

    ax_top = fig.add_subplot(gs[0])
    ax_bot = fig.add_subplot(gs[1], sharex=ax_top)

    ax_top.plot(np.array(t_raw) / 3600, v_raw, alpha=0.5, label="Raw")
    ax_top.step(t_d / 3600, v_d, where="post",
                label=f"Discretized (lowest MAE: {best['label']})")
    ax_top.set_title(f"{method_name} — lowest MAE: {best['label']}")
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
# Plot 3: full table — Parameter, (Bins), MAE, States, Edges, Time
# ---------------------------------------------------------------------------

def plot_structure_table_full(method_name: str, results: list, output_folder: str):
    is_persist = (method_name == "Persist")
    n_traces   = len(results[0]["per_trace"])

    rows = []
    for r in results:
        n_states = _fmt_states(r)
        n_edges  = _fmt_edges(r)
        mae      = f"{r['mae_mean']:.3f}±{r['mae_std']:.3f}"
        t        = f"{r['time_mean']:.2f}±{r['time_std']:.2f}s"
        if is_persist:
            rows.append([r["label"], _bins_display(r), mae, n_states, n_edges, t])
        else:
            rows.append([r["label"], mae, n_states, n_edges, t])

    cols = (
        ["Parameter", "Bins/trace", "MAE", "States (min–max)", "Edges (min–max)", "Time"]
        if is_persist else
        ["Parameter", "MAE", "States (min–max)", "Edges (min–max)", "Time"]
    )

    _save_table(method_name, rows, cols, output_folder,
                suffix="_table_full",
                n_traces=n_traces,
                footnote="* = bin count varied across traces" if is_persist else None)


# ---------------------------------------------------------------------------
# Plot 4: compact table — Parameter, (Bins), States, Edges
# ---------------------------------------------------------------------------

def plot_structure_table_compact(method_name: str, results: list, output_folder: str):
    is_persist = (method_name == "Persist")
    n_traces   = len(results[0]["per_trace"])

    rows = []
    for r in results:
        n_states = _fmt_states(r)
        n_edges  = _fmt_edges(r)
        if is_persist:
            rows.append([r["label"], _bins_display(r), n_states, n_edges])
        else:
            rows.append([r["label"], n_states, n_edges])

    cols = (
        ["Parameter", "Bins/trace", "States (min–max)", "Edges (min–max)"]
        if is_persist else
        ["Parameter", "States (min–max)", "Edges (min–max)"]
    )

    _save_table(method_name, rows, cols, output_folder,
                suffix="_table_compact",
                n_traces=n_traces,
                footnote="* = bin count varied across traces" if is_persist else None,
                highlight_best=False)


# ---------------------------------------------------------------------------
# Plot 5: structure + time — Parameter, (Bins), States, Edges, Time
# ---------------------------------------------------------------------------

def plot_structure_table_with_time(method_name: str, results: list, output_folder: str):
    is_persist = (method_name == "Persist")
    n_traces   = len(results[0]["per_trace"])

    rows = []
    for r in results:
        n_states = _fmt_states(r)
        n_edges  = _fmt_edges(r)
        t        = f"{r['time_mean']:.2f}±{r['time_std']:.2f}s"
        if is_persist:
            rows.append([r["label"], _bins_display(r), n_states, n_edges, t])
        else:
            rows.append([r["label"], n_states, n_edges, t])

    cols = (
        ["Parameter", "Bins/trace", "States (min–max)", "Edges (min–max)", "Time"]
        if is_persist else
        ["Parameter", "States (min–max)", "Edges (min–max)", "Time"]
    )

    _save_table(method_name, rows, cols, output_folder,
                suffix="_table_with_time",
                n_traces=n_traces,
                footnote="* = bin count varied across traces" if is_persist else None,
                highlight_best=False)


# ---------------------------------------------------------------------------
# Plot 6: summary table — best variant per method, all methods combined
# ---------------------------------------------------------------------------

def plot_summary_table(log: dict, output_folder: str):
    """One row per method showing its best-MAE variant. Uses '-' where not applicable."""
    cols = ["Method", "Parameter", "Bins/trace", "MAE", "States (min–max)", "Edges (min–max)", "Time"]
    rows = []

    for method_name, results in log["methods"].items():
        best       = results[0]
        is_persist = (method_name == "Persist")

        bins_col = _bins_display(best) if is_persist else "-"
        rows.append([
            method_name,
            best["label"],
            bins_col,
            f"{best['mae_mean']:.3f}±{best['mae_std']:.3f}",
            f"{best['n_states_mean']:.0f} ({best['n_states_min']:.0f}–{best['n_states_max']:.0f})",
            f"{best['n_edges_mean']:.0f} ({best['n_edges_min']:.0f}–{best['n_edges_max']:.0f})",
            f"{best['time_mean']:.2f}±{best['time_std']:.2f}s",
        ])

    # Determine n_traces from any method
    n_traces = len(list(log["methods"].values())[0][0]["per_trace"])

    _save_table("All methods", rows, cols, output_folder,
                suffix="_summary",
                n_traces=n_traces,
                footnote="Best variant (lowest MAE) per method shown. * = bin count varied across traces.",
                highlight_best=False)


# ---------------------------------------------------------------------------
# Plot 7: MAE vs. States trade-off scatter
# ---------------------------------------------------------------------------

def _pareto_front(states: np.ndarray, maes: np.ndarray) -> np.ndarray:
    """Return boolean mask of Pareto-optimal points (minimise both axes)."""
    n      = len(states)
    pareto = np.ones(n, dtype=bool)
    for i in range(n):
        for j in range(n):
            if i != j:
                if states[j] <= states[i] and maes[j] <= maes[i] \
                        and (states[j] < states[i] or maes[j] < maes[i]):
                    pareto[i] = False
                    break
    return pareto


def _knee_index(states: np.ndarray, maes: np.ndarray) -> int:
    """
    Among Pareto-optimal points, find the knee — the point closest to the
    ideal corner (min states, min MAE) in normalised space.
    """
    mask = _pareto_front(states, maes)
    idx  = np.where(mask)[0]
    s    = states[idx]
    m    = maes[idx]

    s_norm = (s - s.min()) / (s.max() - s.min() + 1e-12)
    m_norm = (m - m.min()) / (m.max() - m.min() + 1e-12)
    dist   = np.sqrt(s_norm ** 2 + m_norm ** 2)
    return int(idx[np.argmin(dist)])


# ---------------------------------------------------------------------------
# Plot 7: MAE vs. States trade-off scatter
# ---------------------------------------------------------------------------

def _pareto_front(states: np.ndarray, maes: np.ndarray) -> np.ndarray:
    """Return boolean mask of Pareto-optimal points (minimise both axes)."""
    n      = len(states)
    pareto = np.ones(n, dtype=bool)
    for i in range(n):
        for j in range(n):
            if i != j:
                if states[j] <= states[i] and maes[j] <= maes[i] \
                        and (states[j] < states[i] or maes[j] < maes[i]):
                    pareto[i] = False
                    break
    return pareto


def _knee_index(states: np.ndarray, maes: np.ndarray,
                all_states: np.ndarray, all_maes: np.ndarray) -> int:
    """
    Find the knee of the Pareto frontier as the point closest to the ideal
    (utopia) point in globally-normalised space. Normalising over the full
    dataset (not just the Pareto set) ensures the scale is meaningful.
    """
    mask   = _pareto_front(states, maes)
    idx    = np.where(mask)[0]
    s_norm = (states[idx] - all_states.min()) / (all_states.max() - all_states.min() + 1e-12)
    m_norm = (maes[idx]   - all_maes.min())   / (all_maes.max()   - all_maes.min()   + 1e-12)
    return int(idx[np.argmin(np.sqrt(s_norm ** 2 + m_norm ** 2))])


def _draw_tradeoff_ax(
        ax,
        states:     np.ndarray,
        maes:       np.ndarray,
        labels:     list,
        colors:     list,
        groups:     list,
        n_traces:   int,
        title:      str,
        all_states: np.ndarray = None,
        all_maes:   np.ndarray = None,
) -> None:
    """
    Draw a trade-off scatter on an existing axes.
    Knee is the Pareto point closest to the ideal point in globally-normalised
    space, shown as a hollow red circle so the underlying dot remains visible.
    """
    states = np.array(states)
    maes   = np.array(maes)

    # Fall back to local range if global arrays not provided
    _all_s = all_states if all_states is not None else states
    _all_m = all_maes   if all_maes   is not None else maes

    # --- Data points ---
    for group_name, color in groups:
        mask = np.array(colors) == color
        ax.scatter(
            states[mask], maes[mask],
            color=color, label=group_name,
            s=70, zorder=3, alpha=0.85,
        )

    # --- Pareto frontier ---
    pareto_mask = _pareto_front(states, maes)
    p_idx       = np.where(pareto_mask)[0]
    p_sorted    = p_idx[np.argsort(states[p_idx])]

    ax.step(
        states[p_sorted], maes[p_sorted],
        where="post",
        color="black", linewidth=1.2, linestyle="--",
        label="Pareto frontier", zorder=2, alpha=0.6,
    )
    ax.scatter(
        states[p_sorted], maes[p_sorted],
        color="black", s=30, zorder=4, alpha=0.5,
    )

    # --- Knee: hollow red circle so the underlying dot stays visible ---
    ki = _knee_index(states, maes, _all_s, _all_m)
    ax.scatter(
        states[ki], maes[ki],
        marker="o", s=220, zorder=5,
        facecolors="none", edgecolors="red", linewidths=1.25,
        label=f"Knee: {labels[ki]}",
    )

    # Dotted crosshairs
    ax.axvline(states[ki], color="red", linewidth=0.6, linestyle=":", alpha=0.4)
    ax.axhline(maes[ki],   color="red", linewidth=0.6, linestyle=":", alpha=0.4)

    ax.set_xlabel("Number of States (TA complexity)")
    ax.set_ylabel("MAE (discretization error)")
    ax.set_title(title)
    ax.legend(fontsize=8, framealpha=0.9)
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.text(
        0.02, 0.02,
        f"Lower-left = better  |  avg of {n_traces} traces  |  ○ = knee (closest to ideal point)",
        transform=ax.transAxes, fontsize=7.5, color="gray", va="bottom",
    )


def plot_tradeoff(log: dict, output_folder: str) -> None:
    """
    Produce four trade-off plots:
      1. Combined — all methods, colored by method
      2. Naive only
      3. SAX only
      4. Persist only
    """
    method_colors = {
        "Naive":   "#1f77b4",
        "SAX":     "#ff7f0e",
        "Persist": "#2ca02c",
    }

    n_traces = len(list(log["methods"].values())[0][0]["per_trace"])

    # Build per-method data
    method_data = {}
    for method_name, results in log["methods"].items():
        color = method_colors.get(method_name, "gray")
        method_data[method_name] = {
            "states": np.array([r["n_states_mean"] for r in results]),
            "maes":   np.array([r["mae_mean"]      for r in results]),
            "labels": [r["label"] for r in results],
            "colors": [color] * len(results),
            "color":  color,
        }

    # ------------------------------------------------------------------ #
    # 1. Combined plot                                                     #
    # ------------------------------------------------------------------ #
    all_states = np.concatenate([d["states"] for d in method_data.values()])
    all_maes   = np.concatenate([d["maes"]   for d in method_data.values()])
    all_labels = sum([d["labels"] for d in method_data.values()], [])
    all_colors = sum([d["colors"] for d in method_data.values()], [])
    groups     = [(name, method_colors[name]) for name in method_data]

    fig, ax = plt.subplots(figsize=(10, 7))
    _draw_tradeoff_ax(
        ax, all_states, all_maes, all_labels, all_colors,
        groups=groups, n_traces=n_traces,
        title="MAE vs. TA Complexity — All Methods",
        all_states=all_states, all_maes=all_maes,
    )
    fig.tight_layout()
    out = os.path.join(output_folder, "tradeoff_combined.png")
    fig.savefig(out, bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"  Saved: {out}")

    # ------------------------------------------------------------------ #
    # 2–4. Per-method plots                                               #
    # ------------------------------------------------------------------ #
    for method_name, d in method_data.items():
        # Each point in a per-method plot uses the same base color;
        # vary lightness by index so individual configs are distinguishable
        n   = len(d["states"])
        fig, ax = plt.subplots(figsize=(9, 6))
        _draw_tradeoff_ax(
            ax, d["states"], d["maes"], d["labels"],
            colors=d["colors"],
            groups=[(method_name, d["color"])],
            n_traces=n_traces,
            title=f"MAE vs. TA Complexity — {method_name}",
            all_states=all_states, all_maes=all_maes,
        )
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
        help="Path to benchmark_log.json. Defaults to the most recent run in TA_Benchmark/.",
    )
    parser.add_argument(
        "--out", default=None,
        help="Output folder for figures. Defaults to the same folder as the log file.",
    )
    args = parser.parse_args()

    if args.log is None:
        args.log = find_latest_log()
        if args.log is None:
            print("No benchmark_log.json found. Run run_benchmark.py first.")
            return
        print(f"Auto-selected log: {args.log}")

    out_dir = args.out if args.out else os.path.dirname(os.path.abspath(args.log))
    os.makedirs(out_dir, exist_ok=True)

    log   = load_log(args.log)
    t_raw = log["t_raw"]
    v_raw = log["v_raw"]

    print(f"Plotting run from: {log.get('timestamp', 'unknown')}")
    print(f"Output folder:     {out_dir}")

    for method_name, results in log["methods"].items():
        print(f"\n=== {method_name} ===")
        plot_combined(method_name, t_raw, v_raw, results, out_dir)
        plot_signal(method_name, t_raw, v_raw, results, out_dir)
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