"""
plot_benchmark.py
=================
Read benchmark_log.json produced by run_benchmark.py and regenerate all figures.

Handles the "status" field per variant:
  - status="ok"    → normal processing
  - status="failed"→ shown in tables with error info, excluded from plots
                     (scatter, signal overlay) and median/sort calculations

Log layout: <TA_Benchmark>/<timestamp>/k<n>/benchmark_log.json

Usage:
    python plot_benchmark.py                       # latest run for k=2,4
    python plot_benchmark.py --k_list 2            # just k=2
    python plot_benchmark.py --log path/to/log.json
"""

import argparse
import json
import os
from collections import Counter

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec


ROOT = os.path.normpath(os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "..", "..",
))


def _rename_label(label):
    """Display 'k=' as 'bins=' in tables. Leaves other tokens (w=, etc.) intact."""
    return label.replace("k=", "bins=")


def _translate_path(stored_path):
    """
    Stored absolute paths from a remote (SLURM/Linux) run won't resolve on
    other machines. Rebuild relative to the local project ROOT by anchoring
    at the 'Data/' segment.
    """
    s = str(stored_path).replace("\\", "/")
    idx = s.find("/Data/")
    if idx != -1:
        return os.path.normpath(os.path.join(ROOT, s[idx + 1:]))
    if s.startswith("Data/"):
        return os.path.normpath(os.path.join(ROOT, s))
    return stored_path


# ---------------------------------------------------------------------------
# Status filtering
# ---------------------------------------------------------------------------

def _ok(r):
    return r.get("status", "ok") == "ok"


def _ok_results(results):
    return [r for r in results if _ok(r)]


def _failed_results(results):
    return [r for r in results if not _ok(r)]


# ---------------------------------------------------------------------------
# Variant pickers
# ---------------------------------------------------------------------------

def _pick_variant_for_target_bins(results, target_bins):
    """
    Find the ok variant whose dominant actual_bins matches target_bins.
    Among matching variants, return lowest median MAE.
    """
    ok = _ok_results(results)
    candidates = []
    for r in ok:
        bin_counts = Counter(pt["actual_bins"] for pt in r["per_trace"])
        dominant_bins = bin_counts.most_common(1)[0][0]
        if dominant_bins == target_bins:
            candidates.append(r)

    if not candidates:
        return None
    candidates.sort(key=lambda r: r["mae_median"])
    return candidates[0]


def _pick_best_variant(results):
    """Return the ok variant with lowest median MAE, or None."""
    ok = _ok_results(results)
    return ok[0] if ok else None


# ---------------------------------------------------------------------------
# Log loading
# ---------------------------------------------------------------------------

def load_log(log_path: str) -> dict:
    with open(log_path) as f:
        return json.load(f)


def find_latest_log(tag_k=2):
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
# Formatters
# ---------------------------------------------------------------------------

def _header_color(tab, n_cols, color="#FFD700"):
    for j in range(n_cols):
        tab[(0, j)].set_facecolor(color)


def _highlight_best(tab, n_cols, color="#CCFFCC"):
    for j in range(n_cols):
        tab[(1, j)].set_facecolor(color)


def _color_failed_row(tab, row_idx, n_cols, color="#FFCCCC"):
    for j in range(n_cols):
        tab[(row_idx + 1, j)].set_facecolor(color)


def _fmt_states(r):
    if not _ok(r):
        return "FAILED"
    return (f"{r['n_states_median']:.0f} "
            f"({r['n_states_min']}-{r['n_states_max']})")


def _fmt_edges(r):
    if not _ok(r):
        return "FAILED"
    return (f"{r['n_edges_median']:.0f} "
            f"({r['n_edges_min']}-{r['n_edges_max']})")


def _fmt_mae(r):
    if not _ok(r):
        return "FAILED"
    return f"{r['mae_median']:.3f} ({r['mae_min']:.3f}-{r['mae_max']:.3f})"


def _fmt_time(r):
    if not _ok(r):
        return "-"
    return f"{r['time_median']:.2f}±{r['time_std']:.2f}s"


def _consistency_marker(r):
    if not _ok(r):
        return f" ({r.get('error_type', 'failed')})"
    n_inc = r["n_total"] - r["n_consistent"]
    if n_inc == 0:
        return ""
    return f" ✗{n_inc}/{r['n_total']}"


def _consistency_cell(r):
    if not _ok(r):
        return r.get("error_type", "failed")
    return f"{r['n_consistent']}/{r['n_total']}"


def _load_raw_for_variant(r):
    path = _translate_path(r["plot_trace_path"])
    data = np.genfromtxt(path, delimiter=";", skip_header=1)
    return data[:, 0], data[:, 1]


def _load_raw_for_path(path):
    path = _translate_path(path)
    data = np.genfromtxt(path, delimiter=";", skip_header=1)
    return data[:, 0], data[:, 1]


# ---------------------------------------------------------------------------
# Shared table renderer
# ---------------------------------------------------------------------------

def _save_table(method_name, rows, cols, output_folder, suffix,
                n_traces=None, footnote=None, highlight_best=True,
                failed_row_indices=None):
    n_cols = len(cols)
    n_data = len(rows)

    row_h = 0.38
    title_h = 0.30
    footer_h = 0.30
    fig_h = (n_data + 1) * row_h + title_h + footer_h
    fig_w = max(7, 2.0 * n_cols)

    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.axis("off")

    footer_frac = footer_h / fig_h
    title_frac = title_h / fig_h
    table_bottom = footer_frac
    table_height = 1.0 - footer_frac - title_frac
    bbox = [0, table_bottom, 1, table_height]

    tab = ax.table(cellText=rows, colLabels=cols, cellLoc="center", bbox=bbox)
    tab.auto_set_font_size(False)
    tab.set_fontsize(9.5)
    _header_color(tab, n_cols)
    if highlight_best:
        _highlight_best(tab, n_cols)
    if failed_row_indices:
        for idx in failed_row_indices:
            _color_failed_row(tab, idx, n_cols)

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
# Combined signal + table (variant + reference)
# ---------------------------------------------------------------------------

def plot_combined_for_variant_and_reference(
        method_name, variant, all_results, output_folder,
        ref_slot, variant_descriptor,
):
    if variant is None:
        print(f"  Skipping {method_name}_{variant_descriptor}_ref{ref_slot} — no matching variant")
        return

    ref_data = variant.get("reference_traces", [])
    if len(ref_data) <= ref_slot:
        print(f"  Skipping {method_name}_{variant_descriptor}_ref{ref_slot} — no reference data")
        return

    ref = ref_data[ref_slot]
    t_raw, v_raw = _load_raw_for_path(ref["trace_path"])
    t_d = np.array(ref["t_d"])
    v_d = np.array(ref["v_d"])
    resids = np.array(ref["resids"])

    fig = plt.figure(figsize=(18, 8))
    gs = GridSpec(2, 2, width_ratios=[4.2, 3.2],
                  height_ratios=[3, 1], hspace=0.28, wspace=0.18)

    ax_top = fig.add_subplot(gs[0, 0])
    ax_bot = fig.add_subplot(gs[1, 0], sharex=ax_top)
    ax_table = fig.add_subplot(gs[:, 1])
    ax_table.axis("off")

    ax_top.plot(np.array(t_raw) / 3600, v_raw, alpha=0.5, label="Raw")
    ax_top.step(t_d / 3600, v_d, where="post", label="Discretized")

    descriptor_str = "" if variant_descriptor == "best" else f"{variant_descriptor}: "
    ax_top.set_title(
        f"{method_name} — {descriptor_str}{_rename_label(variant['label'])}"
        f"{_consistency_marker(variant)}"
    )
    ax_top.set_ylabel("Temperature")
    ax_top.legend()

    ax_bot.axhline(0, color="black", lw=1, ls="--")
    ax_bot.plot(t_d / 3600, resids)
    ax_bot.set_xlabel("Time (hours)")
    ax_bot.set_ylabel("Residual")

    failed_row_indices = []
    highlighted_idx = None
    table_data = []
    for i, r in enumerate(all_results):
        row = [_rename_label(r["label"]) + _consistency_marker(r), _fmt_mae(r)]
        table_data.append(row)
        if not _ok(r):
            failed_row_indices.append(i)
        if r is variant:
            highlighted_idx = i

    cols = ["Parameter", "MAE\n(median, min-max)"]

    n_cols = len(cols)
    tab = ax_table.table(
        cellText=table_data, colLabels=cols, cellLoc="center", loc="center"
    )
    tab.auto_set_font_size(False)
    tab.set_fontsize(9.5)
    tab.scale(1.2, 2.3)
    _header_color(tab, n_cols)

    if highlighted_idx is not None:
        for j in range(n_cols):
            tab[(highlighted_idx + 1, j)].set_facecolor("#CCFFCC")
    else:
        _highlight_best(tab, n_cols)

    for idx in failed_row_indices:
        _color_failed_row(tab, idx, n_cols)

    out = os.path.join(
        output_folder,
        f"{method_name}_TA_Benchmark_{variant_descriptor}_ref{ref_slot}.png",
    )
    plt.savefig(out, bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"  Saved: {out}")


def plot_signal_for_variant_and_reference(
        method_name, variant, output_folder,
        ref_slot, variant_descriptor,
):
    if variant is None:
        return

    ref_data = variant.get("reference_traces", [])
    if len(ref_data) <= ref_slot:
        return

    ref = ref_data[ref_slot]
    t_raw, v_raw = _load_raw_for_path(ref["trace_path"])
    t_d = np.array(ref["t_d"])
    v_d = np.array(ref["v_d"])
    resids = np.array(ref["resids"])

    fig = plt.figure(figsize=(12, 6))
    gs = GridSpec(2, 1, height_ratios=[3, 1], hspace=0.25)

    ax_top = fig.add_subplot(gs[0])
    ax_bot = fig.add_subplot(gs[1], sharex=ax_top)

    ax_top.plot(np.array(t_raw) / 3600, v_raw, alpha=0.5, label="Raw")
    ax_top.step(t_d / 3600, v_d, where="post",
                label=f"Discretized ({_rename_label(variant['label'])})")

    descriptor_str = "" if variant_descriptor == "best" else f"{variant_descriptor}: "
    ax_top.set_title(
        f"{method_name} — {descriptor_str}{_rename_label(variant['label'])}"
        f"{_consistency_marker(variant)}"
    )
    ax_top.set_ylabel("Temperature")
    ax_top.legend()

    ax_bot.axhline(0, color="black", lw=1, ls="--")
    ax_bot.plot(t_d / 3600, resids)
    ax_bot.set_xlabel("Time (hours)")
    ax_bot.set_ylabel("Residual")

    out = os.path.join(
        output_folder,
        f"{method_name}_signal_{variant_descriptor}_ref{ref_slot}.png",
    )
    plt.savefig(out, bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"  Saved: {out}")


# ---------------------------------------------------------------------------
# Combined signal + table (best variant; uses shared reference trace if available)
# ---------------------------------------------------------------------------

def plot_combined(method_name, results, output_folder):
    ok = _ok_results(results)
    if not ok:
        print(f"  Skipping {method_name}_TA_Benchmark.png — no successful variants")
        return

    best = ok[0]

    # Prefer the first shared reference trace so all methods show the same
    # raw trace. Fall back to per-variant median-MAE trace for older logs.
    ref_traces = best.get("reference_traces", [])
    if ref_traces:
        ref = ref_traces[0]
        t_raw, v_raw = _load_raw_for_path(ref["trace_path"])
        t_d = np.array(ref["t_d"])
        v_d = np.array(ref["v_d"])
        resids = np.array(ref["resids"])
    else:
        t_raw, v_raw = _load_raw_for_variant(best)
        t_d = np.array(best["plot_t_d"])
        v_d = np.array(best["plot_v_d"])
        resids = np.array(best["plot_resids"])

    fig = plt.figure(figsize=(18, 8))
    gs = GridSpec(2, 2, width_ratios=[4.2, 3.2],
                  height_ratios=[3, 1], hspace=0.28, wspace=0.18)

    ax_top = fig.add_subplot(gs[0, 0])
    ax_bot = fig.add_subplot(gs[1, 0], sharex=ax_top)
    ax_table = fig.add_subplot(gs[:, 1])
    ax_table.axis("off")

    ax_top.plot(np.array(t_raw) / 3600, v_raw, alpha=0.5, label="Raw")
    ax_top.step(t_d / 3600, v_d, where="post", label="Discretized")
    ax_top.set_title(
        f"{method_name} — {_rename_label(best['label'])}"
        f"{_consistency_marker(best)}"
    )
    ax_top.set_ylabel("Temperature")
    ax_top.legend()

    ax_bot.axhline(0, color="black", lw=1, ls="--")
    ax_bot.plot(t_d / 3600, resids)
    ax_bot.set_xlabel("Time (hours)")
    ax_bot.set_ylabel("Residual")

    failed_row_indices = []
    table_data = []
    for i, r in enumerate(results):
        row = [_rename_label(r["label"]) + _consistency_marker(r), _fmt_mae(r)]
        table_data.append(row)
        if not _ok(r):
            failed_row_indices.append(i)
    cols = ["Parameter", "MAE\n(median, min-max)"]

    n_cols = len(cols)
    tab = ax_table.table(
        cellText=table_data, colLabels=cols, cellLoc="center", loc="center"
    )
    tab.auto_set_font_size(False)
    tab.set_fontsize(9.5)
    tab.scale(1.2, 2.3)
    _header_color(tab, n_cols)
    _highlight_best(tab, n_cols)
    for idx in failed_row_indices:
        _color_failed_row(tab, idx, n_cols)

    footnote_parts = []
    if failed_row_indices:
        footnote_parts.append(f"{len(failed_row_indices)} failed variants in red")
    if footnote_parts:
        ax_table.text(0.5, -0.03, " | ".join(footnote_parts),
                      ha="center", fontsize=8, clip_on=False,
                      transform=ax_table.transAxes, color="gray")

    base_name = os.path.join(output_folder, f"{method_name}_TA_Benchmark")

    plt.savefig(f"{base_name}.png", bbox_inches="tight", dpi=300)
    plt.savefig(f"{base_name}.svg", bbox_inches="tight") # DPI is ignored for pure SVGs
    plt.close(fig)
    print(f"  Saved: {base_name}.png / .svg")


# ---------------------------------------------------------------------------
# Signal-only plot (best variant)
# ---------------------------------------------------------------------------

def plot_signal(method_name, results, output_folder):
    ok = _ok_results(results)
    if not ok:
        print(f"  Skipping {method_name}_signal.png — no successful variants")
        return

    best = ok[0]

    ref_traces = best.get("reference_traces", [])
    if ref_traces:
        ref = ref_traces[0]
        t_raw, v_raw = _load_raw_for_path(ref["trace_path"])
        t_d = np.array(ref["t_d"])
        v_d = np.array(ref["v_d"])
        resids = np.array(ref["resids"])
    else:
        t_raw, v_raw = _load_raw_for_variant(best)
        t_d = np.array(best["plot_t_d"])
        v_d = np.array(best["plot_v_d"])
        resids = np.array(best["plot_resids"])

    fig = plt.figure(figsize=(12, 6))
    gs = GridSpec(2, 1, height_ratios=[3, 1], hspace=0.25)

    ax_top = fig.add_subplot(gs[0])
    ax_bot = fig.add_subplot(gs[1], sharex=ax_top)

    ax_top.plot(np.array(t_raw) / 3600, v_raw, alpha=0.5, label="Raw")
    ax_top.step(t_d / 3600, v_d, where="post",
                label=f"Discretized ({_rename_label(best['label'])})")
    ax_top.set_title(
        f"{method_name} — {_rename_label(best['label'])}"
        f"{_consistency_marker(best)}"
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
# Structure tables
# ---------------------------------------------------------------------------

def _structure_table_rows(results, columns_kind):
    rows = []
    failed_row_indices = []
    for i, r in enumerate(results):
        label = _rename_label(r["label"])
        if columns_kind == "full":
            cells = [label, _fmt_mae(r),
                     _fmt_states(r), _fmt_edges(r),
                     _fmt_time(r), _consistency_cell(r)]
        elif columns_kind == "compact":
            cells = [label, _fmt_states(r), _fmt_edges(r), _consistency_cell(r)]
        elif columns_kind == "with_time":
            cells = [label, _fmt_states(r), _fmt_edges(r), _fmt_time(r)]
        else:
            raise ValueError(f"Unknown columns_kind: {columns_kind}")
        rows.append(cells)
        if not _ok(r):
            failed_row_indices.append(i)
    return rows, failed_row_indices


def _structure_table_cols(columns_kind):
    return {
        "full":      ["Parameter", "MAE\n(median, min-max)",
                      "States\n(median, min-max)", "Edges\n(median, min-max)",
                      "TA training Time(s)", "Consistent"],
        "compact":   ["Parameter",
                      "States\n(median, min-max)", "Edges\n(median, min-max)",
                      "Consistent"],
        "with_time": ["Parameter",
                      "States\n(median, min-max)", "Edges\n(median, min-max)",
                      "TA training Time(s)"],
    }[columns_kind]


def plot_structure_table_full(method_name, results, output_folder):
    n_traces = next((r["n_total"] for r in results if _ok(r)), None)
    rows, failed_idx = _structure_table_rows(results, "full")
    cols = _structure_table_cols("full")

    footnote_parts = []
    if failed_idx:
        footnote_parts.append(f"{len(failed_idx)} failed variants in red")

    _save_table(method_name, rows, cols, output_folder,
                suffix="_table_full",
                n_traces=n_traces,
                footnote=" | ".join(footnote_parts) if footnote_parts else None,
                failed_row_indices=failed_idx)


def plot_structure_table_compact(method_name, results, output_folder):
    n_traces = next((r["n_total"] for r in results if _ok(r)), None)
    rows, failed_idx = _structure_table_rows(results, "compact")
    cols = _structure_table_cols("compact")

    footnote_parts = []
    if failed_idx:
        footnote_parts.append(f"{len(failed_idx)} failed variants in red")

    _save_table(method_name, rows, cols, output_folder,
                suffix="_table_compact",
                n_traces=n_traces,
                footnote=" | ".join(footnote_parts) if footnote_parts else None,
                highlight_best=False,
                failed_row_indices=failed_idx)


def plot_structure_table_with_time(method_name, results, output_folder):
    n_traces = next((r["n_total"] for r in results if _ok(r)), None)
    rows, failed_idx = _structure_table_rows(results, "with_time")
    cols = _structure_table_cols("with_time")

    footnote_parts = []
    if failed_idx:
        footnote_parts.append(f"{len(failed_idx)} failed variants in red")

    _save_table(method_name, rows, cols, output_folder,
                suffix="_table_with_time",
                n_traces=n_traces,
                footnote=" | ".join(footnote_parts) if footnote_parts else None,
                highlight_best=False,
                failed_row_indices=failed_idx)


# ---------------------------------------------------------------------------
# Summary table
# ---------------------------------------------------------------------------

def plot_summary_table(log, output_folder):
    cols = ["Method", "Parameter", "MAE\n(median, min-max)",
            "States\n(median, min-max)", "Edges\n(median, min-max)",
            "TA training Time(s)", "Consistent"]
    rows = []

    for method_name, results in log["methods"].items():
        ok = _ok_results(results)
        if not ok:
            rows.append([method_name, "all variants failed",
                         "-", "-", "-", "-",
                         f"0/{len(results)}"])
            continue
        best = ok[0]
        rows.append([
            method_name, _rename_label(best["label"]),
            _fmt_mae(best), _fmt_states(best), _fmt_edges(best),
            _fmt_time(best), _consistency_cell(best),
        ])

    n_traces = None
    for results in log["methods"].values():
        ok = _ok_results(results)
        if ok:
            n_traces = ok[0]["n_total"]
            break
    tag_k = log.get("tag_k", "?")

    footnote_text = "Best variant (lowest median MAE) per method shown."

    _save_table(f"All methods (k={tag_k})", rows, cols, output_folder,
                suffix="_summary", n_traces=n_traces,
                footnote=footnote_text,
                highlight_best=False)


# ---------------------------------------------------------------------------
# Trade-off scatter
# ---------------------------------------------------------------------------

def _pareto_front(states, maes):
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
               label=f"Knee: {_rename_label(labels[ki])}")
    ax.axvline(states[ki], color="red", linewidth=0.6, linestyle=":", alpha=0.4)
    ax.axhline(maes[ki], color="red", linewidth=0.6, linestyle=":", alpha=0.4)

    ax.set_xlabel("Number of States (TA complexity, median)")
    ax.set_ylabel("MAE (discretization error, median)")
    ax.set_title(title)
    ax.legend(fontsize=8, framealpha=0.9)
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.text(0.02, 0.02,
            f"Lower-left = better  |  median across {n_traces} per-trace TAs  "
            f"|  ○ = knee  |  failed variants excluded",
            transform=ax.transAxes, fontsize=7.5, color="gray", va="bottom")


def plot_tradeoff(log, output_folder):
    method_colors = {"Naive": "#1f77b4", "SAX": "#ff7f0e", "Persist": "#2ca02c"}

    n_traces = None
    for results in log["methods"].values():
        ok = _ok_results(results)
        if ok:
            n_traces = ok[0]["n_total"]
            break

    if n_traces is None:
        print("  Skipping tradeoff plots — no successful variants in any method.")
        return

    method_data = {}
    for method_name, results in log["methods"].items():
        ok = _ok_results(results)
        if not ok:
            continue
        color = method_colors.get(method_name, "gray")
        method_data[method_name] = {
            "states": np.array([r["n_states_median"] for r in ok]),
            "maes":   np.array([r["mae_median"]      for r in ok]),
            "labels": [r["label"] for r in ok],
            "colors": [color] * len(ok),
            "color":  color,
        }

    if not method_data:
        print("  Skipping tradeoff plots — no methods have successful variants.")
        return

    all_states = np.concatenate([d["states"] for d in method_data.values()])
    all_maes = np.concatenate([d["maes"] for d in method_data.values()])
    all_labels = sum([d["labels"] for d in method_data.values()], [])
    all_colors = sum([d["colors"] for d in method_data.values()], [])
    groups = [(name, method_colors.get(name, "gray")) for name in method_data]

    fig, ax = plt.subplots(figsize=(10, 7))
    _draw_tradeoff_ax(ax, all_states, all_maes, all_labels, all_colors,
                      groups=groups, n_traces=n_traces,
                      title=f"MAE vs. TA Complexity — All Methods (k-future={log.get('tag_k', '?')})",
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
                          title=f"MAE vs. TA Complexity — {method_name} (k-future={log.get('tag_k', '?')})",
                          all_states=all_states, all_maes=all_maes)
        fig.tight_layout()
        out = os.path.join(output_folder, f"tradeoff_{method_name}.png")
        fig.savefig(out, bbox_inches="tight", dpi=300)
        plt.close(fig)
        print(f"  Saved: {out}")


# ---------------------------------------------------------------------------
# Failure summary
# ---------------------------------------------------------------------------

def plot_failure_summary(log, output_folder):
    all_failed = []
    for method_name, results in log["methods"].items():
        for r in _failed_results(results):
            all_failed.append({
                "method":     method_name,
                "label":      r["label"],
                "error_type": r.get("error_type", "?"),
                "error_msg":  r.get("error_msg", "")[:120],
            })

    if not all_failed:
        return

    cols = ["Method", "Variant", "Error type", "Error message (truncated)"]
    rows = [
        [f["method"], _rename_label(f["label"]), f["error_type"], f["error_msg"]]
        for f in all_failed
    ]

    tag_k = log.get("tag_k", "?")
    _save_table(f"Failed variants (k={tag_k})", rows, cols, output_folder,
                suffix="_failures", highlight_best=False,
                footnote=f"{len(all_failed)} variants failed during this run")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--log", default=None,
        help="Path to a specific benchmark_log.json. If provided, --k_list is ignored.",
    )
    parser.add_argument(
        "--k_list", type=int, nargs="+", default=[2, 4],
        help="Space-separated list of TAG k values to plot. Default: 2 4",
    )
    parser.add_argument(
        "--out", default=None,
        help="Base output folder. Results saved in k-specific subfolders.",
    )
    args = parser.parse_args()

    logs_to_process = []

    if args.log:
        logs_to_process.append(args.log)
    else:
        for k in args.k_list:
            log_path = find_latest_log(tag_k=k)
            if log_path:
                logs_to_process.append(log_path)
            else:
                print(f"Warning: No benchmark_log.json found for k={k}. Skipping.")

    if not logs_to_process:
        print("No logs found to process. Exiting.")
        return

    for log_path in logs_to_process:
        log = load_log(log_path)
        current_k = log.get("tag_k", "?")

        print(f"\n" + "=" * 40)
        print(f"Processing k={current_k}")
        print(f"Log: {log_path}")
        print("=" * 40)

        if args.out:
            k_out_dir = os.path.join(args.out, f"k{current_k}")
        else:
            k_out_dir = os.path.dirname(os.path.abspath(log_path))

        os.makedirs(k_out_dir, exist_ok=True)
        print(f"Output folder: {k_out_dir}")

        n_failed = sum(
            1
            for results in log["methods"].values()
            for r in _failed_results(results)
        )
        if n_failed:
            print(f"Note: {n_failed} variant(s) failed; will be shown in red.")

        for method_name, results in log["methods"].items():
            print(f"  Plotting method: {method_name}")
            plot_combined(method_name, results, k_out_dir)
            plot_signal(method_name, results, k_out_dir)
            plot_structure_table_full(method_name, results, k_out_dir)
            plot_structure_table_compact(method_name, results, k_out_dir)
            plot_structure_table_with_time(method_name, results, k_out_dir)

        FIXED_BIN_TARGETS = [5, 10]

        reference_indices = log.get("reference_indices", [])
        if not reference_indices:
            print("  No reference_indices in log — skipping per-reference plots.")
        else:
            print(f"\n  Reference traces: {reference_indices}")

            for method_name, results in log["methods"].items():
                variant_pairs = []

                best_variant = _pick_best_variant(results)
                if best_variant is not None:
                    variant_pairs.append((best_variant, "best"))

                for target_bins in FIXED_BIN_TARGETS:
                    picked = _pick_variant_for_target_bins(results, target_bins)
                    if picked is None:
                        print(f"  No variant for {method_name} produces {target_bins} bins; "
                              f"skipping {target_bins}bins plots.")
                        continue
                    descriptor = f"{target_bins}bins"
                    if picked is best_variant:
                        print(f"  Best variant for {method_name} already produces "
                              f"{target_bins} bins; skipping duplicate.")
                        continue
                    variant_pairs.append((picked, descriptor))

                for variant, descriptor in variant_pairs:
                    for ref_slot in range(len(reference_indices)):
                        plot_combined_for_variant_and_reference(
                            method_name, variant, results, k_out_dir,
                            ref_slot, descriptor,
                        )
                        plot_signal_for_variant_and_reference(
                            method_name, variant, k_out_dir,
                            ref_slot, descriptor,
                        )

        plot_summary_table(log, k_out_dir)
        plot_tradeoff(log, k_out_dir)
        plot_failure_summary(log, k_out_dir)

    print("\nAll k-values processed successfully.")


if __name__ == "__main__":
    main()