import matplotlib.pyplot as plt
import numpy as np
import os
import time
import glob
from matplotlib.gridspec import GridSpec

from Discretization.discretizationSetup import csv_to_temp_time_list, format_output, map_bins_to_symbols
from Discretization.naive import equal_width_discretization
from Discretization.sax import sax_discretization_multi
from Discretization.persist import Persist, get_best_bins, discretize_traces_with_bins, flatten_traces_to_ts
from TAG.TALearner import TALearner


# ---------------------------------------------------------------------------
# 1. HELPERS
# ---------------------------------------------------------------------------

def load_trace(path):
    data = np.genfromtxt(path, delimiter=';', skip_header=1)
    return data[:, 0], data[:, 1]


# def sax_bins_celsius(bins_z, original_trace_values):
#     # bins_z already includes outer edges from sax_discretization_multi
#     # just convert z-scores to Celsius using global mean/std
#     v = np.asarray(original_trace_values, dtype=float)
#     mean, std = v.mean(), v.std()
#     if std == 0:
#         std = 1.0
#     return np.sort(bins_z) * std + mean

def sax_bins_celsius(bins_z, global_mean, global_std):
    # bins_z already has outer edges — just invert the same normalization
    return np.sort(bins_z) * global_std + global_mean

def discretized_to_step(discretized_trace, bins_celsius):
    bin_centers = (bins_celsius[:-1] + bins_celsius[1:]) / 2
    times  = np.array([t for _, t in discretized_trace], dtype=float)
    labels = np.array([l for l, _ in discretized_trace], dtype=int)
    labels = np.clip(labels, 0, len(bin_centers) - 1)
    return times, bin_centers[labels]


def compute_errors(t_disc, v_disc, t_raw, v_raw):
    v_raw_interp = np.interp(t_disc, t_raw, v_raw)
    residuals    = v_disc - v_raw_interp
    rmse = float(np.sqrt(np.mean(residuals ** 2)))
    return rmse, residuals


# ---------------------------------------------------------------------------
# 2. TA LEARNING PIPELINE
# ---------------------------------------------------------------------------

def run_ta_pipeline(method_type, params, data_lists, tag_k=2):
    tmp_file = os.path.join(os.getcwd(), "tmp_ta_learning_data.txt")

    if method_type == "naive":
        k_param = params.get('k', 5)
        traces, bins = equal_width_discretization(data_lists, k=k_param)
        bins_c = bins

    elif method_type == "persist":
        k_param = params.get('k', 5)
        ts = flatten_traces_to_ts(data_lists)
        persist_obj = Persist(
            ts,
            break_min=2,           # search from 2 up to k_param; Persist picks best
            break_max=k_param,
            skip=np.array([4, 4])  # default skip to avoid degenerate edge breakpoints
        )
        bins = get_best_bins(persist_obj, ts)
        traces = discretize_traces_with_bins(data_lists, bins)
        bins_c = bins

    elif method_type == "sax":
        w_val = params.get('w', 20)
        k_param = params.get('k', 5)
        traces, bins_z, global_mean, global_std = sax_discretization_multi(
            data_lists, w=w_val, k=k_param
        )
        bins_c = sax_bins_celsius(bins_z, global_mean, global_std)

    else:
        raise ValueError(f"Unknown method: {method_type}")

    actual_k = len(bins_c) - 1
    symbolic_trace, symbol_map, _ = map_bins_to_symbols(traces, actual_k, bins_c)
    format_output(symbolic_trace, tmp_file)
    _ = TALearner(tss_path=tmp_file, display=False, k=tag_k)

    return traces[0], bins_c, actual_k


def create_variant(method_type, params, data_lists):
    return lambda: run_ta_pipeline(method_type, params, data_lists)

# ---------------------------------------------------------------------------
# 3. COMPARISON & PLOTTING
# ---------------------------------------------------------------------------

def compare_discretization_params(
        method_name,
        t_raw,
        v_raw,
        variants_dict,
        output_folder,
        trace_files,
        repeats=5):

    os.makedirs(output_folder, exist_ok=True)

    results = []

    # -----------------------------------------------------------------------
    # RUN BENCHMARKS
    # -----------------------------------------------------------------------

    for label, run_func in variants_dict.items():

        print(
            f"[{method_name}] Processing: {label}...",
            end=" ",
            flush=True
        )

        # ---------------------------------------------------------------
        # Train once
        # ---------------------------------------------------------------

        start_time = time.perf_counter()

        disc_trace, bins_celsius, actual_k = run_func()

        elapsed_s = time.perf_counter() - start_time

        # ---------------------------------------------------------------
        # Convert discretized trace
        # ---------------------------------------------------------------

        t_d, v_d = discretized_to_step(
            disc_trace,
            bins_celsius
        )

        # ---------------------------------------------------------------
        # Compute RMSE across multiple traces
        # ---------------------------------------------------------------

        rmses = []

        for trace_path in trace_files[:repeats]:

            t_eval, v_eval = load_trace(trace_path)

            rmse, residuals = compute_errors(
                t_d,
                v_d,
                t_eval,
                v_eval
            )

            rmses.append(rmse)

        mean_rmse = float(np.mean(rmses))
        std_rmse = float(np.std(rmses))

        # residuals only from main trace for plotting
        rmse_plot, resids = compute_errors(
            t_d,
            v_d,
            t_raw,
            v_raw
        )

        print(
            f"RMSE={mean_rmse:.3f} ± {std_rmse:.3f}"
        )

        results.append({
            'label': label,
            'rmse_mean': mean_rmse,
            'rmse_std': std_rmse,
            'time_mean': elapsed_s,
            'time_std': 0.0,
            'actual_k': actual_k,
            't_d': t_d,
            'v_d': v_d,
            'resids': resids
        })

    # -----------------------------------------------------------------------
    # SORT
    # -----------------------------------------------------------------------

    results.sort(
        key=lambda x: x['rmse_mean']
    )

    best = results[0]

    # -----------------------------------------------------------------------
    # FIGURE
    # -----------------------------------------------------------------------

    fig = plt.figure(figsize=(18, 8))

    gs = GridSpec(
        2,
        2,
        width_ratios=[5.8, 1.6],
        height_ratios=[3, 1],
        hspace=0.28,
        wspace=0.015
    )

    ax_top = fig.add_subplot(gs[0, 0])
    ax_bot = fig.add_subplot(gs[1, 0], sharex=ax_top)
    ax_table = fig.add_subplot(gs[:, 1])

    ax_table.axis('off')

    # -----------------------------------------------------------------------
    # TOP PLOT
    # -----------------------------------------------------------------------

    ax_top.plot(
        t_raw / 3600,
        v_raw,
        color='steelblue',
        alpha=0.5,
        label='Raw Trace'
    )

    ax_top.step(
        best['t_d'] / 3600,
        best['v_d'],
        where='post',
        color='darkorange',
        label=f"Best: {best['label']}"
    )

    ax_top.set_ylabel("Temp (°C)")

    ax_top.set_title(
        f"{method_name} - TA Complexity vs Accuracy"
    )

    ax_top.legend()

    # -----------------------------------------------------------------------
    # RESIDUALS
    # -----------------------------------------------------------------------

    ax_bot.axhline(
        0,
        color='black',
        lw=1,
        ls='--'
    )

    ax_bot.plot(
        best['t_d'] / 3600,
        best['resids'],
        color='darkorange',
        lw=0.8
    )

    ax_bot.fill_between(
        best['t_d'] / 3600,
        best['resids'],
        color='darkorange',
        alpha=0.1
    )

    ax_bot.set_ylabel("Residual (°C)")
    ax_bot.set_xlabel("Time (hours)")

    # -----------------------------------------------------------------------
    # TABLE
    # -----------------------------------------------------------------------

    if method_name == "Persist":

        table_data = [
            [
                r['label'],
                str(r['actual_k']),
                f"{r['rmse_mean']:.2f}±{r['rmse_std']:.2f}",
                f"{r['time_mean']:.2f}s"
            ]
            for r in results
        ]

        cols = [
            "Variant",
            "k",
            "RMSE",
            "Learn"
        ]

    else:

        table_data = [
            [
                r['label'],
                f"{r['rmse_mean']:.2f}±{r['rmse_std']:.2f}",
                f"{r['time_mean']:.2f}s"
            ]
            for r in results
        ]

        cols = [
            "Variant",
            "RMSE",
            "Learn"
        ]

    tab = ax_table.table(
        cellText=table_data,
        colLabels=cols,
        cellLoc='center',
        loc='center'
    )

    # -----------------------------------------------------------------------
    # TABLE STYLING
    # -----------------------------------------------------------------------

    tab.auto_set_font_size(False)

    tab.set_fontsize(10)

    tab.scale(1.15, 2.35)

    tab.auto_set_column_width(
        col=list(range(len(cols)))
    )

    for j in range(len(cols)):
        tab[(0, j)].set_facecolor("#FFD700")

    # -----------------------------------------------------------------------
    # SAVE
    # -----------------------------------------------------------------------

    save_path = os.path.join(
        output_folder,
        f"{method_name}_TA_Benchmark.png"
    )

    plt.savefig(
        save_path,
        bbox_inches='tight',
        dpi=300
    )

    plt.close(fig)

    print(
        f"[{method_name}] Saved benchmark plot."
    )

# ---------------------------------------------------------------------------
# 4. SCALING BENCHMARK
# ---------------------------------------------------------------------------

def benchmark_trace_scaling_with_repeats(all_trace_files, output_folder, alphabet_sizes=None, repeats=5):
    os.makedirs(output_folder, exist_ok=True)

    if alphabet_sizes is None:
        alphabet_sizes = [4, 8, 12, 16, 20]

    trace_counts = list(range(1, len(all_trace_files) + 1))
    n_k = len(alphabet_sizes)
    n_t = len(trace_counts)

    all_means = np.zeros((n_k, n_t))
    all_stds  = np.zeros((n_k, n_t))

    print(f"\n--- 2D Benchmark: {n_t} trace counts x {n_k} alphabet sizes, {repeats} repeats ---")

    for ki, k in enumerate(alphabet_sizes):
        print(f"\n  k={k}")
        for ti, count in enumerate(trace_counts):
            data_lists = csv_to_temp_time_list(input_files=all_trace_files[:count])
            times = []
            print(f"    n_traces={count:2d}: ", end="", flush=True)
            for _ in range(repeats):
                t0 = time.perf_counter()
                run_ta_pipeline("naive", {'k': k}, data_lists)
                times.append(time.perf_counter() - t0)
                print(".", end="", flush=True)
            m, s = float(np.mean(times)), float(np.std(times))
            all_means[ki, ti] = m
            all_stds[ki, ti]  = s
            print(f" {m:.3f}s +/-{s:.3f}s")

    colors = plt.cm.viridis(np.linspace(0.1, 0.9, n_k))
    fig1, ax1 = plt.subplots(figsize=(11, 6))
    for ki, k in enumerate(alphabet_sizes):
        means = all_means[ki]
        stds  = all_stds[ki]
        ax1.plot(trace_counts, means, marker='o', color=colors[ki], linewidth=2, label=f"k={k}")
        ax1.fill_between(trace_counts, means - stds, means + stds, color=colors[ki], alpha=0.15)
    ax1.set_title(f"TA Learning Scaling - Mean of {repeats} Runs")
    ax1.set_xlabel("Number of Input Traces")
    ax1.set_ylabel("Learning Time (seconds)")
    ax1.set_xticks(trace_counts)
    ax1.legend(title="Alphabet size (k)", fontsize=9, framealpha=0.9)
    ax1.grid(True, linestyle='--', alpha=0.4)
    fig1.tight_layout()
    lines_path = os.path.join(output_folder, "TA_Scaling_Lines.png")
    fig1.savefig(lines_path, bbox_inches='tight', dpi=300)
    plt.close(fig1)
    print(f"\nSaved {lines_path}")

    fig2, ax2 = plt.subplots(figsize=(max(8, n_t * 0.7 + 2), max(4, n_k * 0.6 + 1.5)))
    im = ax2.imshow(all_means, aspect='auto', cmap='YlOrRd', origin='lower')
    ax2.set_xticks(range(n_t))
    ax2.set_xticklabels(trace_counts)
    ax2.set_yticks(range(n_k))
    ax2.set_yticklabels(alphabet_sizes)
    ax2.set_xlabel("Number of Input Traces")
    ax2.set_ylabel("Alphabet Size (k)")
    ax2.set_title(f"Mean TA Learning Time (s) - {repeats} repeats per cell")
    threshold = all_means.max() * 0.6
    for ki in range(n_k):
        for ti in range(n_t):
            text_color = 'white' if all_means[ki, ti] > threshold else 'black'
            ax2.text(ti, ki, f"{all_means[ki, ti]:.2f}\n+/-{all_stds[ki, ti]:.2f}",
                     ha='center', va='center', fontsize=6.5, color=text_color)
    plt.colorbar(im, ax=ax2, label='Mean time (s)')
    fig2.tight_layout()
    heatmap_path = os.path.join(output_folder, "TA_Scaling_Heatmap.png")
    fig2.savefig(heatmap_path, bbox_inches='tight', dpi=300)
    plt.close(fig2)
    print(f"Saved {heatmap_path}")


# ---------------------------------------------------------------------------
# 5. MAIN
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    trace_files = [
        "../../data/3-ExtractInterval/1day-experiment/roomA/roomA-1day-tid5.csv",
        "../../data/3-ExtractInterval/1day-experiment/roomA/roomA-1day-tid6.csv",
        "../../data/3-ExtractInterval/1day-experiment/roomA/roomA-1day-tid7.csv"
    ]

    out_dir = "../0-old_scripts-Data/TA_Benchmark2"

    t_raw, v_raw = load_trace(trace_files[0])
    data_lists = csv_to_temp_time_list(input_files=trace_files)
    print(f"Loaded {len(data_lists)} traces for training.")

    # A. Naive Sweep
    naive_vars = {}
    for k_val in [2, 4, 6, 8, 10, 12,14, 16]:
        naive_vars[f"k={k_val}"] = create_variant("naive", {'k': k_val}, data_lists)
    compare_discretization_params(
        "Naive",
        t_raw,
        v_raw,
        naive_vars,
        out_dir,
        trace_files,
        repeats=5
    )

    # B. SAX Sweep
    sax_vars = {}
    for w_val in [20, 100, 288]:
        for k_val in [4, 8, 16]:
            sax_vars[f"w={w_val}, k={k_val}"] = create_variant("sax", {'w': w_val, 'k': k_val}, data_lists)
        compare_discretization_params(
            "SAX",
            t_raw,
            v_raw,
            sax_vars,
            out_dir,
            trace_files,
            repeats=5
        )

    # C. Persist Sweep — k_val is the upper bound; Persist selects optimal k internally
    persist_vars = {}
    for k_val in [2,4,6, 8, 10, 12, 14, 16]:
        persist_vars[f"k_max={k_val}"] = create_variant("persist", {'k': k_val}, data_lists)
    compare_discretization_params(
        "Persist",
        t_raw,
        v_raw,
        persist_vars,
        out_dir,
        trace_files,
        repeats=5
    )