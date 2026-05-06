import matplotlib.pyplot as plt
import numpy as np
import os
import time
import glob
from matplotlib.gridspec import GridSpec

# --- PROJECT IMPORTS ---
from Discretization.discretizationSetup import csv_to_temp_time_list, format_output, map_bins_to_symbols
from Discretization.naive import equal_width_discretization
from Discretization.sax import sax_discretization_multi
from Discretization.persist import Persist, get_best_bins, discretize_traces_with_bins, flatten_traces_to_ts
from TAG.TALearner import TALearner


# ---------------------------------------------------------------------------
# 1. HELPERS  (unchanged from your original)
# ---------------------------------------------------------------------------

def load_trace(path):
    data = np.genfromtxt(path, delimiter=';', skip_header=1)
    return data[:, 0], data[:, 1]


def sax_bins_celsius(breakpoints, original_trace_values, outer_z=3.5):
    v = np.asarray(original_trace_values, dtype=float)
    mean, std = v.mean(), v.std()
    if std == 0:
        std = 1.0
    z_edges = np.concatenate([[-outer_z], np.sort(breakpoints), [outer_z]])
    return z_edges * std + mean


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
    mae  = float(np.mean(np.abs(residuals)))
    return rmse, mae, residuals


# ---------------------------------------------------------------------------
# 2. TA LEARNING PIPELINE  (unchanged from your original)
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
        persist_obj = Persist(ts, break_min=k_param, break_max=k_param, skip=np.array([1, 1]))
        bins = get_best_bins(persist_obj, ts)
        traces = discretize_traces_with_bins(data_lists, bins)
        bins_c = bins
    elif method_type == "sax":
        w_val   = params.get('w', 20)
        k_param = params.get('k', 5)
        traces, bps = sax_discretization_multi(data_lists, w=w_val, k=k_param)
        raw_vals = np.concatenate([np.array([v for v, _ in d]) for d in data_lists])
        bins_c = sax_bins_celsius(bps, raw_vals)
    else:
        raise ValueError(f"Unknown method: {method_type}")

    actual_k = len(bins_c) - 1
    symbolic_trace, symbol_map, _ = map_bins_to_symbols(traces, actual_k, bins_c)
    format_output(symbolic_trace, tmp_file)
    _ = TALearner(tss_path=tmp_file, display=False, k=tag_k)

    return traces[0], bins_c


def create_variant(method_type, params, data_lists):
    return lambda: run_ta_pipeline(method_type, params, data_lists)


# ---------------------------------------------------------------------------
# 3. DISCRETIZATION PARAMETER COMPARISON  (unchanged from your original)
# ---------------------------------------------------------------------------

def compare_discretization_params(method_name, t_raw, v_raw, variants_dict, output_folder):
    os.makedirs(output_folder, exist_ok=True)
    results = []

    for label, run_func in variants_dict.items():
        print(f"[{method_name}] Processing: {label}...")
        start_time = time.perf_counter()
        disc_trace, bins_celsius = run_func()
        elapsed_s = time.perf_counter() - start_time

        t_d, v_d = discretized_to_step(disc_trace, bins_celsius)
        rmse, mae, resids = compute_errors(t_d, v_d, t_raw, v_raw)

        results.append({
            'label': label, 'rmse': rmse, 'mae': mae,
            'time_s': elapsed_s, 't_d': t_d, 'v_d': v_d, 'resids': resids
        })

    results.sort(key=lambda x: x['rmse'])
    best = results[0]

    fig = plt.figure(figsize=(18, 8))
    gs  = GridSpec(2, 2, width_ratios=[5, 1], height_ratios=[3, 1], hspace=0.3, wspace=0.1)

    ax_top   = fig.add_subplot(gs[0, 0])
    ax_bot   = fig.add_subplot(gs[1, 0], sharex=ax_top)
    ax_table = fig.add_subplot(gs[:, 1])
    ax_table.axis('off')

    ax_top.plot(t_raw / 3600, v_raw, color='steelblue', alpha=0.5, label='Raw Trace')
    ax_top.step(best['t_d'] / 3600, best['v_d'], where='post',
                color='darkorange', label=f"Best: {best['label']}")
    ax_top.set_ylabel("Temp (°C)")
    ax_top.set_title(f"{method_name} - TA Complexity vs Accuracy")
    ax_top.legend()

    ax_bot.axhline(0, color='black', lw=1, ls='--')
    ax_bot.plot(best['t_d'] / 3600, best['resids'], color='darkorange', lw=0.8)
    ax_bot.fill_between(best['t_d'] / 3600, best['resids'], color='darkorange', alpha=0.1)
    ax_bot.set_ylabel("Residual (°C)")
    ax_bot.set_xlabel("Time (hours)")

    table_data = [[r['label'], f"{r['rmse']:.4f}", f"{r['mae']:.4f}", f"{r['time_s']:.2f}s"]
                  for r in results]
    cols = ["Variant", "RMSE", "MAE", "TA Learn (s)"]
    tab  = ax_table.table(cellText=table_data, colLabels=cols, cellLoc='center', loc='center left')
    tab.auto_set_font_size(False)
    tab.set_fontsize(8)
    tab.scale(1.2, 2.2)
    tab.auto_set_column_width(col=list(range(len(cols))))
    for j in range(len(cols)):
        tab[(0, j)].set_facecolor("#FFD700")

    plt.savefig(os.path.join(output_folder, f"{method_name}_TA_Benchmark.png"),
                bbox_inches='tight', dpi=300)
    plt.close(fig)


# ---------------------------------------------------------------------------
# 4. 2D SCALING BENCHMARK  — traces × alphabet size
# ---------------------------------------------------------------------------

def benchmark_trace_scaling_with_repeats(
        all_trace_files,
        output_folder,
        alphabet_sizes=None,
        repeats=5,
):
    """
    Benchmarks TA learning time across both number of traces AND alphabet size (k).

    Produces two figures:
      TA_Scaling_Lines.png   — one line+std-band per k, x-axis = n_traces
      TA_Scaling_Heatmap.png — mean time heatmap, rows = k, cols = n_traces

    Args:
        all_trace_files : sorted list of CSV paths; sliced to vary n_traces
        output_folder   : directory to save figures into
        alphabet_sizes  : list of k values to sweep (default [4, 8, 12, 16, 20])
        repeats         : timed repetitions per (n_traces, k) cell for stable estimates
    """
    os.makedirs(output_folder, exist_ok=True)

    if alphabet_sizes is None:
        alphabet_sizes = [4, 8, 12, 16, 20]

    trace_counts = list(range(1, len(all_trace_files) + 1))
    n_k  = len(alphabet_sizes)
    n_t  = len(trace_counts)

    all_means = np.zeros((n_k, n_t))
    all_stds  = np.zeros((n_k, n_t))

    print(f"\n--- 2D Benchmark: {n_t} trace counts × {n_k} alphabet sizes, "
          f"{repeats} repeats each ---")

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
            print(f" {m:.3f}s ±{s:.3f}s")

    # -----------------------------------------------------------------------
    # Figure 1: line plot — one curve per k
    # -----------------------------------------------------------------------
    colors = plt.cm.viridis(np.linspace(0.1, 0.9, n_k))

    fig1, ax1 = plt.subplots(figsize=(11, 6))
    for ki, k in enumerate(alphabet_sizes):
        means = all_means[ki]
        stds  = all_stds[ki]
        ax1.plot(trace_counts, means, marker='o', color=colors[ki],
                 linewidth=2, label=f"k={k}")
        ax1.fill_between(trace_counts, means - stds, means + stds,
                         color=colors[ki], alpha=0.15)

    ax1.set_title(f"TA Learning Scaling — Mean of {repeats} Runs")
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

    # -----------------------------------------------------------------------
    # Figure 2: heatmap — rows = k, cols = n_traces, cells = mean time
    # -----------------------------------------------------------------------
    fig2, ax2 = plt.subplots(figsize=(max(8, n_t * 0.7 + 2), max(4, n_k * 0.6 + 1.5)))
    im = ax2.imshow(all_means, aspect='auto', cmap='YlOrRd', origin='lower')

    ax2.set_xticks(range(n_t))
    ax2.set_xticklabels(trace_counts)
    ax2.set_yticks(range(n_k))
    ax2.set_yticklabels(alphabet_sizes)
    ax2.set_xlabel("Number of Input Traces")
    ax2.set_ylabel("Alphabet Size (k)")
    ax2.set_title(f"Mean TA Learning Time (s) — {repeats} repeats per cell")

    # Annotate each cell with mean ± std
    threshold = all_means.max() * 0.6
    for ki in range(n_k):
        for ti in range(n_t):
            text_color = 'white' if all_means[ki, ti] > threshold else 'black'
            ax2.text(ti, ki,
                     f"{all_means[ki, ti]:.2f}\n±{all_stds[ki, ti]:.2f}",
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
    all_traces = sorted(glob.glob(
        "../../data/3-ExtractInterval/1day-experiment/roomA/*.csv"
    ))[:20]

    out_dir = "../../data/Graphs/TA_Benchmark"

    benchmark_trace_scaling_with_repeats(
        all_trace_files = all_traces,
        output_folder   = out_dir,
        alphabet_sizes  = [4, 8, 12, 16, 20],
        repeats         = 5,
    )




# import matplotlib.pyplot as plt
# import numpy as np
# import os
# import time
# from matplotlib.gridspec import GridSpec
# from pathlib import Path
# import glob
#
#
#
# # --- PROJECT IMPORTS ---
# from Discretization.discretizationSetup import (
#     csv_to_temp_time_list,
#     format_output,
#     map_bins_to_symbols
# )
# from Discretization.naive import equal_width_discretization
# from Discretization.sax import sax_discretization_multi
# from Discretization.persist import Persist, get_best_bins, discretize_traces_with_bins, flatten_traces_to_ts
# from TAG.TALearner import TALearner
#
# # ---------------------------------------------------------------------------
# # 1. HELPER FUNCTIONS
# # ---------------------------------------------------------------------------
#
# def load_trace(path):
#     """Loads semicolon-delimited CSV files."""
#     data = np.genfromtxt(path, delimiter=';', skip_header=1)
#     return data[:, 0], data[:, 1]
#
# def sax_bins_celsius(breakpoints, original_trace_values, outer_z=3.5):
#     """Converts SAX z-normalized breakpoints back to Celsius scale."""
#     v = np.asarray(original_trace_values, dtype=float)
#     mean, std = v.mean(), v.std()
#     if std == 0: std = 1.0
#     z_edges = np.concatenate([[-outer_z], np.sort(breakpoints), [outer_z]])
#     return z_edges * std + mean
#
# def discretized_to_step(discretized_trace, bins_celsius):
#     """Reconstructs a step-function signal from discrete labels and bin edges."""
#     bin_centers = (bins_celsius[:-1] + bins_celsius[1:]) / 2
#     times  = np.array([t for _, t in discretized_trace], dtype=float)
#     labels = np.array([l for l, _ in discretized_trace], dtype=int)
#     labels = np.clip(labels, 0, len(bin_centers) - 1)
#     return times, bin_centers[labels]
#
# def compute_errors(t_disc, v_disc, t_raw, v_raw):
#     """Computes RMSE, MAE, and the residuals."""
#     v_raw_interp = np.interp(t_disc, t_raw, v_raw)
#     residuals    = v_disc - v_raw_interp
#     rmse = float(np.sqrt(np.mean(residuals ** 2)))
#     mae  = float(np.mean(np.abs(residuals)))
#     return rmse, mae, residuals
#
# # ---------------------------------------------------------------------------
# # 2. THE TA LEARNING PIPELINE
# # ---------------------------------------------------------------------------
#
# def run_ta_pipeline(method_type, params, data_lists, tag_k=2):
#     """
#     Executes the full chain: Discretize -> Map -> TAG.
#     """
#     # Absolute path to avoid Windows os.makedirs('') errors
#     tmp_file = os.path.join(os.getcwd(), "tmp_ta_learning_data.txt")
#
#     if method_type == "naive":
#         k_param = params.get('k', 5)
#         traces, bins = equal_width_discretization(data_lists, k=k_param)
#         bins_c = bins
#     elif method_type == "persist":
#         k_param = params.get('k', 5)
#         ts = flatten_traces_to_ts(data_lists)
#         persist_obj = Persist(ts, break_min=k_param, break_max=k_param, skip=np.array([1, 1]))
#         bins = get_best_bins(persist_obj, ts)
#         traces = discretize_traces_with_bins(data_lists, bins)
#         bins_c = bins
#     elif method_type == "sax":
#         w_val = params.get('w', 20)
#         k_param = params.get('k', 5)
#         traces, bps = sax_discretization_multi(data_lists, w=w_val, k=k_param)
#         raw_vals = np.concatenate([np.array([v for v, _ in d]) for d in data_lists])
#         bins_c = sax_bins_celsius(bps, raw_vals)
#     else:
#         raise ValueError(f"Unknown method: {method_type}")
#
#     # FIX: Calculate actual_k based on bins returned to avoid IndexError
#     actual_k = len(bins_c) - 1
#     symbolic_trace, symbol_map, _ = map_bins_to_symbols(traces, actual_k, bins_c)
#     format_output(symbolic_trace, tmp_file)
#
#     # TA Generation
#     _ = TALearner(tss_path=tmp_file, display=False, k=tag_k)
#
#     return traces[0], bins_c
#
# def create_variant(method_type, params, data_lists):
#     """Factory function to prevent scoping errors in loops."""
#     return lambda: run_ta_pipeline(method_type, params, data_lists)
#
# # ---------------------------------------------------------------------------
# # 3. COMPARISON & PLOTTING ENGINE
# # ---------------------------------------------------------------------------
#
# def compare_discretization_params(method_name, t_raw, v_raw, variants_dict, output_folder):
#     os.makedirs(output_folder, exist_ok=True)
#     results = []
#
#     for label, run_func in variants_dict.items():
#         print(f"[{method_name}] Processing: {label}...")
#         start_time = time.perf_counter()
#         disc_trace, bins_celsius = run_func()
#         elapsed_s = time.perf_counter() - start_time
#
#         t_d, v_d = discretized_to_step(disc_trace, bins_celsius)
#         rmse, mae, resids = compute_errors(t_d, v_d, t_raw, v_raw)
#
#         results.append({
#             'label': label, 'rmse': rmse, 'mae': mae,
#             'time_s': elapsed_s, 't_d': t_d, 'v_d': v_d, 'resids': resids
#         })
#
#     results.sort(key=lambda x: x['rmse'])
#     best = results[0]
#
#     fig = plt.figure(figsize=(18, 8))
#
#     # --- ADJUSTMENTS START HERE ---
#     # 1. Increased width ratio from 2.5:1 to 5:1 (pushes table left)
#     # 2. Reduced wspace from 0.3 to 0.1 (removes horizontal gap)
#     gs = GridSpec(2, 2, width_ratios=[5, 1], height_ratios=[3, 1], hspace=0.3, wspace=0.1)
#     # --- ADJUSTMENTS END HERE ---
#
#     ax_top = fig.add_subplot(gs[0, 0])
#     ax_bot = fig.add_subplot(gs[1, 0], sharex=ax_top)
#
#     # Position the table subplot
#     ax_table = fig.add_subplot(gs[:, 1])
#     ax_table.axis('off')
#
#     # Main Plot
#     ax_top.plot(t_raw / 3600, v_raw, color='steelblue', alpha=0.5, label='Raw Trace')
#     ax_top.step(best['t_d'] / 3600, best['v_d'], where='post', color='darkorange', label=f"Best: {best['label']}")
#     ax_top.set_ylabel("Temp (°C)")
#     ax_top.set_title(f"{method_name} - TA Complexity vs Accuracy")
#     ax_top.legend()
#
#     # Residual Plot
#     ax_bot.axhline(0, color='black', lw=1, ls='--')
#     ax_bot.plot(best['t_d'] / 3600, best['resids'], color='darkorange', lw=0.8)
#     ax_bot.fill_between(best['t_d'] / 3600, best['resids'], color='darkorange', alpha=0.1)
#     ax_bot.set_ylabel("Residual (°C)")
#     ax_bot.set_xlabel("Time (hours)")
#
#     # Result Table
#     table_data = [[r['label'], f"{r['rmse']:.4f}", f"{r['mae']:.4f}", f"{r['time_s']:.2f}s"] for r in results]
#     cols = ["Variant", "RMSE", "MAE", "TA Learn (s)"]
#
#     # Anchor to 'center left' to stay close to the plot
#     tab = ax_table.table(cellText=table_data, colLabels=cols, cellLoc='center', loc='center left')
#
#     tab.auto_set_font_size(False)
#     tab.set_fontsize(8)  # Reduced slightly to accommodate long SAX strings
#     tab.scale(1.2, 2.2)  # Increased horizontal scale (1.2) to give columns more room
#
#     # --- AUTO-ADJUST COLUMN WIDTHS ---
#     # This prevents the text from clipping the edges
#     tab.auto_set_column_width(col=list(range(len(cols))))
#
#     # Color the header row
#     for j in range(len(cols)):
#         tab[(0, j)].set_facecolor("#FFD700")
#
#     # Use a tight layout to ensure the table isn't cut off by the figure edge
#     plt.savefig(os.path.join(output_folder, f"{method_name}_TA_Benchmark.png"),
#                 bbox_inches='tight',
#                 dpi=300)
#     plt.close(fig)
#
# def benchmark_trace_scaling_with_repeats(all_trace_files, output_folder, repeats=5):
#     """
#     Benchmarks TA learning time for 1 to 10 traces, repeating each
#     measurement to calculate mean and standard deviation.
#     """
#     os.makedirs(output_folder, exist_ok=True)
#     counts = list(range(1, 21))
#
#     mean_times = []
#     std_times = []
#
#     # Use a fixed discretization to isolate the scaling of the TALearner
#     fixed_params = {'k': 11}
#
#     print(f"\n--- Starting Multi-Run Benchmark ({repeats} repeats each) ---")
#
#     for count in counts:
#         iteration_times = []
#         current_files = all_trace_files[:count]
#         data_lists = csv_to_temp_time_list(input_files=current_files)
#
#         print(f"Testing {count} traces: ", end="", flush=True)
#
#         for r in range(repeats):
#             start_time = time.perf_counter()
#             # We only care about the time taken by the TA generator
#             _ = run_ta_pipeline("naive", fixed_params, data_lists)
#             elapsed = time.perf_counter() - start_time
#             iteration_times.append(elapsed)
#             print(".", end="", flush=True)
#
#         avg = np.mean(iteration_times)
#         std = np.std(iteration_times)
#         mean_times.append(avg)
#         std_times.append(std)
#
#         print(f" Average: {avg:.3f}s (±{std:.3f}s)")
#
#     # Convert to numpy for easy math
#     mean_times = np.array(mean_times)
#     std_times = np.array(std_times)
#
#     # --- PLOTTING ---
#     plt.figure(figsize=(10, 6))
#
#     # Plot the mean line
#     plt.plot(counts, mean_times, marker='o', color='navy', label='Mean Learning Time', linewidth=2)
#
#     # Add shaded area for Standard Deviation
#     plt.fill_between(counts,
#                      mean_times - std_times,
#                      mean_times + std_times,
#                      color='navy', alpha=0.2, label='Std. Deviation')
#
#     plt.title(f"TA Learning Scaling (Mean of {repeats} Runs)")
#     plt.xlabel("Number of Input Traces (Training Set Size)")
#     plt.ylabel("Time (seconds)")
#     plt.xticks(counts)
#     plt.grid(True, linestyle='--', alpha=0.5)
#     plt.legend()
#
#     save_path = os.path.join(output_folder, "TA_Scaling_Statistical.png")
#     plt.savefig(save_path, bbox_inches='tight', dpi=300)
#     print(f"\nGraph saved to: {save_path}")
#     plt.show()
#
#
# # ---------------------------------------------------------------------------
# # 4. MAIN EXECUTION
# # ---------------------------------------------------------------------------
#
# if __name__ == "__main__":
#
#     # 1. Gather all available traces (ensure you have at least 10 in this path)
#     all_traces = glob.glob("../../data/3-ExtractInterval/1day-experiment/roomA/*.csv")
#     all_traces = sorted(all_traces)[:20] # Ensure we have exactly 10
#
#     out_dir = "../../data/Graphs/TA_Benchmark"
#
#     # Load raw data for the run_ta_pipeline requirement
#     t_raw, v_raw = load_trace(all_traces[0])
#
#     # 2. Run the benchmark
#     benchmark_trace_scaling_with_repeats(all_traces, out_dir)
#
#
#     #
#     # # --- CONFIGURE PATHS ---
#     # # Update these to the specific 3 traces you want to use
#     # trace_files = [
#     #     "../../data/3-ExtractInterval/1day-experiment/roomA/roomA-1day-tid5.csv",
#     #     "../../data/3-ExtractInterval/1day-experiment/roomA/roomA-1day-tid6.csv",
#     #     "../../data/3-ExtractInterval/1day-experiment/roomA/roomA-1day-tid7.csv"
#     # ]
#     #
#     # out_dir = "../../data/Graphs/TA_Benchmark"
#     #
#     # # 1. Load the first trace for plotting/error calculation (Ground Truth)
#     # # We still compare accuracy against one primary trace to keep metrics consistent
#     # t_raw, v_raw = load_trace(trace_files[0])
#     #
#     # # 2. Load all 3 traces for training
#     # # csv_to_temp_time_list accepts a list of files and returns a list of lists
#     # data_lists = csv_to_temp_time_list(input_files=trace_files)
#     #
#     # print(f"Loaded {len(data_lists)} traces for training.")
#     #
#     # # A. Naive Sweep
#     # naive_vars = {}
#     # for k_val in [4, 8, 12, 16, 20]:
#     #     naive_vars[f"k={k_val}"] = create_variant("naive", {'k': k_val}, data_lists)
#     # compare_discretization_params("Naive", t_raw, v_raw, naive_vars, out_dir)
#     #
#     # # B. SAX Sweep
#     # sax_vars = {}
#     # for w_val in [20, 100, 288]: # Reduced slightly for speed with 3 traces
#     #     for k_val in [4, 8, 16, 20]:
#     #         sax_vars[f"w={w_val}, k={k_val}"] = create_variant("sax", {'w': w_val, 'k': k_val}, data_lists)
#     # compare_discretization_params("SAX", t_raw, v_raw, sax_vars, out_dir)
#     #
#     # # C. Persist Sweep
#     # persist_vars = {}
#     # for k_val in [4, 8, 12, 20]:
#     #     persist_vars[f"k={k_val}"] = create_variant("persist", {'k': k_val}, data_lists)
#     # compare_discretization_params("Persist", t_raw, v_raw, persist_vars, out_dir)