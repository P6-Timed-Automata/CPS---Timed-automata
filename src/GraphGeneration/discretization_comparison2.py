import matplotlib.pyplot as plt
import numpy as np
import os
import time
from matplotlib.gridspec import GridSpec

from Discretization.discretizationSetup import (
    csv_to_temp_time_list,
    format_output,
    map_bins_to_symbols
)

from Discretization.naive import equal_width_discretization
from Discretization.sax import sax_discretization_multi
from Discretization.persist import (
    Persist,
    get_best_bins,
    discretize_traces_with_bins,
    flatten_traces_to_ts
)

from TAG.TALearner import TALearner


# ---------------------------------------------------------------------------
# 1. HELPERS
# ---------------------------------------------------------------------------

def load_trace(path):
    data = np.genfromtxt(path, delimiter=';', skip_header=1)
    return data[:, 0], data[:, 1]


def sax_bins_celsius(bins_z, global_mean, global_std):
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
    rmse         = float(np.sqrt(np.mean(residuals ** 2)))
    return rmse, residuals


# ---------------------------------------------------------------------------
# 2. PIPELINE  (single trace)
# ---------------------------------------------------------------------------

def run_ta_pipeline(method_type, params, single_trace_data, tag_k=2):
    """Runs discretization + TA on a list containing exactly one trace."""

    tmp_file = os.path.join(os.getcwd(), "tmp_ta_learning_data.txt")

    if method_type == "naive":
        traces, bins_c = equal_width_discretization(single_trace_data, k=params['k'])

    elif method_type == "persist":
        ts = flatten_traces_to_ts(single_trace_data)
        persist_obj = Persist(ts, break_min=2, break_max=params['k'], skip=np.array([4, 4]))
        bins_c  = get_best_bins(persist_obj, ts)
        traces  = discretize_traces_with_bins(single_trace_data, bins_c)

    elif method_type == "sax":
        traces, bins_z, mean_, std_ = sax_discretization_multi(
            single_trace_data, w=params['w'], k=params['k']
        )
        bins_c = sax_bins_celsius(bins_z, mean_, std_)

    else:
        raise ValueError(method_type)

    actual_k = len(bins_c) - 1

    symbolic_trace, symbol_map, _ = map_bins_to_symbols(traces, actual_k, bins_c)
    format_output(symbolic_trace, tmp_file)
    _ = TALearner(tmp_file, display=False, k=tag_k)

    return traces, bins_c, actual_k


# ---------------------------------------------------------------------------
# 3. MAIN EXPERIMENT LOOP
# ---------------------------------------------------------------------------

def compare_discretization_params(
        method_name,
        t_raw,          # raw signal of trace 0, for the plot background
        v_raw,
        variants_dict,  # {label: (method_type, params)}
        data_lists,     # one entry per trace
        trace_files,    # matching file paths
        output_folder,
        repeats=5):

    os.makedirs(output_folder, exist_ok=True)
    results = []

    for label, (method_type, params) in variants_dict.items():
        print(f"[{method_name}] {label} ...", flush=True)

        per_trace_rmse  = []
        per_trace_times = []
        plot_t_d = plot_v_d = plot_resids = None
        actual_k = None

        per_trace_k = []

        for i, (trace_path, trace_data) in enumerate(zip(trace_files, data_lists)):

            t_i, v_i = load_trace(trace_path)

            start = time.perf_counter()
            disc_traces, bins_c, actual_k = run_ta_pipeline(method_type, params, [trace_data])
            per_trace_k.append(actual_k)
            elapsed = time.perf_counter() - start

            per_trace_times.append(elapsed)

            t_d, v_d    = discretized_to_step(disc_traces[0], bins_c)
            rmse, resids = compute_errors(t_d, v_d, t_i, v_i)
            per_trace_rmse.append(rmse)

            if i == 0:
                plot_t_d, plot_v_d, plot_resids = t_d, v_d, resids

        mean_rmse = float(np.mean(per_trace_rmse))
        std_rmse  = float(np.std(per_trace_rmse))
        mean_time = float(np.mean(per_trace_times))
        std_time  = float(np.std(per_trace_times))

        print(f"  RMSE={mean_rmse:.3f}±{std_rmse:.3f} | time={mean_time:.3f}±{std_time:.3f}s")

        results.append({
            'label':     label,
            'rmse_mean': mean_rmse,
            'rmse_std':  std_rmse,
            'time_mean': mean_time,
            'time_std':  std_time,
            'actual_k':  actual_k,
            't_d':       plot_t_d,
            'v_d':       plot_v_d,
            'resids':    plot_resids,
            'per_trace_k': per_trace_k,
        })

    # -----------------------------------------------------------------------
    # BEST MODEL
    # -----------------------------------------------------------------------
    results.sort(key=lambda x: x['rmse_mean'])
    best = results[0]

    # -----------------------------------------------------------------------
    # PLOT
    # -----------------------------------------------------------------------
    fig = plt.figure(figsize=(18, 8))
    gs  = GridSpec(2, 2, width_ratios=[4.5, 2.8], height_ratios=[3, 1],
                   hspace=0.28, wspace=0.18)

    ax_top   = fig.add_subplot(gs[0, 0])
    ax_bot   = fig.add_subplot(gs[1, 0], sharex=ax_top)
    ax_table = fig.add_subplot(gs[:, 1])
    ax_table.axis('off')

    ax_top.plot(t_raw / 3600, v_raw, alpha=0.5, label="Raw")
    ax_top.step(best['t_d'] / 3600, best['v_d'], where='post', label="Discretized")
    ax_top.set_title(f"{method_name} — best variant: {best['label']}")
    ax_top.legend()

    ax_bot.axhline(0, color='black', lw=1, ls='--')
    ax_bot.plot(best['t_d'] / 3600, best['resids'])
    ax_bot.set_xlabel("Time (hours)")
    ax_bot.set_ylabel("Residual")

    if method_name == "Persist":
        table_data = [
            [
                r['label'],
                ", ".join(map(str, r['per_trace_k'])),
                f"{r['rmse_mean']:.2f}±{r['rmse_std']:.2f}",
                f"{r['time_mean']:.2f}±{r['time_std']:.2f}s"
            ]
            for r in results
        ]
        cols = ["Parameter", "k per trace", "RMSE", "Time"]
    else:
        table_data = [
            [r['label'],
             f"{r['rmse_mean']:.2f}±{r['rmse_std']:.2f}",
             f"{r['time_mean']:.2f}±{r['time_std']:.2f}s"]
            for r in results
        ]
        cols = ["Parameter", "RMSE", "Time"]

    tab = ax_table.table(cellText=table_data, colLabels=cols,
                         cellLoc='center', loc='center')
    tab.auto_set_font_size(False)
    tab.set_fontsize(9.5)
    tab.scale(1.2, 2.3)
    for j in range(len(cols)):
        tab[(0, j)].set_facecolor("#FFD700")

    plt.savefig(os.path.join(output_folder, f"{method_name}_TA_Benchmark.png"),
                bbox_inches='tight', dpi=300)
    plt.close(fig)
    print(f"[{method_name}] Saved.")


# ---------------------------------------------------------------------------
# 4. MAIN
# ---------------------------------------------------------------------------

if __name__ == "__main__":

    trace_files = [
        "../../data/3-ExtractInterval/1day-experiment/roomA/roomA-1day-tid5.csv",
        "../../data/3-ExtractInterval/1day-experiment/roomA/roomA-1day-tid6.csv",
        "../../data/3-ExtractInterval/1day-experiment/roomA/roomA-1day-tid7.csv",
        "../../data/3-ExtractInterval/1day-experiment/roomA/roomA-1day-tid8.csv",
        "../../data/3-ExtractInterval/1day-experiment/roomA/roomA-1day-tid9.csv",
    ]

    out_dir = "../../Data/Graphs/TA_Benchmark2"

    t_raw, v_raw = load_trace(trace_files[0])

    data_lists = csv_to_temp_time_list(input_files=trace_files)
    print(f"Loaded {len(data_lists)} traces.")

    # -----------------------------------------------------------------------
    # NAIVE
    # -----------------------------------------------------------------------
    naive_vars = {f"k={k}": ("naive", {'k': k}) for k in [2, 4, 6, 8, 10, 12, 14, 16]}

    compare_discretization_params(
        "Naive", t_raw, v_raw, naive_vars, data_lists, trace_files, out_dir
    )

    # -----------------------------------------------------------------------
    # SAX
    # -----------------------------------------------------------------------
    sax_vars = {
        f"w={w}, k={k}": ("sax", {'w': w, 'k': k})
        for w in [20, 100, 288]
        for k in [4, 8, 16]
    }

    compare_discretization_params(
        "SAX", t_raw, v_raw, sax_vars, data_lists, trace_files, out_dir
    )

    # -----------------------------------------------------------------------
    # PERSIST
    # -----------------------------------------------------------------------
    persist_vars = {f"k_max={k}": ("persist", {'k': k}) for k in [2, 4, 6, 8, 10, 12, 14, 16]}

    compare_discretization_params(
        "Persist", t_raw, v_raw, persist_vars, data_lists, trace_files, out_dir
    )