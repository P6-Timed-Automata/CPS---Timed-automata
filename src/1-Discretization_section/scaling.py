# # scaling_experiment.py
#
# import time
# import os
# import numpy as np
# import matplotlib.pyplot as plt
# from pathlib import Path
#
# from DataProcessing.processData import get_trace_files
# from Discretization.discretizationSetup import csv_to_temp_time_list, format_output, map_bins_to_symbols
# from Discretization.naive import equal_width_discretization
# from Discretization.sax import sax_discretization_multi
# from Discretization.persist import Persist, get_best_bins, discretize_traces_with_bins, flatten_traces_to_ts
# from TAG.TALearner import TALearner
#
#
# # ---------------------------------------------------------------------------
# # HELPERS
# # ---------------------------------------------------------------------------
#
# def _discretize(method_type: str, params: dict, subset_data: list):
#     if method_type == "naive":
#         return equal_width_discretization(subset_data, k=params['bins'])
#
#     elif method_type == "sax":
#         traces, bins_z, mean_, std_ = sax_discretization_multi(
#             subset_data, w=params['w'], k=params['bins']
#         )
#         bins_c = np.sort(bins_z) * std_ + mean_
#         return traces, bins_c
#
#     elif method_type == "persist":
#         ts = flatten_traces_to_ts(subset_data)
#         persist_obj = Persist(ts, break_min=2, break_max=params['bins'], skip=np.array([4, 4]))
#         bins_c = get_best_bins(persist_obj, ts)
#         traces = discretize_traces_with_bins(subset_data, bins_c)
#         return traces, bins_c
#
#     else:
#         raise ValueError(f"Unknown method: {method_type}")
#
#
# def check_traces(folder: Path, required: int) -> None:
#     available = sorted(get_trace_files(folder_path=str(folder)))
#     print(f"Found {len(available)} trace files in:\n  {folder}")
#     if len(available) < required:
#         raise RuntimeError(
#             f"Need at least {required} traces, but only {len(available)} found."
#         )
#     print(f"Using first {required}.\n")
#
#
# # ---------------------------------------------------------------------------
# # SINGLE EXPERIMENT  — returns data for combined plot
# # ---------------------------------------------------------------------------
#
# def run_scaling_experiment(
#         all_data: list,
#         method_type: str,
#         params: dict,
#         output_folder: str,
#         max_traces: int = 20,
#         repeats: int = 5,
#         tag_k: int = 2) -> dict:
#     """
#     Runs one scaling experiment and saves its individual plot.
#     Returns a result dict for use in the combined plot.
#     """
#     os.makedirs(output_folder, exist_ok=True)
#     tmp_file = os.path.join(os.getcwd(), "_tmp_scaling_input.txt")
#
#     trace_counts = list(range(1, max_traces + 1))
#     means, stds  = [], []
#
#     for n in trace_counts:
#         subset = all_data[:n]
#
#         traces, bins_c = _discretize(method_type, params, subset)
#         actual_bins    = len(bins_c) - 1
#         symbolic_traces, _, _ = map_bins_to_symbols(traces, actual_bins, bins_c)
#         format_output(symbolic_traces, tmp_file)
#
#         run_times = []
#         for _ in range(repeats):
#             t0 = time.perf_counter()
#             TALearner(tmp_file, display=False, k=tag_k)
#             run_times.append(time.perf_counter() - t0)
#
#         m = float(np.mean(run_times))
#         s = float(np.std(run_times))
#         means.append(m)
#         stds.append(s)
#         print(f"  n={n:2d}  mean={m:.3f}s  std={s:.3f}s", flush=True)
#
#     if os.path.exists(tmp_file):
#         os.remove(tmp_file)
#
#     # --- Individual plot ---
#     label_str = "_".join(f"{k}{v}" for k, v in params.items())
#     _save_single_plot(
#         trace_counts, means, stds,
#         title=f"TA Learning Scaling — {method_type.upper()} (Mean of {repeats} Runs)",
#         out_path=os.path.join(output_folder, f"TA_Scaling_{method_type}_{label_str}.png"),
#     )
#
#     return {
#         'label':         f"{method_type.upper()} ({label_str})",
#         'trace_counts':  trace_counts,
#         'means':         means,
#         'stds':          stds,
#     }
#
#
# def _save_single_plot(trace_counts, means, stds, title, out_path):
#     fig, ax = plt.subplots(figsize=(10, 5))
#     ax.errorbar(
#         trace_counts, means, yerr=stds,
#         fmt='o-', color='navy', ecolor='navy',
#         elinewidth=1.5, capsize=6, capthick=1.5,
#         linewidth=2, markersize=5, label='Mean Learning Time',
#     )
#     ax.set_xlabel("Number of Input Traces (Training Set Size)")
#     ax.set_ylabel("Time (seconds)")
#     ax.set_title(title)
#     ax.set_xticks(trace_counts)
#     ax.grid(True, linestyle='--', alpha=0.5)
#     ax.legend()
#     fig.tight_layout()
#     fig.savefig(out_path, bbox_inches='tight', dpi=300)
#     plt.close(fig)
#     print(f"Saved → {out_path}")
#
#
# # ---------------------------------------------------------------------------
# # COMBINED PLOT
# # ---------------------------------------------------------------------------
#
# # One distinct color per line
# _COLORS = ['navy', 'firebrick', 'forestgreen', 'darkorange', 'purple', 'teal']
#
# def plot_combined_scaling(results: list, output_folder: str, repeats: int) -> None:
#     """
#     Plots all experiment results on one axes with one line per experiment.
#
#     Args:
#         results:       List of dicts returned by run_scaling_experiment.
#         output_folder: Where to save the PNG.
#         repeats:       Used only for the title string.
#     """
#     fig, ax = plt.subplots(figsize=(12, 6))
#
#     for result, color in zip(results, _COLORS):
#         ax.errorbar(
#             result['trace_counts'],
#             result['means'],
#             yerr=result['stds'],
#             fmt='o-',
#             color=color,
#             ecolor=color,
#             elinewidth=1.2,
#             capsize=5,
#             capthick=1.2,
#             linewidth=2,
#             markersize=4,
#             label=result['label'],
#         )
#
#     ax.set_xlabel("Number of Input Traces (Training Set Size)")
#     ax.set_ylabel("Time (seconds)")
#     ax.set_title(f"TA Learning Scaling — All Methods (Mean of {repeats} Runs)")
#     ax.set_xticks(results[0]['trace_counts'])
#     ax.grid(True, linestyle='--', alpha=0.5)
#     ax.legend()
#     fig.tight_layout()
#
#     out_path = os.path.join(output_folder, "TA_Scaling_combined.png")
#     fig.savefig(out_path, bbox_inches='tight', dpi=300)
#     plt.close(fig)
#     print(f"Saved → {out_path}")
#
#
# # ---------------------------------------------------------------------------
# # MAIN
# # ---------------------------------------------------------------------------
#
# if __name__ == "__main__":
#
#     BASE_DIR     = Path(__file__).resolve().parent.parent.parent
#    # TRACE_FOLDER = BASE_DIR / "Data" / "3-ExtractInterval" / "1day-experiment" / "roomA"
#     TRACE_FOLDER = BASE_DIR / "Data" / "synthetic_data" / "clean_train"
#     OUT_DIR      = BASE_DIR / "Data" / "Graphs" / "TA_Scaling-5-100-5-10-noRepeat"
#
#     MAX_TRACES = 50
#     REPEATS    = 5
#     TAG_K      = 2
#
#     EXPERIMENTS = [
#         ("naive",   {'bins': 5}),
#         ("sax",     {'w': 288, 'bins': 5}),
#         #("persist", {'bins': 5}),
#     ]
#
#     check_traces(TRACE_FOLDER, MAX_TRACES)
#
#     # Load once, reuse across all experiments
#     all_files = sorted(get_trace_files(folder_path=str(TRACE_FOLDER)))[:MAX_TRACES]
#     all_data  = csv_to_temp_time_list(input_files=all_files)
#
#     all_results = []
#
#     for method_type, params in EXPERIMENTS:
#         print(f"{'=' * 60}")
#         print(f"  Method : {method_type.upper()}")
#         print(f"  Params : {params}")
#         print(f"  Traces : 1–{MAX_TRACES}  |  Repeats : {REPEATS}  |  k : {TAG_K}")
#         print(f"{'=' * 60}")
#
#         result = run_scaling_experiment(
#             all_data      = all_data,
#             method_type   = method_type,
#             params        = params,
#             output_folder = str(OUT_DIR),
#             max_traces    = MAX_TRACES,
#             repeats       = REPEATS,
#             tag_k         = TAG_K,
#         )
#         all_results.append(result)
#         print()
#
#     plot_combined_scaling(all_results, str(OUT_DIR), REPEATS)
#     print("All experiments done.")

# scaling_experiment.py

import time
import os
import csv
import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path
from datetime import datetime

from DataProcessing.processData import get_trace_files
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


# =============================================================================
# HELPERS
# =============================================================================

def _discretize(method_type: str, params: dict, subset_data: list):

    if method_type == "naive":
        return equal_width_discretization(
            subset_data,
            k=params['bins']
        )

    elif method_type == "sax":

        traces, bins_z, mean_, std_ = sax_discretization_multi(
            subset_data,
            w=params['w'],
            k=params['bins']
        )

        bins_c = np.sort(bins_z) * std_ + mean_

        return traces, bins_c

    elif method_type == "persist":

        ts = flatten_traces_to_ts(subset_data)

        persist_obj = Persist(
            ts,
            break_min=2,
            break_max=params['bins'],
            skip=np.array([4, 4])
        )

        bins_c = get_best_bins(persist_obj, ts)

        traces = discretize_traces_with_bins(
            subset_data,
            bins_c
        )

        return traces, bins_c

    else:
        raise ValueError(f"Unknown method: {method_type}")


def check_traces(folder: Path, required: int) -> None:

    available = sorted(
        get_trace_files(folder_path=str(folder))
    )

    print(f"Found {len(available)} trace files in:\n  {folder}")

    if len(available) < required:
        raise RuntimeError(
            f"Need at least {required} traces, but only {len(available)} found."
        )

    print(f"Using first {required}.\n")


# =============================================================================
# CSV LOGGING
# =============================================================================

def append_scaling_log(
        log_path: str,
        dataset_name: str,
        method_type: str,
        params: dict,
        trace_count: int,
        repeat_id: int,
        runtime: float,
        actual_bins: int,
        tag_k: int):

    file_exists = os.path.exists(log_path)

    with open(log_path, 'a', newline='') as f:

        writer = csv.writer(f)

        if not file_exists:
            writer.writerow([
                "timestamp",
                "dataset",
                "method",
                "params",
                "trace_count",
                "repeat",
                "runtime_seconds",
                "actual_bins",
                "tag_k"
            ])

        writer.writerow([
            datetime.now().isoformat(),
            dataset_name,
            method_type,
            str(params),
            trace_count,
            repeat_id,
            runtime,
            actual_bins,
            tag_k
        ])


# =============================================================================
# SINGLE EXPERIMENT
# =============================================================================

def run_scaling_experiment(
        dataset_name: str,
        all_data: list,
        method_type: str,
        params: dict,
        output_folder: str,
        log_path: str,
        max_traces: int = 20,
        repeats: int = 5,
        tag_k: int = 2) -> dict:

    os.makedirs(output_folder, exist_ok=True)

    tmp_file = os.path.join(
        os.getcwd(),
        "_tmp_scaling_input.txt"
    )

    trace_counts = list(range(1, max_traces + 1))

    means = []
    stds = []

    for n in trace_counts:

        subset = all_data[:n]

        traces, bins_c = _discretize(
            method_type,
            params,
            subset
        )

        actual_bins = len(bins_c) - 1

        symbolic_traces, _, _ = map_bins_to_symbols(
            traces,
            actual_bins,
            bins_c
        )

        format_output(
            symbolic_traces,
            tmp_file
        )

        run_times = []

        for repeat_id in range(repeats):

            t0 = time.perf_counter()

            TALearner(
                tmp_file,
                display=False,
                k=tag_k
            )

            runtime = time.perf_counter() - t0

            run_times.append(runtime)

            append_scaling_log(
                log_path     = log_path,
                dataset_name = dataset_name,
                method_type  = method_type,
                params       = params,
                trace_count  = n,
                repeat_id    = repeat_id,
                runtime      = runtime,
                actual_bins  = actual_bins,
                tag_k        = tag_k
            )

        m = float(np.mean(run_times))
        s = float(np.std(run_times))

        means.append(m)
        stds.append(s)

        print(
            f"  n={n:2d}  mean={m:.3f}s  std={s:.3f}s",
            flush=True
        )

    if os.path.exists(tmp_file):
        os.remove(tmp_file)

    label_str = "_".join(
        f"{k}{v}" for k, v in params.items()
    )

    _save_single_plot(
        trace_counts = trace_counts,
        means        = means,
        stds         = stds,
        title        = (
            f"{dataset_name} — "
            f"{method_type.upper()} "
            f"(Mean of {repeats} Runs)"
        ),
        out_path     = os.path.join(
            output_folder,
            f"{dataset_name}_{method_type}_{label_str}.png"
        ),
    )

    return {
        'dataset':      dataset_name,
        'label':        f"{dataset_name} — {method_type.upper()} ({label_str})",
        'trace_counts': trace_counts,
        'means':        means,
        'stds':         stds,
    }


# =============================================================================
# PLOTTING
# =============================================================================

_COLORS = [
    'navy',
    'firebrick',
    'forestgreen',
    'darkorange',
    'purple',
    'teal',
    'brown',
]


def _save_single_plot(
        trace_counts,
        means,
        stds,
        title,
        out_path):

    fig, ax = plt.subplots(figsize=(10, 5))

    ax.errorbar(
        trace_counts,
        means,
        yerr=stds,
        fmt='o-',
        linewidth=2,
        markersize=5,
        capsize=5,
    )

    ax.set_xlabel("Training Trace Count")
    ax.set_ylabel("Learning Time (seconds)")
    ax.set_title(title)

    ax.grid(True, linestyle='--', alpha=0.5)

    fig.tight_layout()

    fig.savefig(
        out_path,
        bbox_inches='tight',
        dpi=300
    )

    plt.close(fig)

    print(f"Saved → {out_path}")


def plot_combined_scaling(
        results: list,
        output_folder: str,
        repeats: int):

    fig, ax = plt.subplots(figsize=(12, 6))

    for result, color in zip(results, _COLORS):

        ax.errorbar(
            result['trace_counts'],
            result['means'],
            yerr=result['stds'],
            fmt='o-',
            color=color,
            ecolor=color,
            linewidth=2,
            markersize=4,
            capsize=5,
            label=result['label'],
        )

    ax.set_xlabel("Training Trace Count")
    ax.set_ylabel("Learning Time (seconds)")

    ax.set_title(
        f"TA Learning Scaling "
        f"(Mean of {repeats} Runs)"
    )

    ax.grid(True, linestyle='--', alpha=0.5)

    ax.legend()

    fig.tight_layout()

    out_path = os.path.join(
        output_folder,
        "combined_scaling.png"
    )

    fig.savefig(
        out_path,
        bbox_inches='tight',
        dpi=300
    )

    plt.close(fig)

    print(f"Saved → {out_path}")


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":

    BASE_DIR = Path(__file__).resolve().parent.parent.parent

    OUT_DIR = (
            BASE_DIR
            / "Data"
            / "Graphs"
            / "ScalingExperiments"
    )

    os.makedirs(OUT_DIR, exist_ok=True)

    LOG_PATH = OUT_DIR / "scaling_results.csv"

    MAX_TRACES = 30
    REPEATS = 5
    TAG_K = 2

    # -------------------------------------------------------------------------
    # DATASETS
    # -------------------------------------------------------------------------

    DATASETS = [

        (
            "clean",
            BASE_DIR / "Data" / "synthetic_data" / "clean_train"
        ),

        (
            "noisy",
            BASE_DIR / "Data" / "synthetic_data" / "noisy_train"
        ),

        # (
        #     "negative",
        #     BASE_DIR / "Data" / "synthetic_data" / "negative"
        # ),
    ]

    # -------------------------------------------------------------------------
    # EXPERIMENTS
    # -------------------------------------------------------------------------

    EXPERIMENTS = [

        ("naive", {
            'bins': 5
        }),

        ("sax", {
            'w': 288,
            'bins': 5
        }),

        # ("persist", {
        #     'bins': 5
        # }),
    ]

    # -------------------------------------------------------------------------
    # RUN
    # -------------------------------------------------------------------------

    all_results = []

    for dataset_name, trace_folder in DATASETS:

        print("=" * 70)
        print(f"DATASET: {dataset_name}")
        print("=" * 70)

        check_traces(
            trace_folder,
            MAX_TRACES
        )

        all_files = sorted(
            get_trace_files(
                folder_path=str(trace_folder)
            )
        )[:MAX_TRACES]

        all_data = csv_to_temp_time_list(
            input_files=all_files
        )

        for method_type, params in EXPERIMENTS:

            print(f"{'-' * 60}")
            print(f"Method : {method_type.upper()}")
            print(f"Params : {params}")
            print(f"Dataset: {dataset_name}")
            print(f"{'-' * 60}")

            result = run_scaling_experiment(
                dataset_name = dataset_name,
                all_data     = all_data,
                method_type  = method_type,
                params       = params,
                output_folder= str(OUT_DIR),
                log_path     = str(LOG_PATH),
                max_traces   = MAX_TRACES,
                repeats      = REPEATS,
                tag_k        = TAG_K,
            )

            all_results.append(result)

            print()

    plot_combined_scaling(
        all_results,
        str(OUT_DIR),
        REPEATS
    )

    print("\nAll experiments done.")