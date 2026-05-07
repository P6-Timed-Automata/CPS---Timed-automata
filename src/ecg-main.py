import json, os
import time
from pathlib import Path

import numpy as np
import graphviz

from TAG.TALearner import TALearner
from GraphGeneration.graphs import plot_discretized_traces

from Discretization.discretizationSetup import (
    csv_to_temp_time_list,
    format_output,
    map_bins_to_symbols
)
from Discretization.naive import equal_width_discretization

from DataProcessing.processData import get_trace_files


# -----------------------
# TIMER
# -----------------------
total_start = time.perf_counter()

BASE_DIR = Path(__file__).resolve().parent.parent


# -----------------------
# PARAMETERS
# -----------------------
room = "train"
period = "ecg"
discretization_method = "naiv"

symbols = 5

k_min = 2
k_max = 4
k_increment = 2


# -----------------------
# PATHS
# -----------------------
if period == "ecg":
    experiment_folder = BASE_DIR / "Data" / "3-ExtractInterval" / "ecg-experimenrt" / room
else:
    experiment_folder = BASE_DIR / "Data" / "3-ExtractInterval" / f"{period}-experiment" / room


discretize_data_path = (
    BASE_DIR / "Data" / "4-DiscretizationData" / discretization_method / period
    / f"{room}-200traces-{period}-{discretization_method}-s{symbols}.txt"
)

ta_output_path = (
    BASE_DIR / "Data" / "5-TaResults" / discretization_method / period
)


# -----------------------
# LOAD DATA
# -----------------------
all_traces = get_trace_files(folder_path=experiment_folder)
raw_traces = all_traces[:200]

print(f"Using {len(raw_traces)} traces")


# -----------------------
# ECG PROCESSING
# -----------------------
if period == "ecg":

    all_rr = []
    ecg_traces = []

    min_rr_ms = 300  # physiological constraint

    for trace_file in raw_traces:

        data = np.genfromtxt(trace_file, delimiter=';', skip_header=1)

        # raw input is assumed ms → convert to seconds
        times = data[:, 0] / 1000.0
        values = data[:, 1]

        # signal preprocessing
        signal = values - np.median(values)
        abs_sig = np.abs(signal)

        threshold = np.mean(abs_sig) + 0.5 * np.std(abs_sig)

        # candidate peaks
        candidates = np.where(
            (abs_sig[1:-1] > abs_sig[:-2]) &
            (abs_sig[1:-1] >= abs_sig[2:]) &
            (abs_sig[1:-1] > threshold)
        )[0] + 1

        if len(candidates) < 2:
            continue

        # enforce RR constraint
        peaks = [candidates[0]]
        for idx in candidates[1:]:
            if times[idx] - times[peaks[-1]] >= (min_rr_ms / 1000.0):
                peaks.append(idx)

        if len(peaks) < 3:
            continue

        peak_times = times[np.array(peaks)]

        # RR in SECONDS → convert to MILLISECONDS (IMPORTANT FIX)
        rr = np.diff(peak_times) * 1000.0

        cum_time = np.concatenate([[0], np.cumsum(rr)])

        ecg_traces.append((rr, cum_time))
        all_rr.extend(rr)

    all_rr = np.array(all_rr)

    if len(all_rr) == 0:
        raise ValueError("No RR intervals found. Check preprocessing.")

    bins = np.linspace(all_rr.min(), all_rr.max(), symbols + 1)

    traces = []
    for rr, cum_time in ecg_traces:

        labels = np.clip(
            np.digitize(rr, bins) - 1,
            0,
            symbols - 1
        )

        trace = [
            (int(label), float(t))
            for label, t in zip(labels, cum_time)
        ]

        traces.append(trace)

else:
    # -----------------------
    # NON-ECG PATH
    # -----------------------
    data_lists = csv_to_temp_time_list(input_files=raw_traces)
    traces, bins = equal_width_discretization(data_lists, symbols)


# -----------------------
# SYMBOL MAPPING
# -----------------------
symbolic_trace, symbol_map, mapping = map_bins_to_symbols(
    traces, symbols, bins
)

print("SYMBOL MAP:", symbol_map)
print("UNIQUE LABELS IN TRACE:", set(l for trace in traces for l, _ in trace))


# -----------------------
# SAVE FOR TAG
# -----------------------
format_output(symbolic_trace, discretize_data_path)

print("NUM TRACES:", len(traces))
print("FIRST TRACE LENGTH:", len(traces[0]) if traces else 0)


# -----------------------
# RUN TAG LEARNER
# -----------------------
for k in range(k_min, k_max + 1, k_increment):

    start = time.perf_counter()

    title = f"{room}-200traces-{period}-{discretization_method}-s{symbols}-k{k}"

    xml_path = (
        BASE_DIR / "Data" / "6-XMLOutput" / discretization_method / period
        / f"{room}-200traces-{period}-{discretization_method}-s{symbols}-k{k}.xml"
    )

    learner = TALearner(
        tss_path=discretize_data_path,
        display=False,
        k=k
    )

    learner.ta.show(
        title=title,
        savePng=True,
        output_path=ta_output_path
    )

    learner.ta.export_ta(
        path=xml_path,
        symbol_map=symbol_map
    )

    end = time.perf_counter()
    print(f"Done: k={k} | time={end - start:.2f}s")


# -----------------------
# TOTAL TIME
# -----------------------
total_end = time.perf_counter()
print(f"Total runtime: {total_end - total_start:.2f}s")