# #### raw graphs
#
# from pathlib import Path
# from GraphGeneration.graphs import plot_traces
#
# # script is at: CPS---Timed-automata/src/output_graphs.py
# # parent       = src/
# # parent.parent = CPS---Timed-automata/
# BASE_DIR = Path(__file__).resolve().parent.parent
#
# rooms   = ["A", "B", "C", "D", "E", "F"]
# periods = ["1day", "7day", "14day", "30day"]
#
# for period in periods:
#     for room in rooms:
#         plot_traces(
#             trace_folder  = BASE_DIR / "Data" / "3-ExtractInterval" / f"{period}-experiment" / f"room{room}",
#             output_folder = BASE_DIR / "Data" / "Graphs" / "interval_traces" / period / f"room{room}"
#         )
#
# ####

from pathlib import Path
from GraphGeneration.graphs import plot_discretized_traces, plot_traces
from DataProcessing.processData import get_trace_files
from Discretization.discretizationSetup import csv_to_temp_time_list
from Discretization.naive import equal_width_discretization
from Discretization.persist import Persist, get_best_bins, flatten_traces_to_ts, discretize_traces_with_bins
from Discretization.sax import sax_discretization_multi
import string

import numpy as np

def make_mapping(k):
    return {i: string.ascii_lowercase[i] for i in range(k)}

BASE_DIR = Path(__file__).resolve().parent.parent

rooms   = ["A", "B", "C", "D", "E", "F"]
periods = ["1day", "7day", "30day"]
symbols = 11
w       = 200  # SAX PAA segments

# for period in periods:
#     for room in rooms:
#         experiment_folder = BASE_DIR / "Data" / "3-ExtractInterval" / f"{period}-experiment" / f"room{room}"
#         raw_traces = get_trace_files(folder_path=experiment_folder)
#         data_lists = csv_to_temp_time_list(input_files=raw_traces)
#
#         # --- Naive ---
#         traces, bins = equal_width_discretization(data_lists, symbols)
#         plot_discretized_traces(
#             discretized_traces=traces,
#             output_folder=BASE_DIR / "Data" / "Graphs" / "discretized" / "naive" / period / room,
#             mapping=make_mapping(symbols)
#         )
#
#         # --- Persist ---
#         ts = flatten_traces_to_ts(data_lists)
#         p = Persist(x=ts, break_min=2, break_max=16, divergence="w", candidates="EW", skip=np.array([4, 4]))
#         bins = get_best_bins(p, ts)
#         traces = discretize_traces_with_bins(data_lists, bins)
#         persist_symbols = len(bins) - 1
#         plot_discretized_traces(
#             discretized_traces=traces,
#             output_folder=BASE_DIR / "Data" / "Graphs" / "discretized" / "persist" / period / room,
#             mapping=make_mapping(persist_symbols)
#         )
#
#         # --- SAX ---
#         w = 200
#         traces, breakpoints = sax_discretization_multi(data_lists, w=w, k=symbols)
#         plot_discretized_traces(
#             discretized_traces=traces,
#             output_folder=BASE_DIR / "Data" / "Graphs" / "discretized" / "sax" / period / room,
#             mapping=make_mapping(symbols)
#         )


for period in periods:
    for room in rooms:
        experiment_folder = BASE_DIR / "Data" / "3-ExtractInterval" / f"{period}-experiment" / f"room{room}"
        raw_traces = get_trace_files(folder_path=experiment_folder)

        if not raw_traces:
            print(f"No traces found for room {room} / {period}, skipping.")
            continue

        data_lists = csv_to_temp_time_list(input_files=raw_traces)

        # --- Naive (per-trace) ---
        for i, trace in enumerate(data_lists):
            traces, bins = equal_width_discretization([trace], symbols)
            plot_discretized_traces(
                discretized_traces=traces,
                output_folder=BASE_DIR / "Data" / "Graphs" / "discretized" / "naive (11 individual bins)" / period / room / f"trace_{i}",
                 mapping=make_mapping(symbols)
             )

        # # --- Persist (per-trace) ---
        # for i, trace in enumerate(data_lists):
        #     ts = flatten_traces_to_ts([trace])
        #     p = Persist(
        #         x=ts,
        #         break_min=2,
        #         break_max=16,
        #         divergence="w",
        #         candidates="EW",
        #         skip=np.array([4, 4])
        #     )
        #     bins = get_best_bins(p, ts)
        #     traces = discretize_traces_with_bins([trace], bins)
        #     persist_symbols = len(bins) - 1
        #     plot_discretized_traces(
        #         discretized_traces=traces,
        #         output_folder=BASE_DIR / "Data" / "Graphs" / "discretized" / "persist (individual bins(11))" / period / room / f"trace_{i}",
        #         mapping=make_mapping(persist_symbols)
        #     )

        print(f"Done: room {room} / {period}")