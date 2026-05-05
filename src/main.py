import json, os
from TAG.TALearner import TALearner
import graphviz
from pathlib import Path
import time

from GraphGeneration.graphs import plot_discretized_traces

from Discretization.discretizationSetup import (
    csv_to_temp_time_list,
    format_output,
    map_bins_to_symbols
)
from Discretization.naive import equal_width_discretization

from DataProcessing.processData import (
    format_temperature_data,
    extract_time_intervals,
    get_trace_files
)

total_start = time.perf_counter()

BASE_DIR = Path(__file__).resolve().parent.parent

# PARAMETERS
room = "A"
discretization_method = "naiv"
period = "1day"

symbols = 5

k_min = 2
k_max = 4
k_increment = 2

# Paths
experiment_folder = (
        BASE_DIR / "Data" / "3-ExtractInterval" / f"{period}-experiment" / f"room{room}"
)

discretinize_data_path = (
        BASE_DIR / "Data" / "4-DiscretizationData" / discretization_method / period
        / f"{room}-200traces-{period}-{discretization_method}-s{symbols}.txt"
)

TA_output_path = (
        BASE_DIR / "Data" / "5-TaResults" / discretization_method / period
)

# Load traces
all_traces = get_trace_files(folder_path=experiment_folder)
rawTraces = all_traces[:200]

print(f"Using {len(rawTraces)} traces")

# Preprocess once
data_lists = csv_to_temp_time_list(input_files=rawTraces)

# Discretize once
traces, bins = equal_width_discretization(data_lists, symbols)

# Map to symbols
symbolic_trace, symbol_map, mapping = map_bins_to_symbols(traces, symbols, bins)

# Save for TAG
format_output(symbolic_trace, discretinize_data_path)

# Run TAG for different k
for k in range(k_min, k_max + 1, k_increment):
    start = time.perf_counter()

    title = f"{room}-200traces-{period}-{discretization_method}-s{symbols}-k{k}"

    xml_path = (
            BASE_DIR / "Data" / "6-XMLOutput" / discretization_method / period
            / f"{room}-200traces-{period}-{discretization_method}-s{symbols}-k{k}.xml"
    )

    learner = TALearner(
        tss_path=discretinize_data_path,
        display=False,
        k=k
    )

    learner.ta.show(title=title, savePng=True, output_path=TA_output_path)
    learner.ta.export_ta(path=xml_path, symbol_map=symbol_map)

    end = time.perf_counter()

    print(f"Done: k={k} | time={end - start:.2f}s")
total_end = time.perf_counter()
print(f"Total runtime: {total_end - total_start:.2f}s")