import json
import os
from TAG.TALearner import TALearner
import graphviz
from pathlib import Path

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

BASE_DIR = Path(__file__).resolve().parent.parent


# PARAMETERS SETTINGS
room = "ecg"
discretization_method = "naiv"
period = "ecg-train"

# Parameter for Naiv
symbols = 5

# Parameter for TAG
k_min = 4
k_max = 4
k_increment = 2

if period.startswith("ecg-"):
    split_name = period.replace("ecg-", "")
    experiment_folder = BASE_DIR / "Data" / \
        "3-ExtractInterval" / "ecg-experimenrt" / split_name
else:
    experiment_folder = BASE_DIR / "Data" / \
        "3-ExtractInterval" / f"{period}-experiment"

all_traces = get_trace_files(folder_path=experiment_folder)
if not all_traces:
    raise RuntimeError(f"No ECG trace files found in {experiment_folder}")

max_trace_files = 20
selected_traces = all_traces[:max_trace_files]

for trace_nr, input_file in enumerate(selected_traces, start=1):

    # Paths
    discretinize_data_path = (BASE_DIR / "Data" / "4-DiscretizationData" / discretization_method / period
                              / f"{room}-{trace_nr}trace-{period}-{discretization_method}-s{symbols}-trace.txt"
                              )

    # Prepare input for Naiv using a single input file per trace
    data_lists = csv_to_temp_time_list(input_files=[input_file])

    if not data_lists:
        print(f"Skipping trace {trace_nr}: no data to discretize")
        continue

    # Discretize with naiv
    traces, bins = equal_width_discretization(data_lists, symbols)

    # Prpare format for TAG
    symbolic_trace, symbol_map, mapping = map_bins_to_symbols(
        traces, symbols, bins)
    format_output(symbolic_res_list=symbolic_trace,
                  output_path=discretinize_data_path)

    # Now vary k
    for k in range(k_min, k_max + 1, k_increment):
        title = f"{room}-{trace_nr}trace-{period}-{discretization_method}-s{symbols}-k{k}-ta"
        TA_output_path = BASE_DIR / "Data" / \
            "5-TaResults" / discretization_method / period
        xml_path = (BASE_DIR / "Data" / "6-XMLOutput" / discretization_method / period
                    / f"{room}-{trace_nr}trace-{period}-{discretization_method}-s{symbols}-k{k}.xml")

        try:
            learner = TALearner(
                tss_path=discretinize_data_path, display=False, k=k)
            learner.ta.show(title=title, savePng=True,
                            output_path=TA_output_path)
            learner.ta.export_ta(path=xml_path, symbol_map=symbol_map)
            print(f"Done: trace={trace_nr}, k={k}, symbols={symbols}")
        except Exception as exc:
            print(
                f"WARNING: TA generation failed for trace={trace_nr}, k={k}: {exc}")
            continue

        print("-------------------------------------------------------------------------------------")
