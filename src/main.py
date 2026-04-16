import json, os
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
room = "A"
discretization_method = "naiv"
period = "1day"

# Parameter for Naiv
symbols = 15

# Parameter for TAG
k_min = 4
k_max = 4
k_increment = 1

experiment_folder = BASE_DIR / "Data" / "3-ExtractInterval" / f"{period}-experiment"

all_traces = get_trace_files(folder_path = experiment_folder)
len_traces = len(all_traces)  + 1

trace_nr = 4


# Paths
discretinize_data_path = (BASE_DIR/ "Data"/ "4-DiscretizationData"/ discretization_method / period
                          / f"{room}-{trace_nr}trace-{period}-{discretization_method}-s{symbols}-trace.txt"
                          )

discrete_graph_path = (
        BASE_DIR / "Data" / "Graphs" / "Discretized" / discretization_method / period
)

# Prepare input for Naiv
rawTraces = all_traces[:trace_nr]
data_lists = csv_to_temp_time_list(input_files=rawTraces)

# Discretize with naiv
traces, bins = equal_width_discretization(data_lists, symbols)

# Prpare format for TAG
symbolic_trace, symbol_map, mapping = map_bins_to_symbols(traces, symbols, bins)
format_output(symbolic_res_list=symbolic_trace, output_path=discretinize_data_path)

#plot discretized data
plot_discretized_traces(
    discretized_traces=traces,
    output_folder=discrete_graph_path,
    bins=bins,
    mapping=mapping
)

# Save symbolic output
format_output(symbolic_res_list=symbolic_trace, output_path=discretinize_data_path)

# Now vary k
for k in range(k_min, k_max + 1, k_increment):
    #Paths
    title = f"{room}-{trace_nr}trace-{period}-{discretization_method}-s{symbols}-k{k}-ta"
    TA_output_path = (BASE_DIR / "Data" / "5-TaResults" / discretization_method / period)
    xml_path = (BASE_DIR / "Data" / "6-XMLOutput" / discretization_method / period
                / f"{room}-{trace_nr}trace-{period}-{discretization_method}-s{symbols}-k{k}.xml")

    # Tranform to TA
    learner = TALearner(tss_path=discretinize_data_path,display=False,k=k)
    learner.ta.show(title=title,savePng=True,output_path=TA_output_path)
    learner.ta.export_ta(path=xml_path, symbol_map=symbol_map)

    print(f"Done: trace={trace_nr}, k={k}, symbols={symbols}")
    print("-------------------------------------------------------------------------------------")




#FOR CLUSTER

# for trace_nr in range(1, len_traces):
#
#     # Paths
#     discretinize_data_path = (BASE_DIR/ "Data"/ "4-DiscretizationData"/ discretization_method / period
#                               / f"{room}-{trace_nr}trace-{period}-{discretization_method}-s{symbols}-trace.txt"
#                               )
#
#     # Prepare input for Naiv
#     rawTraces = all_traces[:trace_nr]
#     data_lists = csv_to_temp_time_list(input_files=rawTraces)
#
#     # Discretize with naiv
#     traces, bins = equal_width_discretization(data_lists, symbols)
#
#     # Prpare format for TAG
#     symbolic_trace, symbol_map, mapping = map_bins_to_symbols(traces, symbols, bins)
#     format_output(symbolic_res_list=symbolic_trace, output_path=discretinize_data_path)
#
#     # Now vary k
#     for k in range(k_min, k_max + 1, k_increment):
#         #Paths
#         title = f"{room}-{trace_nr}trace-{period}-{discretization_method}-s{symbols}-k{k}-ta"
#         TA_output_path = (BASE_DIR / "Data" / "5-TaResults" / discretization_method / period)
#         xml_path = (BASE_DIR / "Data" / "6-XMLOutput" / discretization_method / period
#                     / f"{room}-{trace_nr}trace-{period}-{discretization_method}-s{symbols}-k{k}.xml")
#
#         # Tranform to TA
#         learner = TALearner(tss_path=discretinize_data_path,display=False,k=k)
#         learner.ta.show(title=title,savePng=True,output_path=TA_output_path)
#         learner.ta.export_ta(path=xml_path, symbol_map=symbol_map)
#
#         print(f"Done: trace={trace_nr}, k={k}, symbols={symbols}")
#         print("-------------------------------------------------------------------------------------")



