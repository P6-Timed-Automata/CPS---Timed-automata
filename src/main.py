import json, os
from TAG.TALearner import TALearner

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


# #Prepare Data for TAG
room = "A"
b = 15
#k = 6
discretization_method = "naiv"
period = "30day"
trace_nr = 1

experiment_folder = f"../Data/3-ExtractInterval/{period}-experiment"
for trace_nr in range(1, 11):  # 1 → 10 traces
    rawTraces = get_trace_files(folder_path=experiment_folder, max_files=trace_nr)

    data_lists = csv_to_temp_time_list(input_files=rawTraces)

    # Discretize (same for all k)
    traces, bins = equal_width_discretization(data_lists, b)
    symbolic_trace, symbol_map, mapping = map_bins_to_symbols(traces, b, bins)

    # Save discretized data ONCE per trace_nr
    discretinize_data_path = f"../Data/4-DiscretizationData/{discretization_method}/{period}/{room}-{trace_nr}trace-{period}-{discretization_method}-b{b}-trace.txt"

    format_output(symbolic_res_list=symbolic_trace, output_path=discretinize_data_path)

    # Now vary k
    for k in range(4, 11, 2):  # 6, 8, 10

        learner = TALearner(
            tss_path=discretinize_data_path,
            display=False,
            k=k
        )

        title = f"{room}-{trace_nr}trace-{period}-{discretization_method}-b{b}-k{k}-ta"
        TA_output_path = f"../Data/5-TaResults/{discretization_method}/{period}"

        learner.ta.show(
            title=title,
            savePng=True,
            output_path=TA_output_path
        )

        xml_path = f'../Data/6-XMLOutput/{discretization_method}/{period}/{room}-{trace_nr}trace-{period}-{discretization_method}-b{b}-k{k}.xml'

        learner.ta.export_ta(path=xml_path, symbol_map=symbol_map)

        print(f"Done: trace={trace_nr}, k={k}")
#
# rawTraces = get_trace_files(folder_path = experiment_folder, max_files=trace_nr)
#
# data_lists = csv_to_temp_time_list(input_files=rawTraces)
#
# # Discretenize
#
# traces, bins = equal_width_discretization(data_lists, b)
# symbolic_trace, symbol_map, mapping = map_bins_to_symbols(traces, b, bins)
#
# discretinize_data_path = f"../Data/4-DiscretizationData/{discretization_method}/{period}/{room}-{trace_nr}trace-{period}-{discretization_method}-b{b}-trace.txt"
# format_output(symbolic_res_list=symbolic_trace, output_path=discretinize_data_path)
#
#
# # Call TAG
# learner = TALearner(tss_path=discretinize_data_path, display=False, k=k)
#
# title = f"{room}-{trace_nr}trace-{period}-{discretization_method}-b{b}-k{k}-ta"
# TA_output_path = f"../Data/5-TaResults/{discretization_method}/{period}"
#
# learner.ta.show(title = title, savePng = True, output_path = TA_output_path)
#
# xml_path = f'../Data/6-XMLOutput/{discretization_method}/{period}/{room}-{trace_nr}trace-{period}-{discretization_method}-b{b}-k{k}.xml'
# learner.ta.export_ta(path=xml_path, symbol_map=symbol_map)
#
#
