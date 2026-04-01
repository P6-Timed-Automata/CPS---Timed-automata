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
    extract_time_intervals
)

# input_formated_raw_data = '../Data/FormatedRawData/formated_raw_data.csv'
# output_path_interval_data = '../Data/ExtractInterval'
# # Process Data
# # Full 24-hour traces, one per day
# extract_time_intervals(input_formated_raw_data, os.path.join(output_path_interval_data, "experiment_1_full_days"), "trace")
#
# # Full 1-hour traces, one per day
# extract_time_intervals(input_formated_raw_data, os.path.join(output_path_interval_data, "experiment_2_daily_windowed"), "trace", trace_days=1, window=(0, 3600) )
#
#
# # 7-day traces
# extract_time_intervals(input_formated_raw_data, os.path.join(output_path_interval_data, "experiment_3_weekly"), "trace", trace_days=7)
#
# # First 5 hours of each day, grouped into weekly traces
# extract_time_intervals(input_formated_raw_data, os.path.join(output_path_interval_data, "experiment_4_weekly_windowed"), "trace", trace_days=7, window=(0, 18000))


#Prepare Data for TAG
k = 3
input_files = [
    #'DataProcessing/formated_data.csv',
    #'DataProcessing/formated_data2.csv'
    '../Data/ExtractInterval/experiment_1_full_days/trace_trace1.csv'
]

data_lists = csv_to_temp_time_list(input_files)
# print(len(data_lists))
# print(data_lists)


# Discretenize

traces, bins = equal_width_discretization(data_lists, k)
# print(len(traces))
# print(traces)


symbolic_trace, symbol_map, mapping = map_bins_to_symbols(traces, k, bins)
# print(len(symbolic_trace))
# print(symbolic_trace)
# print(symbol_map)

tss_path = '../Data/DiscretizationData/trace1/output.txt'
format_output(symbolic_trace, tss_path)


# Call TAG
#tss_path = 'Discretization/output.txt'
#xml_path = '../Data/XMLOutput/model.xml'

# with open('Discretization/symbol_map.json') as f:
#     symbol_map = json.load(f)

learner = TALearner(tss_path=tss_path, display=True)

title = "Final Automaton.txt"
TA_output_path = os.path.join("Data", "TaResults", title)
os.makedirs(TA_output_path, exist_ok=True)

learner.ta.show(title = title, savePng = True, output_path = TA_output_path)

# os.makedirs('output', exist_ok=True)

# XML_output_path = os.path.join("Data", "XMLOutput", "model.xml")
# os.makedirs(XML_output_path, exist_ok=True)


XML_output_path = "Data/XMLOutput/model.xml"
os.makedirs(os.path.dirname(XML_output_path), exist_ok=True)
learner.ta.export_ta(path=XML_output_path, symbol_map=symbol_map)
print(f"UPPAAL model written to {XML_output_path}")



