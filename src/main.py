import json, os
from TAG.TALearner import TALearner

from Discretization.discretizationSetup import (
    csv_to_temp_time_list,
    format_output,
    map_bins_to_symbols
)
from Discretization.naive import equal_width_discretization

#Process Data
k = 3
input_files = [
    'DataProcessing/formated_data.csv',
    'DataProcessing/formated_data2.csv'
]

data_lists = csv_to_temp_time_list(input_files)
# print(len(data_lists))
# print(data_lists)


traces, bins = equal_width_discretization(data_lists, k)
# print(len(traces))
# print(traces)


symbolic_trace, symbol_map, mapping = map_bins_to_symbols(traces, k, bins)
# print(len(symbolic_trace))
# print(symbolic_trace)
# print(symbol_map)


format_output(symbolic_trace)


# Call TAG
tss_path = 'Discretization/output.txt'
xml_path = 'output/model.xml'

# with open('Discretization/symbol_map.json') as f:
#     symbol_map = json.load(f)

learner = TALearner(tss_path=tss_path, display=True)
learner.ta.show()

os.makedirs('output', exist_ok=True)
learner.ta.export_ta(xml_path, symbol_map=symbol_map)
print(f"UPPAAL model written to {xml_path}")



