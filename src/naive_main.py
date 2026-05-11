import json, os
from TAG.TALearner import TALearner
import graphviz
from pathlib import Path
import random
import numpy as np


from Discretization.discretizationSetup import (
    csv_to_temp_time_list,
    format_output,
    map_bins_to_symbols,
    preprocess_test_traces
)
from Discretization.naive import (
    equal_width_discretization
)

from DataProcessing.processData import (
    get_trace_files
)

from DataProcessing.negative_samples_production import (
    generate_negative_samples,
    plot_and_save_traces
)



BASE_DIR = Path(__file__).resolve().parent.parent




# PARAMETERS SETTINGS
room = "A"
discretization_method = "naiv"
period = "1day"

# Parameter for Naiv
symbols = 25

# Parameter for TAG
k_min = 4
k_max = 4
k_increment = 2


#Prepare train traces
train_folder = BASE_DIR / "Data" / "3-ExtractInterval" / f"{period}-experiment"/ f"{room}-train"
train_raw_traces = get_trace_files(folder_path = train_folder)
train_raw_lists = csv_to_temp_time_list(input_files=train_raw_traces)
train_traces, bins = equal_width_discretization(train_raw_lists, symbols)
symbolic_train_trace, symbol_map, mapping = map_bins_to_symbols(train_traces, symbols, bins)


#Prepare test traces (positive and negative samples)
test_positive_folder = BASE_DIR / "Data" / "3-ExtractInterval" / f"{period}-experiment"/f"{room}-test/positive"
test_negative_folder = BASE_DIR / "Data" / "3-ExtractInterval" / f"{period}-experiment"/f"{room}-test/negative"
test_positive_raw_traces = get_trace_files(folder_path = test_positive_folder)
test_negative_raw_traces = get_trace_files(folder_path = test_negative_folder)

test_positive_raw_lists = csv_to_temp_time_list(input_files=test_positive_raw_traces)
test_negative_raw_lists = csv_to_temp_time_list(input_files=test_negative_raw_traces)


test_positive_traces_lists = preprocess_test_traces(test_traces = test_positive_raw_lists, bins = bins, s = symbols)
test_negative_traces_lists = preprocess_test_traces(test_traces = test_negative_raw_lists, bins = bins, s = symbols)


# Plot negative samples
negative_graph =  BASE_DIR / "Data" / "Graphs" / "Observed-negative-samples" / f"{discretization_method}-{period}-s{symbols}-experiment"
title_prefix = f"{discretization_method}-{period}-s{symbols}"

# plot_and_save_traces(
#     traces=test_negative_traces_lists,
#     positive_traces=test_positive_traces_lists,
#     output_folder=negative_graph,
#     symbol_map=symbol_map,
#     title_prefix = title_prefix
# )

#Path to log data
log_data_path = BASE_DIR /"Data" /"8-LoggedData" / f"{discretization_method}-log.csv"


# Parameter for nr of Traces
len_traces = len(train_raw_traces)  + 1
start_traces = 1
len_traces = 2

for trace_nr in range(start_traces, len_traces):
    # Paths
    discretinize_data_path = (BASE_DIR/ "Data"/ "4-DiscretizationData"/ discretization_method / period
                              / f"{room}-{trace_nr}trace-{period}-{discretization_method}-s{symbols}-trace.txt"
                              )

    symbolic_train_trace_subset = symbolic_train_trace[:trace_nr ]

    format_output(symbolic_res_list=symbolic_train_trace_subset, output_path=discretinize_data_path)


    # Loop over varying K-future
    for k in range(k_min, k_max + 1, k_increment):

        #Prepare Paths
        title = f"{room}-{trace_nr}trace-{period}-{discretization_method}-s{symbols}-k{k}-ta"
        TA_output_path = (BASE_DIR / "Data" / "5-TaResults" / discretization_method / period)
        xml_path = (BASE_DIR / "Data" / "6-XMLOutput" / discretization_method / period
                    / f"{room}-{trace_nr}trace-{period}-{discretization_method}-s{symbols}-k{k}.xml")
        run_id = f"{period}-{room}-{trace_nr}trace-s{symbols}-k{k}"

        # Tranform to TA
        learner = TALearner(tss_path=discretinize_data_path,display=False,k=k )
        learner.ta.show(title=title,savePng=True,output_path=TA_output_path)
        learner.ta.export_ta(path=xml_path, symbol_map=symbol_map)

        # Compute metrics
        metrics = learner.ta.evaluate_classifier(positive_tss = test_positive_traces_lists, negative_tss = test_negative_traces_lists,  save_path = log_data_path, run_id= run_id, timed=True)

        print(f"Done: trace:{trace_nr}, k:{k}, symbols={symbols}, Positive Acceptance Rate: {metrics['PAR']:.2f}%, Negative Acceptance Rate: {metrics['NAR']:.2f}% ")
        print("-------------------------------------------------------------------------------------")




