import numpy as np
import os
import matplotlib.pyplot as plt
from TAG.TALearner import TALearner
from pathlib import Path

from Discretization.sax import (
    sax_discretization_multi,
    sax_discretization
)

from Discretization.discretizationSetup import (
    csv_to_temp_time_list,
    format_output,
    map_bins_to_symbols,
    preprocess_test_traces,
    sax_preprocess_traces
)

from DataProcessing.processData import (
    format_temperature_data,
    extract_time_intervals,
    get_trace_files
)

from pathlib import Path
import shutil


BASE_DIR = Path(__file__).resolve().parent.parent



# PARAMETERS SETTINGS
data_type ="ecg"
room = "A"
discretization_method = "sax"
sim_nr = 10000
period_nr = 1


if data_type == "temp":
    period = f"{period_nr}day"
    time = 86400
elif data_type == "ecg":
    period = "1beat"
    time = 275



# Parameter for SAX
symbols = 10
# w = 200
w_values = [24, 48]

# Parameter for TAG
k_min = 4
k_max = 4
k_increment = 2

#Prepare train traces
if (data_type == "temp"):
    train_folder = BASE_DIR / "Data" / "3-ExtractInterval" / f"{period}-experiment"/ f"{room}-train"
elif(data_type == "ecg"):
    train_folder = BASE_DIR / "Data" / "3-ExtractInterval" /"ecg" / f"{period}-experiment"/ f"{period}-train"

train_raw_traces = get_trace_files(folder_path = train_folder)
train_raw_lists = csv_to_temp_time_list(input_files=train_raw_traces)

# PATHS FOR NEGATIVE DATA
# test_negative_folder = (
#     BASE_DIR
#     / "Data"
#     / "3-ExtractInterval"
#     / f"{period}-experiment"
#     / f"{room}-test"
#     / "negative"
# )
#
# # Folder where all files will be collected
# combined_negative_folder = test_negative_folder / "combined"
# combined_negative_folder.mkdir(exist_ok=True)
#
# # ======================================================
# # CONCATENATE FILES FROM SUBFOLDERS
# # ======================================================
#
# for subfolder in test_negative_folder.iterdir():
#
#     # Skip if it's not a folder or if it's the output folder itself
#     if not subfolder.is_dir() or subfolder.name == "combined":
#         continue
#
#     for file in subfolder.iterdir():
#
#         if file.is_file():
#
#             # Optional: preserve unique filenames
#             destination = combined_negative_folder / f"{subfolder.name}_{file.name}"
#
#             shutil.copy(file, destination)
#
# print(f"All negative files copied to: {combined_negative_folder}")

#Prepare test traces (positive and negative samples)
if (data_type == "temp"):
    test_positive_folder = BASE_DIR / "Data" / "3-ExtractInterval" / f"{period}-experiment"/f"{room}-test/positive"
    test_negative_folder = BASE_DIR / "Data" / "3-ExtractInterval" / f"{period}-experiment"/f"{room}-test"/"negative"/"combined"
elif(data_type == "ecg"):
    test_positive_folder = BASE_DIR / "Data" / "3-ExtractInterval" /  "ecg" / f"{period}-experiment"/ f"{period}-test/positive"
    test_negative_folder = BASE_DIR / "Data" / "3-ExtractInterval" / "ecg" / f"{period}-experiment"/f"{period}-test"/"negative"

test_positive_raw_traces = get_trace_files(folder_path = test_positive_folder)

test_negative_raw_traces = get_trace_files(folder_path = test_negative_folder)



test_positive_raw_lists = csv_to_temp_time_list(input_files=test_positive_raw_traces)
test_negative_raw_lists = csv_to_temp_time_list(input_files=test_negative_raw_traces)

# print(test_positive_raw_lists)

#Path to log data
if (data_type == "temp"):
    log_data_path = BASE_DIR /"Data" /"8-LoggedData" / "metrics"/ f"{discretization_method}-temp-log.csv"
elif(data_type == "ecg"):
    log_data_path = BASE_DIR /"Data" /"8-LoggedData" / "metrics"/ f"{discretization_method}-ecg-log.csv"



# Parameter for nr of Traces
len_traces = len(train_raw_traces)  + 1
start_traces = 1
len_traces = 51

trace_list = [700]

#700


for w in w_values:

    train_traces, bins, breakpoints, global_mean, global_std = sax_discretization_multi(train_raw_lists,w, symbols)
    symbolic_train_trace, symbol_map, mapping = map_bins_to_symbols(train_traces,bins)

    # test_positive_traces_lists = preprocess_test_traces(test_traces = test_positive_raw_lists, bins = bins)
    # test_negative_traces_lists = preprocess_test_traces(test_traces = test_negative_raw_lists, bins = bins)

    test_positive_traces_lists = sax_preprocess_traces(
        test_positive_raw_lists,
        w,
        breakpoints,
        global_mean,
        global_std
    )


    test_negative_traces_lists = sax_preprocess_traces(
        test_negative_raw_lists,
        w,
        breakpoints,
        global_mean,
        global_std
    )



    for trace_nr in range(start_traces, len_traces):
        # Paths
        discretinize_data_path = (BASE_DIR/ "Data"/ "4-DiscretizationData"/ discretization_method / period
                                  / f"{room}-{trace_nr}trace-{period}-{discretization_method}-s{symbols}-trace.txt"
                                  )
        symbolic_train_trace_subset = symbolic_train_trace[:trace_nr ]



        format_output(symbolic_traces=symbolic_train_trace_subset, output_path=discretinize_data_path)


        # Loop over varying K-future
        for k in range(k_min, k_max + 1, k_increment):

            #Prepare Paths
            title = f"{room}-{trace_nr}trace-{period}-{discretization_method}-s{symbols}-w{w}-k{k}-ta"
            TA_output_path = (BASE_DIR / "Data" / "5-TaResults" / discretization_method / period)
            xml_path = (BASE_DIR / "Data" / "6-XMLOutput" / discretization_method / period
                        / f"{room}-{trace_nr}trace-{period}-{discretization_method}-s{symbols}--w{w}-k{k}.xml")
            run_id = f"{period}-{room}-{trace_nr}trace-s{symbols}-w{w}-k{k}"

            # Tranform to TA
            learner = TALearner(tss_path=discretinize_data_path,display=False,k=k)
            learner.ta.show(title=title,savePng=True,output_path=TA_output_path)
            learner.ta.export_ta( ta=learner.ta, path=xml_path, symbol_map=symbol_map, data_type = data_type, time = time, sim_nr=sim_nr)

            # Compute metrics
            metrics = learner.ta.evaluate_classifier(positive_tss = test_positive_traces_lists, negative_tss = test_negative_traces_lists,  save_path = log_data_path, run_id= run_id, timed=True)

            print(f"Done: trace:{trace_nr}, k:{k}, w:{w}, symbols={symbols}, Positive Acceptance Rate: {metrics['PAR']:.2f}%, Negative Acceptance Rate: {metrics['NAR']:.2f}% ")
            print("-------------------------------------------------------------------------------------")



