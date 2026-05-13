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
    preprocess_test_traces
)

from DataProcessing.processData import (
    format_temperature_data,
    extract_time_intervals,
    get_trace_files
)

BASE_DIR = Path(__file__).resolve().parent.parent


# PARAMETERS SETTINGS
data_type ="temp"
room = "A"
discretization_method = "sax"
period_nr = 1

time = 0
period = ""
if data_type == "temp":
    period = f"{period_nr}day"
    time = 86400
elif data_type == "ecg":
    period = "1beat"
    time = 250



# Parameter for SAX
symbols = 10
# w = 200
w_values = [10,20]
# Parameter for TAG
k_min = 4
k_max = 4
k_increment = 2

#Prepare train traces
if (data_type == "temp"):
    train_folder = BASE_DIR / "Data" / "3-ExtractInterval" / f"{period}-experiment"/ f"{room}-train"
elif(data_type == "ecg"):
    train_folder = BASE_DIR / "Data" / "3-ExtractInterval" /"ecg" / f"{period}-experiment"/ f"{period}-train"


for w in w_values:

    train_raw_traces = get_trace_files(folder_path = train_folder)
    train_raw_lists = csv_to_temp_time_list(input_files=train_raw_traces)

    train_traces, bins, _, _ = sax_discretization_multi(train_raw_lists,w, symbols)

    symbolic_train_trace, symbol_map, mapping = map_bins_to_symbols(train_traces, symbols, bins)

    #Prepare test traces (positive and negative samples)
    test_positive_folder = ""
    test_negative_folder = ""
    if (data_type == "temp"):
        test_positive_folder = BASE_DIR / "Data" / "3-ExtractInterval" / f"{period}-experiment"/f"{room}-test/positive"
        test_negative_folder = BASE_DIR / "Data" / "3-ExtractInterval" / f"{period}-experiment"/f"{room}-test/negative"
    elif(data_type == "ecg"):
        test_positive_folder = BASE_DIR / "Data" / "3-ExtractInterval" /  "ecg" / f"{period}-experiment"/ f"{period}-test/positive"
        test_negative_folder = BASE_DIR / "Data" / "3-ExtractInterval" / "ecg" / f"{period}-experiment"/f"{period}-test/negative"

    test_positive_raw_traces = get_trace_files(folder_path = test_positive_folder)
    test_negative_raw_traces = get_trace_files(folder_path = test_negative_folder)

    test_positive_raw_lists = csv_to_temp_time_list(input_files=test_positive_raw_traces)
    test_negative_raw_lists = csv_to_temp_time_list(input_files=test_negative_raw_traces)

    test_positive_traces_lists = preprocess_test_traces(test_traces = test_positive_raw_lists, bins = bins, s = symbols)
    test_negative_traces_lists = preprocess_test_traces(test_traces = test_negative_raw_lists, bins = bins, s = symbols)

    #Path to log data
    log_data_path = ""
    if (data_type == "temp"):
        log_data_path = BASE_DIR /"Data" /"8-LoggedData" / f"{discretization_method}-temp-log.csv"
    elif(data_type == "ecg"):
        log_data_path = BASE_DIR /"Data" /"8-LoggedData" / f"{discretization_method}-ecg-log.csv"


    # Parameter for nr of Traces
    len_traces = len(train_raw_traces)  + 1
    start_traces = 1
    len_traces = 10



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
            title = f"{room}-{trace_nr}trace-{period}-{discretization_method}-s{symbols}-w{w}-k{k}-ta"
            TA_output_path = (BASE_DIR / "Data" / "5-TaResults" / discretization_method / period)
            xml_path = (BASE_DIR / "Data" / "6-XMLOutput" / discretization_method / period
                        / f"{room}-{trace_nr}trace-{period}-{discretization_method}-s{symbols}--w{w}-k{k}.xml")
            run_id = f"{period}-{room}-{trace_nr}trace-s{symbols}-w{w}-k{k}"

            # Tranform to TA
            learner = TALearner(tss_path=discretinize_data_path,display=False,k=k)
            learner.ta.show(title=title,savePng=True,output_path=TA_output_path)
            learner.ta.export_ta(path=xml_path, symbol_map=symbol_map, data_type = data_type, time = time)

            # Compute metrics
            metrics = learner.ta.evaluate_classifier(positive_tss = test_positive_traces_lists, negative_tss = test_negative_traces_lists,  save_path = log_data_path, run_id= run_id, timed=True)

            print(f"Done: trace:{trace_nr}, k:{k}, w:{w}, symbols={symbols}, Positive Acceptance Rate: {metrics['PAR']:.2f}%, Negative Acceptance Rate: {metrics['NAR']:.2f}% ")
            print("-------------------------------------------------------------------------------------")



