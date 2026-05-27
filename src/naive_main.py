
from TAG.TALearner import TALearner

from pathlib import Path

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
    period
)



BASE_DIR = Path(__file__).resolve().parent.parent


# PARAMETERS SETTINGS
data_type ="ecg"
room = "A"
discretization_method = "naiv"
sim_nr = 10000
period_nr = 1

if data_type == "temp":
    period = f"{period_nr}day"
    time = 86400
elif data_type == "ecg":
    period = "1beat"
    time = 275


# Parameter for Naiv
symbols = 10

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
train_traces, bins = equal_width_discretization(train_raw_lists, symbols)
symbolic_train_trace, symbol_map, mapping = map_bins_to_symbols(train_traces, bins)


#Prepare test traces (positive and negative samples)
if (data_type == "temp"):
    test_positive_folder = BASE_DIR / "Data" / "3-ExtractInterval" / f"{period}-experiment"/f"{room}-test/positive"
    test_negative_folder = BASE_DIR / "Data" / "3-ExtractInterval" / f"{period}-experiment"/f"{room}-test/negative/combined"
elif(data_type == "ecg"):
    test_positive_folder = BASE_DIR / "Data" / "3-ExtractInterval" /  "ecg" / f"{period}-experiment"/ f"{period}-test/positive"
    test_negative_folder = BASE_DIR / "Data" / "3-ExtractInterval" / "ecg" / f"{period}-experiment"/f"{period}-test/negative"

test_positive_raw_traces = get_trace_files(folder_path = test_positive_folder)
test_negative_raw_traces = get_trace_files(folder_path = test_negative_folder)

test_positive_raw_lists = csv_to_temp_time_list(input_files=test_positive_raw_traces)
test_negative_raw_lists = csv_to_temp_time_list(input_files=test_negative_raw_traces)


test_positive_traces_lists = preprocess_test_traces(test_traces = test_positive_raw_lists, bins = bins)
test_negative_traces_lists = preprocess_test_traces(test_traces = test_negative_raw_lists, bins = bins)


# Plot negative samples
negative_graph =  BASE_DIR / "Data" / "Graphs" / "Observed-negative-samples" / f"{discretization_method}-{period}-s{symbols}-experiment"
title_prefix = f"{discretization_method}-{period}-s{symbols}"

#Path to log data
if (data_type == "temp"):
    log_data_path = BASE_DIR /"Data" /"8-LoggedData" / "metrics"/ f"{discretization_method}-temp-log.csv"
elif(data_type == "ecg"):
    log_data_path = BASE_DIR /"Data" /"8-LoggedData" / "metrics"/ f"{discretization_method}-ecg-log.csv"


# Parameter for nr of Traces
len_traces = len(train_raw_traces)  + 1
start_traces = 1
len_traces = 51

trace_list = [600,700,800,900]

for trace_nr in trace_list: #range(start_traces, len_traces):


    # Paths

    if (data_type == "temp"):
        discretinize_data_path = (BASE_DIR/ "Data"/ "4-DiscretizationData"/ discretization_method / period
                                  / f"{room}-{trace_nr}trace-{period}-{discretization_method}-s{symbols}-trace.txt"
                                  )
    elif (data_type == "ecg"):
        discretinize_data_path = (BASE_DIR/ "Data"/ "4-DiscretizationData"/ discretization_method / period
                                  / f"{room}-{trace_nr}trace-{period}-{discretization_method}-s{symbols}-trace.txt"
                                  )

    symbolic_train_trace_subset = symbolic_train_trace[:trace_nr ]

    format_output(symbolic_traces=symbolic_train_trace_subset, output_path=discretinize_data_path)

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
        # learner.ta.show(title=title,savePng=True,output_path=TA_output_path)
        learner.ta.export_ta(ta = learner.ta, path=xml_path, symbol_map=symbol_map, data_type = data_type, time = time, sim_nr = sim_nr )

        # Compute metrics
        metrics = learner.ta.evaluate_classifier(positive_tss = test_positive_traces_lists, negative_tss = test_negative_traces_lists,  save_path = log_data_path, run_id= run_id, timed=True)

        print(f"Done: trace:{trace_nr}, k:{k}, symbols={symbols}, Positive Acceptance Rate: {metrics['PAR']:.2f}%, Negative Acceptance Rate: {metrics['NAR']:.2f}% ")
        print("-------------------------------------------------------------------------------------")




