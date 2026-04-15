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


#Format data
# rawData= "../Data/1-Raw/dataset-2023-02-27_2023-12-31.csv"
# formatedRawData = "../Data/2-FormatedRawData/dataset-2023-02-27-formatedRaw.csv"

room = "A"
room_raw_data = f"../Data/1-Raw/room{room}.csv"
room_file = f"../Data/2-FormatedRawData/dataset-{room}-formatedRaw.csv"

#format_temperature_data(input_file = room_raw_data,output_file=room_file, col=2)

#Extract traces

# 1 day traces
extract_1day = f"../Data/3-ExtractInterval/{room}/1day"
# extract_time_intervals(input_file=room_file, output_folder=extract_1day, output_prefix= f"{room}-1day")

# 7 days traces
extract_7day = f"../Data/3-ExtractInterval/{room}/7day"
# extract_time_intervals(input_file=room_file, output_folder=extract_7day, output_prefix= f"{room}-7day", trace_days=7)

# 30 day traces
extract_30day = f"../Data/3-ExtractInterval/{room}/30day"
# extract_time_intervals(input_file=room_file, output_folder=extract_30day, output_prefix= f"{room}-30day", trace_days=30)


# Process Data

# # Full 24-hour traces, one per day
#extractIntervalPath1day = "../Data/3-ExtractInterval/2023-02-27/1day"
# #os.path.join(output_path_interval_data, "experiment_1_full_days")
#extract_time_intervals(input_file=formatedRawData, output_folder=extractIntervalPath1day, output_prefix= "2023-02-27-1day")


# # Full 1-hour traces, one per day"experiment_1_full_days"),
# extractIntervalPath1day1hour = "../Data/3-ExtractInterval/2023-02-27/1day-wd-1h-inter-0-3600"
# #os.path.join(output_path_interval_data, "experiment_2_daily_windowed")
# extract_time_intervals(input_file=formatedRawData, output_folder = extractIntervalPath1day1hour , output_prefix= "2023-02-27-1day-wd-1h-inter-0-3600", trace_days=1, window=(0, 3600) )

# # 7-day traces
# extractIntervalPath7day = "../Data/3-ExtractInterval/2023-02-27/7day"
# #os.path.join(output_path_interval_data, "experiment_3_weekly")
# extract_time_intervals(input_file=formatedRawData, output_folder = extractIntervalPath7day,output_prefix= "2023-02-27-7day", trace_days=7)

# # First 5 hours of each day, grouped into weekly traces
# extractIntervalPath7day = "../Data/3-ExtractInterval/2023-02-27/7day-wd-5h-inter-0-18000"
# #os.path.join(output_path_interval_data, "experiment_4_weekly_windowed")
# extract_time_intervals(input_file=formatedRawData, output_folder=extractIntervalPath7day, output_prefix= "2023-02-27-7day-wd-5h-inter-0-18000", trace_days=7, window=(0, 18000))

from pathlib import Path
# Project root = CPS---Timed-automata/
BASE_DIR = Path(__file__).resolve().parent.parent

# #Prepare Data for TAG
b = 15
#k = 6
discretization_method = "naiv"
period = "30day"
trace_nr = 1

experiment_folder = BASE_DIR / "Data" / "3-ExtractInterval" / f"{period}-experiment"
#experiment_folder = f"../Data/3-ExtractInterval/{period}-experiment"
for trace_nr in range(1, 11):  # 1 → 10 traces
    rawTraces = get_trace_files(folder_path=experiment_folder, max_files=trace_nr)

    data_lists = csv_to_temp_time_list(input_files=rawTraces)

    # Discretize (same for all k)
    traces, bins = equal_width_discretization(data_lists, b)
    symbolic_trace, symbol_map, mapping = map_bins_to_symbols(traces, b, bins)

    # Save discretized data ONCE per trace_nr
    discretinize_data_path = (
            BASE_DIR
            / "Data"
            / "4-DiscretizationData"
            / discretization_method
            / period
            / f"{room}-{trace_nr}trace-{period}-{discretization_method}-b{b}-trace.txt"
    )

    #discretinize_data_path = f"../Data/4-DiscretizationData/{discretization_method}/{period}/{room}-{trace_nr}trace-{period}-{discretization_method}-b{b}-trace.txt"

    # make sure parent folders exist
    discretinize_data_path.parent.mkdir(parents=True, exist_ok=True)

    format_output(symbolic_res_list=symbolic_trace, output_path=discretinize_data_path)

    # Now vary k
    for k in range(4, 11, 2):  # 4, 6, 8, 10

        learner = TALearner(
            tss_path=discretinize_data_path,
            display=False,
            k=k
        )

        title = f"{room}-{trace_nr}trace-{period}-{discretization_method}-b{b}-k{k}-ta"
        TA_output_path = (
                BASE_DIR
                / "Data"
                / "5-TaResults"
                / discretization_method
                / period
        )
        TA_output_path.mkdir(parents=True, exist_ok=True)
        #TA_output_path = f"../Data/5-TaResults/{discretization_method}/{period}"


        learner.ta.show(
            title=title,
            savePng=True,
            output_path=TA_output_path
        )

        xml_path = (
                BASE_DIR
                / "Data"
                / "6-XMLOutput"
                / discretization_method
                / period
                / f"{room}-{trace_nr}trace-{period}-{discretization_method}-b{b}-k{k}.xml"
        )
        xml_path.parent.mkdir(parents=True, exist_ok=True)

        #xml_path = f'../Data/6-XMLOutput/{discretization_method}/{period}/{room}-{trace_nr}trace-{period}-{discretization_method}-b{b}-k{k}.xml'

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

