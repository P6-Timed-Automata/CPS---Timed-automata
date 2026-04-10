import json, os
from TAG.TALearner import TALearner

from Discretization.discretizationSetup import (
    csv_to_temp_time_list,
    format_output,
    map_bins_to_symbols
)
from Discretization.naive import equal_width_discretization

from dataProcessing.processData import (
    format_temperature_data,
    extract_time_intervals,
    get_trace_files
)

rawData= "../Data/1-Raw/dataset-2023-02-27_2023-12-31.csv"
formatedRawData = "../Data/2-FormatedRawData/dataset-2023-02-27-formatedRaw.csv"

format_temperature_data(input_file = rawData,output_file=formatedRawData, col=2)

# Process Data

# Full 24-hour traces, one per day
extractIntervalPath1day = "../Data/3-ExtractInterval/2023-02-27/1day"
#os.path.join(output_path_interval_data, "experiment_1_full_days")
extract_time_intervals(input_file=formatedRawData, output_folder=extractIntervalPath1day, output_prefix= "2023-02-27-1day")

# Full 1-hour traces, one per day"experiment_1_full_days"),
extractIntervalPath1day1hour = "../Data/3-ExtractInterval/2023-02-27/1day-wd-1h-inter-0-3600"
#os.path.join(output_path_interval_data, "experiment_2_daily_windowed")
extract_time_intervals(input_file=formatedRawData, output_folder = extractIntervalPath1day1hour , output_prefix= "2023-02-27-1day-wd-1h-inter-0-3600", trace_days=1, window=(0, 3600) )

# 7-day traces
extractIntervalPath7day = "../Data/3-ExtractInterval/2023-02-27/7day"
#os.path.join(output_path_interval_data, "experiment_3_weekly")
extract_time_intervals(input_file=formatedRawData, output_folder = extractIntervalPath7day,output_prefix= "2023-02-27-7day", trace_days=7)

# First 5 hours of each day, grouped into weekly traces
extractIntervalPath7day = "../Data/3-ExtractInterval/2023-02-27/7day-wd-5h-inter-0-18000"
#os.path.join(output_path_interval_data, "experiment_4_weekly_windowed")
extract_time_intervals(input_file=formatedRawData, output_folder=extractIntervalPath7day, output_prefix= "2023-02-27-7day-wd-5h-inter-0-18000", trace_days=7, window=(0, 18000))


# #Prepare Data for TAG
b = 12
rawTraces = [
    '../Data/3-ExtractInterval/2023-02-27/1day/2023-02-27-1day-tid1.csv'
]

data_lists = csv_to_temp_time_list(input_files=rawTraces)

# Discretenize

traces, bins = equal_width_discretization(data_lists, b)
symbolic_trace, symbol_map, mapping = map_bins_to_symbols(traces, b, bins)

discretinize_data_path = "../Data/4-DiscretizationData/naiv/1day/2023-02-27-tid1-1trace-1day-naiv-b12-trace.txt"
format_output(symbolic_res_list=symbolic_trace, output_path=discretinize_data_path)


# Call TAG

k = 4
learner = TALearner(tss_path=discretinize_data_path, display=False, k=k)

title = "2023-02-27-tid1-1trace-1day-naiv-b12-k4-ta"
TA_output_path = "../Data/5-TaResults/naiv/1day"
learner.ta.show(title = title, savePng = True, output_path = TA_output_path)

xml_path = '../Data/6-XMLOutput/naiv/1day/2023-02-27-tid1-1trace-1day-naiv-b12-k4.xml'
learner.ta.export_ta(path=xml_path, symbol_map=symbol_map)


