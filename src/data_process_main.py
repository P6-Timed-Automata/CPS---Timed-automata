import os
from TAG.TALearner import TALearner

from DataProcessing.processData import (
    format_temperature_data,
    extract_time_intervals,
    get_trace_files
)

#Format data
# rawData= "../Data/1-Raw/dataset-2023-02-27_2023-12-31.csv"
# formatedRawData = "../Data/2-FormatedRawData/dataset-2023-02-27-formatedRaw.csv"

room = "A"
# room_raw_data = f"../Data/1-Raw/room{room}.csv"
room_raw_data = f"../Data/1-Raw/dataset-2023-02-27_2023-12-31.csv"
room_file = f"../Data/2-FormatedRawData/dataset-{room}-formatedRaw.csv"

format_temperature_data(input_file = room_raw_data,output_file=room_file, col=2)

#Extract traces

# 1 day traces
extract_1day = f"../Data/3-ExtractInterval/{room}/1day"
extract_time_intervals(input_file=room_file, output_folder=extract_1day, output_prefix= f"{room}-1day")

# 7 days traces
extract_7day = f"../Data/3-ExtractInterval/{room}/7day"
#extract_time_intervals(input_file=room_file, output_folder=extract_7day, output_prefix= f"{room}-7day", trace_days=7)

# 14 day traces
extract_14day = f"../Data/3-ExtractInterval/{room}/14day"
#extract_time_intervals(input_file=room_file, output_folder=extract_14day, output_prefix= f"{room}-14day", trace_days=14)


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
