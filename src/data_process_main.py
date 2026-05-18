import os
from pathlib import Path
from TAG.TALearner import TALearner

from DataProcessing.processData import (
    format_temperature_data,
    convert_ns_to_ms,
    extract_time_intervals,
    get_trace_files,
    format_ecg_data,
    extract_ecg_intervals_by_samples,
    extract_ecg_intervals_by_time_window,
    extract_ecg_intervals_by_beats,
    extract_fixed_beats_traces
)

# Format data
# rawData= "../Data/1-Raw/dataset-2023-02-27_2023-12-31.csv"
# formatedRawData = "../Data/2-FormatedRawData/dataset-2023-02-27-formatedRaw.csv"

room = "A"
# room_raw_data = f"../Data/1-Raw/room{room}.csv"
# room_raw_data = f"../Data/1-Raw/dataset-2023-02-27_2023-12-31.csv"
# room_file = f"../Data/2-FormatedRawData/dataset-{room}-formatedRaw.csv"

# format_temperature_data(input_file=room_raw_data, output_file=room_file, col=2)

# Extract traces

# 1 day traces
# extract_1day = f"../Data/3-ExtractInterval/{room}/1day"
# extract_time_intervals(
#    input_file=room_file, output_folder=extract_1day, output_prefix=f"{room}-1day")

# 7 days traces
# extract_7day = f"../Data/3-ExtractInterval/{room}/7day"
# extract_time_intervals(input_file=room_file, output_folder=extract_7day, output_prefix= f"{room}-7day", trace_days=7)

# 14 day traces
# extract_14day = f"../Data/3-ExtractInterval/{room}/14day"
# extract_time_intervals(input_file=room_file, output_folder=extract_14day, output_prefix= f"{room}-14day", trace_days=14)


# 30 day traces
# extract_30day = f"../Data/3-ExtractInterval/{room}/30day"
# extract_time_intervals(input_file=room_file, output_folder=extract_30day, output_prefix= f"{room}-30day", trace_days=30)


# ============= ECG DATA PROCESSING =============
# Process patient_100_ecg.csv with heartbeat-based splitting

BASE_DIR = Path(__file__).resolve().parent.parent
ecg_raw = BASE_DIR / "Data-marco" / "patient_100_ecg.csv"
ecg_formatted = BASE_DIR / "Data-marco" / "Formatted" /  "patient_100_ecg_formatted.csv"

output_egg_formated = BASE_DIR / "Data" / "2-FormatedRawData" / "patient_100_ecg.csv"

# Step 1: Format ECG data (use MLII lead - column 1)
# format_ecg_data(input_file=str(ecg_raw),
# #                 output_file=str(ecg_formatted), lead_col=1)


# Transfer from microseconds into milisecpnds
# convert_ns_to_ms(input_file=ecg_formatted, output_file=output_egg_formated)

#Extract traces

beats = 1

egg_output_folder = BASE_DIR / "Data" /"3-ExtractInterval" /"ecg" / f"{beats}beat"
ecg_output_prefix = f"{beats}beat"
#
extract_fixed_beats_traces(
    input_file=output_egg_formated,
    output_folder=egg_output_folder,
    n_beats = beats,
    output_prefix = ecg_output_prefix)

# Step 2a: Extract traces by fixed number of samples (e.g., 500 samples = 1 trace)
# Good for: consistent chunk sizes, simple analysis
# extract_ecg_intervals_by_samples(
#     input_file=ecg_formatted,
#     output_folder="../Data/3-ExtractInterval/ecg-by-samples",
#     output_prefix="patient100-samples",
#     samples_per_trace=500
# )

# Step 2b: Extract traces by time window (e.g., 5-second windows = 1 trace)
# Good for: comparing across different sampling rates, clinical use cases
# extract_ecg_intervals_by_time_window(
#     input_file=ecg_formatted,
#     output_folder="../Data/3-ExtractInterval/ecg-by-time",
#     output_prefix="patient100-5sec",
#     window_seconds=5
# )

# Step 2c: Extract traces by detected heartbeats
# Good for: heartbeat-aligned segmentation and beat-count traces
# extract_ecg_intervals_by_beats(
#     input_file=str(ecg_formatted),
#     output_folder=str(BASE_DIR / "Data" /
#                       "3-ExtractInterval" / "ecg-experimenrt"),
#     output_prefix="patient100-beats",
#     beats_per_trace=50,
#     min_rr_seconds=0.3
# )


# Process Data

# # Full 24-hour traces, one per day
# extractIntervalPath1day = "../Data/3-ExtractInterval/2023-02-27/1day"
# #os.path.join(output_path_interval_data, "experiment_1_full_days")
# extract_time_intervals(input_file=formatedRawData, output_folder=extractIntervalPath1day, output_prefix= "2023-02-27-1day")


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
