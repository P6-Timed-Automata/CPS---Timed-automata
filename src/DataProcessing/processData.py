import numpy as np
from datetime import datetime, timedelta
from collections import Counter
import os



def format_temperature_data(input_file, output_file, col):
    """
    Loads raw temperature CSV data and cleans invalid values.

    Keeps the original timestamp.

    Args:
        input_file: raw CSV file
        output_file: path to save cleaned CSV
        col: column index of temperature data to use
    """
    #Reads inputfile
    data = np.genfromtxt(
        input_file,
        delimiter=',',
        dtype=str,
        usecols=(0, 1, col),
        encoding="utf-8-sig",
        skip_header=1,
        invalid_raise=False
    )

    #remove invalid rows
    bad_values = {'#I/T', '#N/A', '', 'nan', 'NaN', None}
    mask = np.array([str(v) not in bad_values for v in data[:, 2]])
    data = data[mask]

    # Splite data into 3 arrays
    ids = data[:, 0].astype(int)
    timestamps = data[:, 1]
    temps = data[:, 2].astype(float)

    # Convert timestamos into Python datetime objects
    parsed = np.array([
        datetime.strptime(t, "%Y-%m-%dT%H:%M:%S%z")
        for t in timestamps
    ])

    # All data is sorted by time
    order = np.argsort(parsed)
    parsed = parsed[order]
    ids = ids[order]
    temps = temps[order]

    # Combines everything into one array
    result = np.column_stack((ids, parsed.astype(str), temps))

    dirpath = os.path.dirname(output_file)
    if dirpath:
        os.makedirs(dirpath, exist_ok=True)

    np.savetxt(
        output_file,
        result,
        delimiter=';',
        header='id;timestamp;temperature',
        fmt='%s',
        comments=''
    )

    print(f"Saved {len(result)} rows to {output_file}")



def extract_time_intervals(input_file, output_folder, output_prefix, trace_days=1):
    """
        Extracts time traces from formatted temperature data.

        Each trace contains a fixed number of consecutive calendar days.

        Parameters:
            input_file (str): Path to the formatted CSV file (output from format_temperature_data).
            output_folder (str): Directory where extracted trace files will be saved.
            output_prefix (str):Prefix used for naming output trace files.
            trace_days (int): Number of consecutive days per trace.
        """
    os.makedirs(output_folder, exist_ok=True)

    #Load input file
    data = np.genfromtxt(
        input_file,
        delimiter=';',
        dtype=str,
        skip_header=1
    )
    # parse timestamps
    timestamps = np.array([
        datetime.strptime(t, "%Y-%m-%d %H:%M:%S%z")
        for t in data[:, 1]
    ])

    #Converts temperature strings → floats
    temps = data[:, 2].astype(float)

    # Sort
    order = np.argsort(timestamps)
    timestamps = timestamps[order]
    temps = temps[order]

    # Extract calendar days by remove timestamps
    days = np.array([t.date() for t in timestamps])

    # Removes duplicates days and sort in chronical order
    unique_days = np.sort(np.unique(days))

    trace_idx = 1  # file numbereing
    i = 0  # unique_days numbering

    # Loop while enough days to form a trace
    while i <= len(unique_days) - trace_days:

        # Pick candidate trace window
        selected_days = unique_days[i:i + trace_days]

        # Check if days are consecutive(NO missing days allowed)
        expected = np.arange(
            selected_days[0].toordinal(),
            selected_days[0].toordinal() + trace_days
        )

        actual = np.array([d.toordinal() for d in selected_days])

        # if missing day then skip candidate window
        if not np.array_equal(actual, expected):
            i += 1
            continue

        # Select all timestamps that belong to those selected days
        mask = np.isin(days, selected_days)

        t = timestamps[mask]
        x = temps[mask]

        # Safety check
        if len(t) == 0:
            i += trace_days
            continue

        # rebase time so data start by 0 and increment by delay
        t0 = t[0]
        rel_time = np.array([(ti - t0).total_seconds() for ti in t])

        # combine data
        out = np.column_stack((rel_time, x))

        np.savetxt(
            os.path.join(output_folder, f"{output_prefix}-tid{trace_idx}.csv"),
            out,
            delimiter=';',
            header="time_seconds;temperature",
            fmt=['%.0f', '%.5f'],
            comments=''
        )

        trace_idx += 1
        i += trace_days   # ✔ IMPORTANT: non-overlapping traces

    print(f"Done extracting intervals of:{trace_days}. Saved {trace_idx - 1} traces.")

def get_trace_files(folder_path, extension=".csv", max_files=None,):
    files = []

    for f in os.listdir(folder_path):
        if f.endswith(extension):
            full_path = os.path.join(folder_path, f)
            files.append(full_path)

    if max_files is not None:
        files = files[:max_files]

    return sorted(files)




# def format_temperature_data(input_file, output_file, col):
#     # Load data
#     data = np.genfromtxt(
#         input_file,
#         delimiter=',',
#         dtype=str,
#         usecols=(0, 1, col),
#         encoding="utf-8-sig",
#         skip_header=1
#     )
#
#     # Remove invalid temperature rows
#     bad_values = {'#I/T', '', '#N/A', 'NaN', 'nan', None}
#     #mask = (data[:, 2] != '#I/T') & (data[:, 2] != '')
#     mask = np.array([str(v) not in bad_values for v in data[:, 2]])
#     data = data[mask]
#
#
#     ids          = data[:, 0].astype(int)
#     timestamps   = data[:, 1]
#     temperatures = data[:, 2].astype(float)
#
#     def parse_ts(ts):
#         return datetime.strptime(ts, "%Y-%m-%dT%H:%M:%S%z")
#
#     parsed = np.array([parse_ts(ts) for ts in timestamps])
#
#     # --- Discard incomplete 24-hour periods ---
#     # Group by calendar date, then keep only days whose count matches the mode.
#     dates       = np.array([t.date() for t in parsed])
#     date_counts = Counter(dates)
#     expected    = Counter(date_counts.values()).most_common(1)[0][0]
#     valid_dates = {d for d, c in date_counts.items() if c == expected}
#     day_mask    = np.array([d in valid_dates for d in dates])
#
#     discarded = len(dates) - day_mask.sum()
#     print(f"Discarding {len(date_counts) - len(valid_dates)} incomplete day(s) ({discarded} rows)")
#
#     ids          = ids[day_mask]
#     temperatures = temperatures[day_mask]
#     parsed       = parsed[day_mask]
#     # -----------------------------------------
#
#     # Compute delays relative to first retained timestamp
#     t0     = parsed[0]
#     delays = np.array([(t - t0).total_seconds() for t in parsed])
#
#     result = np.column_stack((ids, delays, temperatures))
#     result = result[~np.isnan(result.astype(float)).any(axis=1)]
#
#     dirpath = os.path.dirname(output_file)
#     if dirpath:
#         os.makedirs(dirpath, exist_ok=True)
#
#     np.savetxt(
#         output_file,
#         result,
#         delimiter=';',
#         header='id;time_seconds;temperature',
#         fmt=['%d', '%.0f', '%.5f'],
#         comments=''
#     )
#     print(f"Saved {len(result)} rows to {output_file}")
#
# def extract_time_intervals(input_file, output_folder, output_prefix, trace_days=1, window=None):
#     """
#     Extracts time traces from a formatted CSV produced by format_temperature_data.
#     Each trace spans trace_days consecutive calendar days.
#
#     Args:
#         input_file   : formatted CSV from format_temperature_data
#         output_folder: folder to save trace files into (created if not exists)
#         output_prefix: prefix for output files → {prefix}_trace1.csv, etc.
#         trace_days   : number of consecutive days per trace (default 1)
#                        e.g. 7 for weekly traces, 30 for monthly
#         window       : (start_sec, end_sec) within-day offset in seconds applied
#                        to every day in the trace (None = full 24-hour day)
#                        e.g. (0, 18000) for the first 5 hours of each day
#     """
#
#     os.makedirs(output_folder, exist_ok=True)
#
#     data  = np.genfromtxt(input_file, delimiter=';', dtype=str, skip_header=1)
#     times = data[:, 1].astype(int)
#     temps = data[:, 2].astype(float)
#
#     sampling_interval = int(np.median(np.diff(times)))
#     rows_per_day      = 86400 // sampling_interval
#
#     if window is not None:
#         win_start, win_end = window
#         assert 0 <= win_start < win_end <= 86400, "window must be within [0, 86400] (a day) and start < end"
#         win_start_row = win_start // sampling_interval
#         win_end_row   = win_end   // sampling_interval
#     else:
#         win_start_row, win_end_row = 0, rows_per_day
#
#     # Group rows by calendar day index
#     n_days      = len(times) // rows_per_day
#     day_indices = np.arange(n_days)
#
#     # Compute the absolute day number from t=0 for each day to detect gaps
#     day_offsets = np.array([int(times[d * rows_per_day]) // 86400 for d in day_indices])
#
#     # Group consecutive calendar days into traces, discarding incomplete groups
#     trace_idx = 1
#     d = 0
#     while d <= n_days - trace_days:
#         group = day_indices[d : d + trace_days]
#
#         # Check all days in group are consecutive calendar days
#         expected_offsets = np.arange(day_offsets[d], day_offsets[d] + trace_days)
#         if not np.array_equal(day_offsets[d : d + trace_days], expected_offsets):
#             # Gap detected — skip forward to the next day after the break
#             gap_pos = np.where(np.diff(day_offsets[d : d + trace_days]) != 1)[0][0]
#             print(f"Gap detected at day {d + gap_pos + 1} (calendar offset {day_offsets[d + gap_pos]}→{day_offsets[d + gap_pos + 1]}), discarding incomplete trace {trace_idx}")
#             d += gap_pos + 1
#             continue
#
#         # Collect window rows from each day in the group
#         segments = []
#         for day in group:
#             day_start_row = day * rows_per_day
#             segments.append((
#                 times[day_start_row + win_start_row : day_start_row + win_end_row],
#                 temps[day_start_row + win_start_row : day_start_row + win_end_row]
#             ))
#
#         trace_times = np.concatenate([s[0] for s in segments])
#         trace_temps = np.concatenate([s[1] for s in segments])
#         rebased     = trace_times - trace_times[0]
#
#         filtered = np.column_stack((rebased, trace_temps))
#         out_file = os.path.join(output_folder, f"{output_prefix}-tid{trace_idx}.csv")
#         np.savetxt(
#             out_file,
#             filtered,
#             delimiter=';',
#             fmt=['%d', '%.5f'],
#             header="time_seconds;temperature",
#             comments=''
#         )
#         print(f"Trace {trace_idx}: {trace_times[0]}s–{trace_times[-1]}s ({trace_days} day(s)) - {len(filtered)} rows - {out_file}")
#         trace_idx += 1
#         d += trace_days
#
