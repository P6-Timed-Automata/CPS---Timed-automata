import numpy as np
from datetime import datetime
from collections import Counter
import os

def format_temperature_data(input_file, output_file):
    # Load data
    data = np.genfromtxt(
        input_file,
        delimiter=';',
        dtype=str,
        usecols=(0, 1, 2),
        encoding="utf-8-sig",
        skip_header=1
    )

    # Remove invalid temperature rows
    mask = (data[:, 2] != '#I/T') & (data[:, 2] != '')
    data = data[mask]

    ids          = data[:, 0].astype(int)
    timestamps   = data[:, 1]
    temperatures = data[:, 2].astype(float)

    def parse_ts(ts):
        return datetime.strptime(ts, "%Y-%m-%dT%H:%M:%S%z")

    parsed = np.array([parse_ts(ts) for ts in timestamps])

    # --- Discard incomplete 24-hour periods ---
    # Group by calendar date, then keep only days whose count matches the mode.
    dates       = np.array([t.date() for t in parsed])
    date_counts = Counter(dates)
    expected    = Counter(date_counts.values()).most_common(1)[0][0]
    valid_dates = {d for d, c in date_counts.items() if c == expected}
    day_mask    = np.array([d in valid_dates for d in dates])

    discarded = len(dates) - day_mask.sum()
    print(f"Discarding {len(date_counts) - len(valid_dates)} incomplete day(s) ({discarded} rows)")

    ids          = ids[day_mask]
    temperatures = temperatures[day_mask]
    parsed       = parsed[day_mask]
    # -----------------------------------------

    # Compute delays relative to first retained timestamp
    t0     = parsed[0]
    delays = np.array([(t - t0).total_seconds() for t in parsed])

    result = np.column_stack((ids, delays, temperatures))
    result = result[~np.isnan(result.astype(float)).any(axis=1)]

    np.savetxt(
        output_file,
        result,
        delimiter=';',
        header='id;time_seconds;temperature',
        fmt=['%d', '%.0f', '%.5f'],
        comments=''
    )
    print(f"Saved {len(result)} rows to {output_file}")


def extract_time_intervals(input_file, output_folder, output_prefix, window=None):
    """
    Args:
        input_file   : formatted CSV from format_temperature_data
        output_folder: folder to save trace files into (created if not exists)
        output_prefix: prefix for output files → {prefix}_trace1.csv, etc.
        window       : (start_sec, end_sec) within-day offset in seconds
                       (None = full 24-hour day)
    """
    os.makedirs(output_folder, exist_ok=True)

    data  = np.genfromtxt(input_file, delimiter=';', dtype=str, skip_header=1)
    times = data[:, 1].astype(int)
    temps = data[:, 2].astype(float)

    sampling_interval = int(np.median(np.diff(times)))
    rows_per_day      = 86400 // sampling_interval

    if window is not None:
        win_start, win_end = window
        assert 0 <= win_start < win_end <= 86400, "window must be within [0, 86400] and start < end"
        win_start_row = win_start // sampling_interval
        win_end_row   = win_end   // sampling_interval
    else:
        win_start_row, win_end_row = 0, rows_per_day

    rows_per_trace = win_end_row - win_start_row
    n_days         = len(times) // rows_per_day

    for day in range(n_days):
        day_start_row = day * rows_per_day
        segment_times = times[day_start_row + win_start_row : day_start_row + win_end_row]
        segment_temps = temps[day_start_row + win_start_row : day_start_row + win_end_row]

        rebased_times = segment_times - segment_times[0]
        filtered      = np.column_stack((rebased_times, segment_temps))

        out_file = os.path.join(output_folder, f"{output_prefix}_trace{day + 1}.csv")
        np.savetxt(
            out_file,
            filtered,
            delimiter=';',
            fmt=['%d', '%.5f'],
            header="time_seconds;temperature",
            comments=''
        )
        print(f"Trace {day + 1}: {segment_times[0]}s–{segment_times[-1]}s - {len(filtered)} rows - {out_file}")

#Tests
# Full 24-hour traces, all days
extract_time_intervals("formated_raw_data.csv", "experiment_1_full_days", "trace")

# First 5 hours of each day
extract_time_intervals("formated_raw_data.csv", "experiment_2_5-hour_days", "trace", window=(0, 18000))

# Custom window within each day
extract_time_intervals("formated_raw_data.csv", "experiment_3_custom_window", "trace", window=(3600, 21600))
