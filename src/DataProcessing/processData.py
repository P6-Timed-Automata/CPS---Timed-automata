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
    # Reads inputfile
    data = np.genfromtxt(
        input_file,
        delimiter=',',
        dtype=str,
        usecols=(0, 1, col),
        encoding="utf-8-sig",
        skip_header=1,
        invalid_raise=False
    )

    # remove invalid rows
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
    os.makedirs(output_folder, exist_ok=True)

    data = np.genfromtxt(input_file, delimiter=';', dtype=str, skip_header=1)
    timestamps = np.array([
        datetime.strptime(t, "%Y-%m-%d %H:%M:%S%z")
        for t in data[:, 1]
    ])
    temps = data[:, 2].astype(float)

    order = np.argsort(timestamps)
    timestamps = timestamps[order]
    temps = temps[order]

    days = np.array([t.date() for t in timestamps])

    # Count rows per day and keep only days matching the modal count
    unique_days, counts = np.unique(days, return_counts=True)
    expected_count = Counter(counts).most_common(1)[0][0]
    valid_days = unique_days[counts == expected_count]

    print(f"Expected {expected_count} rows/day. "
          f"Dropped {len(unique_days) - len(valid_days)} incomplete day(s).")

    # Find runs of consecutive valid days
    valid_days = np.sort(valid_days)
    ordinals = np.array([d.toordinal() for d in valid_days])
    # True where a gap exists
    gaps = np.diff(ordinals) != 1
    run_boundaries = np.concatenate(
        ([0], np.where(gaps)[0] + 1, [len(valid_days)]))

    trace_idx = 1
    for b in range(len(run_boundaries) - 1):
        run = valid_days[run_boundaries[b]:run_boundaries[b + 1]]

        # Slide a window of trace_days across this run
        for i in range(0, len(run) - trace_days + 1, trace_days):
            selected_days = run[i:i + trace_days]
            mask = np.isin(days, selected_days)

            t = timestamps[mask]
            x = temps[mask]

            t0 = t[0]
            rel_time = np.array([(ti - t0).total_seconds() for ti in t])

            out = np.column_stack((rel_time, x))
            np.savetxt(
                os.path.join(
                    output_folder, f"{output_prefix}-tid{trace_idx}.csv"),
                out,
                delimiter=';',
                header="time_seconds;temperature",
                fmt=['%.0f', '%.5f'],
                comments=''
            )
            trace_idx += 1

    print(f"Saved {trace_idx - 1} traces of {trace_days} day(s).")


def get_trace_files(folder_path, extension=".csv", max_files=None,):
    files = []

    for f in os.listdir(folder_path):
        if f.endswith(extension):
            full_path = os.path.join(folder_path, f)
            files.append(full_path)

    if max_files is not None:
        files = files[:max_files]

    return sorted(files)


# ============== ECG Data Processing Functions ==============

def format_ecg_data(input_file, output_file, lead_col=1):
    """
    Loads and cleans ECG data.

    Args:
        input_file: raw ECG CSV file (with ts, MLII, V5 columns)
        output_file: path to save cleaned CSV
        lead_col: which lead to extract (1=MLII, 2=V5)
    """
    # Read ECG data
    data = np.genfromtxt(
        input_file,
        delimiter=',',
        dtype=str,
        usecols=(0, lead_col),
        encoding="utf-8-sig",
        skip_header=1,
        invalid_raise=False
    )

    # Remove invalid rows
    bad_values = {'#I/T', '#N/A', '', 'nan', 'NaN', None}
    mask = np.array([str(v) not in bad_values for v in data[:, 1]])
    data = data[mask]

    # Parse timestamps (in microseconds) and values
    timestamps = data[:, 0].astype(np.int64)
    values = data[:, 1].astype(float)

    # Sort by timestamp
    order = np.argsort(timestamps)
    timestamps = timestamps[order]
    values = values[order]

    # Combine into output
    result = np.column_stack((timestamps, values))

    dirpath = os.path.dirname(output_file)
    if dirpath:
        os.makedirs(dirpath, exist_ok=True)

    np.savetxt(
        output_file,
        result,
        delimiter=';',
        header='timestamp_us;ecg_value',
        fmt=['%d', '%.5f'],
        comments=''
    )

    print(f"Saved {len(result)} ECG samples to {output_file}")


def extract_ecg_intervals_by_samples(input_file, output_folder, output_prefix, samples_per_trace=500, max_traces=None):
    """
    Split ECG data into fixed-size chunks (by number of samples).

    Args:
        input_file: formatted ECG CSV file
        output_folder: where to save traces
        output_prefix: prefix for output files
        samples_per_trace: number of samples per trace (e.g., 500)
    """
    os.makedirs(output_folder, exist_ok=True)

    data = np.genfromtxt(input_file, delimiter=';', dtype=str, skip_header=1)
    timestamps = data[:, 0].astype(np.int64)
    values = data[:, 1].astype(float)

    trace_idx = 1
    for i in range(0, len(timestamps) - samples_per_trace, samples_per_trace):
        t_chunk = timestamps[i:i + samples_per_trace]
        v_chunk = values[i:i + samples_per_trace]

        # Convert to relative time (milliseconds)
        t0 = t_chunk[0]
        # Convert microseconds to milliseconds
        rel_time = (t_chunk - t0) / 1000.0

        out = np.column_stack((rel_time, v_chunk))
        np.savetxt(
            os.path.join(output_folder, f"{output_prefix}-tid{trace_idx}.csv"),
            out,
            delimiter=';',
            header="time_ms;ecg_value",
            fmt=['%.0f', '%.5f'],
            comments=''
        )
        trace_idx += 1

        if max_traces and trace_idx > max_traces:
            break

    saved_count = trace_idx - 1
    limit_note = f" (limited to {max_traces})" if max_traces and saved_count >= max_traces else ""
    print(
        f"Saved {saved_count} ECG traces of {samples_per_trace} samples each{limit_note}.")


def extract_ecg_intervals_by_time_window(input_file, output_folder, output_prefix, window_seconds=5):
    """
    Split ECG data into time windows.

    Args:
        input_file: formatted ECG CSV file
        output_folder: where to save traces
        output_prefix: prefix for output files
        window_seconds: duration of each trace in seconds
    """
    os.makedirs(output_folder, exist_ok=True)

    data = np.genfromtxt(input_file, delimiter=';', dtype=str, skip_header=1)
    timestamps = data[:, 0].astype(np.int64)
    values = data[:, 1].astype(float)

    window_us = window_seconds * 1_000_000  # Convert to microseconds

    trace_idx = 1
    i = 0
    while i < len(timestamps):
        t_start = timestamps[i]
        t_end = t_start + window_us

        # Find all samples within this window
        mask = (timestamps >= t_start) & (timestamps < t_end)
        t_chunk = timestamps[mask]
        v_chunk = values[mask]

        if len(t_chunk) > 0:
            # Convert to relative time (milliseconds)
            t0 = t_chunk[0]
            rel_time = (t_chunk - t0) / 1000.0

            out = np.column_stack((rel_time, v_chunk))
            np.savetxt(
                os.path.join(
                    output_folder, f"{output_prefix}-tid{trace_idx}.csv"),
                out,
                delimiter=';',
                header="time_ms;ecg_value",
                fmt=['%.0f', '%.5f'],
                comments=''
            )
            trace_idx += 1

        # Move to next window
        i += np.sum(mask)
        if i <= 0:  # Safety check to avoid infinite loop
            i = i + 1

    print(f"Saved {trace_idx - 1} ECG traces of {window_seconds}s each.")


def _detect_ecg_peaks(timestamps, values, min_rr_seconds=0.3):
    if len(values) < 3:
        return np.array([], dtype=int)

    signal = values - np.median(values)
    window = min(11, len(signal) if len(signal) % 2 == 1 else len(signal) - 1)
    if window < 3:
        window = 3

    kernel = np.ones(window) / float(window)
    smooth = np.convolve(signal, kernel, mode='same')
    abs_signal = np.abs(smooth)

    threshold = max(
        np.mean(abs_signal) + 0.5 * np.std(abs_signal),
        np.percentile(abs_signal, 75)
    )

    candidates = np.where(
        (abs_signal[1:-1] > abs_signal[:-2]) &
        (abs_signal[1:-1] >= abs_signal[2:]) &
        (abs_signal[1:-1] > threshold)
    )[0] + 1

    if len(candidates) == 0:
        threshold = np.mean(abs_signal) + 0.25 * np.std(abs_signal)
        candidates = np.where(
            (abs_signal[1:-1] > abs_signal[:-2]) &
            (abs_signal[1:-1] >= abs_signal[2:]) &
            (abs_signal[1:-1] > threshold)
        )[0] + 1

    if len(candidates) == 0:
        return np.array([], dtype=int)

    min_distance_us = int(min_rr_seconds * 1_000_000)
    peaks = [candidates[0]]
    for idx in candidates[1:]:
        if timestamps[idx] - timestamps[peaks[-1]] >= min_distance_us:
            peaks.append(idx)

    return np.array(peaks, dtype=int)


def _save_ecg_segment(output_folder, output_prefix, trace_idx, t_chunk, v_chunk):
    t0 = t_chunk[0]
    rel_time = (t_chunk - t0) / 1000.0
    out = np.column_stack((rel_time, v_chunk))
    np.savetxt(
        os.path.join(output_folder, f"{output_prefix}-tid{trace_idx}.csv"),
        out,
        delimiter=';',
        header="time_ms;ecg_value",
        fmt=['%.0f', '%.5f'],
        comments=''
    )


def extract_ecg_intervals_by_time_and_beats(
    input_file,
    output_folder,
    output_prefix,
    window_seconds=5,
    beats_per_trace=50,
    min_rr_seconds=0.3,
    max_traces=None
):
    """
    Split ECG data into fixed time windows, then cut beats inside each window.

    Args:
        input_file: formatted ECG CSV file
        output_folder: where to save traces
        output_prefix: prefix for output files
        window_seconds: length of each time slice in seconds
        beats_per_trace: number of beats per saved trace inside each window
        min_rr_seconds: minimum distance between peaks in seconds
    """
    os.makedirs(output_folder, exist_ok=True)

    data = np.genfromtxt(input_file, delimiter=';', dtype=str, skip_header=1)
    timestamps = data[:, 0].astype(np.int64)
    values = data[:, 1].astype(float)

    window_us = window_seconds * 1_000_000  # Convert to microseconds
    trace_idx = 1
    i = 0

    while i < len(timestamps):
        t_start = timestamps[i]
        t_end = t_start + window_us

        mask = (timestamps >= t_start) & (timestamps < t_end)
        t_window = timestamps[mask]
        v_window = values[mask]

        if len(t_window) >= 3:
            peaks = _detect_ecg_peaks(t_window, v_window, min_rr_seconds)
            if len(peaks) >= beats_per_trace:
                for j in range(0, len(peaks) - beats_per_trace + 1, beats_per_trace):
                    slice_start = peaks[j]
                    slice_end = peaks[j + beats_per_trace - 1]
                    if slice_end + 1 < len(t_window):
                        slice_end += 1

                    _save_ecg_segment(
                        output_folder,
                        output_prefix,
                        trace_idx,
                        t_window[slice_start:slice_end + 1],
                        v_window[slice_start:slice_end + 1]
                    )
                    trace_idx += 1

                    if max_traces and trace_idx > max_traces:
                        break

                if max_traces and trace_idx > max_traces:
                    break

        i += np.sum(mask)
        if i <= 0:
            i += 1

        if max_traces and trace_idx > max_traces:
            break

    saved_count = trace_idx - 1
    limit_note = f" (limited to {max_traces})" if max_traces and saved_count >= max_traces else ""
    print(
        f"Saved {saved_count} ECG traces from {window_seconds}s windows "
        f"with {beats_per_trace} beats each{limit_note}."
    )


def extract_ecg_intervals_by_beats(
    input_file,
    output_folder,
    output_prefix,
    beats_per_trace=50,
    min_rr_seconds=0.3,
    max_traces=None
):
    """
    Split ECG data into traces by detected heartbeats.

    Args:
        input_file: formatted ECG CSV file
        output_folder: where to save traces
        output_prefix: prefix for output files
        beats_per_trace: number of detected beats per trace
        min_rr_seconds: minimum distance between peaks in seconds
    """
    os.makedirs(output_folder, exist_ok=True)

    data = np.genfromtxt(input_file, delimiter=';', dtype=str, skip_header=1)
    timestamps = data[:, 0].astype(float).astype(np.int64)
    values = data[:, 1].astype(float)

    if len(values) < 3:
        raise ValueError("ECG file must contain at least 3 samples")

    # Remove low-frequency baseline by centering around median
    signal = values - np.median(values)

    # Simple smoothing to reduce noise
    window = min(11, len(signal) if len(signal) % 2 == 1 else len(signal) - 1)
    if window < 3:
        window = 3
    kernel = np.ones(window) / float(window)
    smooth = np.convolve(signal, kernel, mode='same')

    # Use absolute amplitude for peak detection so inverted ECG still works
    abs_signal = np.abs(smooth)
    threshold = max(np.mean(abs_signal) + 0.5 *
                    np.std(abs_signal), np.percentile(abs_signal, 75))

    # Local maxima detection on the absolute smoothed signal
    candidates = np.where(
        (abs_signal[1:-1] > abs_signal[:-2]) &
        (abs_signal[1:-1] >= abs_signal[2:]) &
        (abs_signal[1:-1] > threshold)
    )[0] + 1

    if len(candidates) == 0:
        # Fallback: lower threshold to capture any peaks
        threshold = np.mean(abs_signal) + 0.25 * np.std(abs_signal)
        candidates = np.where(
            (abs_signal[1:-1] > abs_signal[:-2]) &
            (abs_signal[1:-1] >= abs_signal[2:]) &
            (abs_signal[1:-1] > threshold)
        )[0] + 1

    if len(candidates) == 0:
        raise ValueError("No heartbeat peaks found in ECG signal")

    # Enforce minimum distance between detected peaks
    min_distance_us = int(min_rr_seconds * 1_000_000)
    peaks = [candidates[0]]
    for idx in candidates[1:]:
        if timestamps[idx] - timestamps[peaks[-1]] >= min_distance_us:
            peaks.append(idx)
    peaks = np.array(peaks, dtype=int)

    if len(peaks) < beats_per_trace:
        raise ValueError(
            f"Not enough detected beats ({len(peaks)}) for one trace of {beats_per_trace} beats"
        )

    trace_idx = 1
    for i in range(0, len(peaks) - beats_per_trace + 1, beats_per_trace):
        slice_start = peaks[i]
        slice_end = peaks[i + beats_per_trace - 1]

        # Extend the slice slightly to include the interval after the last beat
        if slice_end + 1 < len(timestamps):
            slice_end = slice_end + 1

        t_chunk = timestamps[slice_start:slice_end + 1]
        v_chunk = values[slice_start:slice_end + 1]

        t0 = t_chunk[0]
        rel_time = (t_chunk - t0) / 1000.0

        out = np.column_stack((rel_time, v_chunk))
        np.savetxt(
            os.path.join(output_folder, f"{output_prefix}-tid{trace_idx}.csv"),
            out,
            delimiter=';',
            header="time_ms;ecg_value",
            fmt=['%.0f', '%.5f'],
            comments=''
        )
        trace_idx += 1

        if max_traces and trace_idx > max_traces:
            break

    saved_count = trace_idx - 1
    limit_note = f" (limited to {max_traces})" if max_traces and saved_count >= max_traces else ""
    print(
        f"Saved {saved_count} ECG traces of {beats_per_trace} beats each{limit_note}. "
        f"Detected {len(peaks)} heartbeat peaks."
    )


if __name__ == "__main__":
    from pathlib import Path

    BASE_DIR = Path(__file__).resolve().parent.parent.parent

    rooms = ["A", "B", "C", "D", "E", "F"]
    periods = [("1day", 1), ("7day", 7), ("30day", 30)]
    room_col_map = {"A": 2, "B": 3, "C": 4, "D": 5, "E": 6, "F": 7}
    raw_input_file = BASE_DIR / "Data" / "1-Raw" / \
        "dataset-2023-02-27_2023-12-31.csv"

    for room in rooms:
        formated_file = BASE_DIR / "Data" / \
            "2-FormatedRawData" / f"room{room}-formated.csv"

        if not formated_file.exists():
            print(f"Formatting room {room}...")
            format_temperature_data(
                raw_input_file, formated_file, col=room_col_map[room])
        else:
            print(f"Skipping formatting for room {room} (already exists)")

        for period, period_number in periods:
            experiment_folder = BASE_DIR / "Data" / "3-ExtractInterval" / \
                f"{period}-experiment" / f"room{room}"

            if experiment_folder.exists() and any(experiment_folder.iterdir()):
                print(
                    f"Skipping extraction for room {room} / {period} (already exists)")
                continue

            print(f"Extracting intervals for room {room} / {period}...")
            extract_time_intervals(
                formated_file, experiment_folder, f"room{room}-{period}", period_number)


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
