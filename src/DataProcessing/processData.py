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

import os
import numpy as np
from scipy.signal import find_peaks


def extract_fixed_beats_traces(
        input_file,
        output_folder,
        output_prefix,
        n_beats=3,
        pre_peak=50,
        post_peak=50,
        min_distance=200,
        stride=None,
        seed=None
):
    """
    Extract fixed-length multi-beat ECG traces.

    Each trace contains EXACTLY n_beats peaks.
    """

    os.makedirs(output_folder, exist_ok=True)

    if seed is not None:
        np.random.seed(seed)

    data = np.genfromtxt(input_file, delimiter=';', skip_header=1)

    times = data[:, 0]
    values = data[:, 1]

    peaks, _ = find_peaks(values, distance=min_distance)

    trace_idx = 1

    if stride is None:
        stride = n_beats  # non-overlapping by default

    i = 0

    while i + n_beats <= len(peaks):

        selected_peaks = peaks[i:i + n_beats]

        start = max(0, selected_peaks[0] - pre_peak)
        end = min(len(values), selected_peaks[-1] + post_peak)

        segment_t = times[start:end]
        segment_x = values[start:end]

        # reset time
        t0 = segment_t[0]
        rel_time = segment_t - t0

        out = np.column_stack((rel_time, segment_x))

        out_path = os.path.join(
            output_folder,
            f"{output_prefix}-tid{trace_idx}.csv"
        )

        np.savetxt(
            out_path,
            out,
            delimiter=';',
            header="time_seconds;ecg_value",
            fmt=['%.0f', '%.5f'],
            comments=''
        )

        trace_idx += 1
        i += stride

    print(f"Saved {trace_idx - 1} fixed-beat traces (n_beats={n_beats})")


def extract_ecg_intervals_by_samples(input_file, output_folder, output_prefix, samples_per_trace=500):
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

    print(
        f"Saved {trace_idx - 1} ECG traces of {samples_per_trace} samples each.")


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


def extract_ecg_intervals_by_beats(
    input_file,
    output_folder,
    output_prefix,
    beats_per_trace=50,
    min_rr_seconds=0.3
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

    print(
        f"Saved {trace_idx - 1} ECG traces of {beats_per_trace} beats each. "
        f"Detected {len(peaks)} heartbeat peaks."
    )

def convert_ns_to_ms(input_file, output_file):

    with open(input_file, "r") as f:
        lines = f.readlines()

    converted = []

    for line in lines[1:]:

        line = line.strip()
        if not line:
            continue

        time_ns, value = line.split(";")

        # nanoseconds -> milliseconds
        time_ms = float(time_ns) / 1_000_000

        converted.append((time_ms, float(value)))

    with open(output_file, "w") as f:

        f.write("timestamp_ms;ecg_value\n")

        for time_ms, v in converted:
            f.write(f"{time_ms:.3f};{v:.5f}\n")

    print(f"Converted file saved to: {output_file}")

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


