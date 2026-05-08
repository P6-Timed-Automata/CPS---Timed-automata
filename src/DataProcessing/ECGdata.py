#!/usr/bin/env python3
"""
ECG → FORMATTED → PEAKS → FIXED-LENGTH TA-SAFE TRACES

TA FIXES:
- integer time only (NO decimals)
- normalized & scaled time (0 → 1000)
- fixed trace length
- deterministic peak grouping
"""

import os
import numpy as np
from pathlib import Path

# ================= CONFIG =================
ROOT_DIR = Path(__file__).resolve().parent.parent.parent

RAW_INPUT_FILE = ROOT_DIR / "Data-marco" / "patient_100_ecg.csv"
FORMATTED_FILE = ROOT_DIR / "Data-marco" / "Formatted" / "ecg_formatted.csv"

OUTPUT_DIR = ROOT_DIR / "Data" / "3-ExtractInterval" / "ecg-experiment"

# -------- CONTROL --------
MAX_SAMPLES = 200000
TRACE_LENGTH = 256          # ALL traces identical length
BEATS_PER_TRACE = 2
MIN_RR_SECONDS = 0.3

# TA TIME SCALING (IMPORTANT FIX)
TA_TIME_SCALE = 1000        # time becomes 0..1000 integers


# ================= STEP 1: FORMAT =================
def format_ecg_data(input_file, output_file, max_samples=None):
    data = np.genfromtxt(input_file, delimiter=',', dtype=str, skip_header=1)

    timestamps = data[:, 0].astype(np.int64)
    values = data[:, 1].astype(float)

    mask = np.isfinite(values)
    timestamps = timestamps[mask]
    values = values[mask]

    order = np.argsort(timestamps)
    timestamps = timestamps[order]
    values = values[order]

    values = values - np.median(values)

    # HARD LIMIT (deterministic)
    if max_samples is not None and len(values) > max_samples:
        timestamps = timestamps[:max_samples]
        values = values[:max_samples]

    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    np.savetxt(
        output_file,
        np.column_stack((timestamps, values)),
        delimiter=';',
        header='timestamp_us;ecg',
        fmt=['%d', '%.6f'],
        comments=''
    )

    print(f"[FORMAT] {len(values)} samples saved")


# ================= STEP 2: PEAK DETECTION =================
def detect_peaks(t, x, min_rr_s=0.3):
    x = x - np.median(x)

    smooth = np.convolve(x, np.ones(7)/7, mode='same')
    abs_x = np.abs(smooth)

    threshold = np.mean(abs_x) + 0.6 * np.std(abs_x)

    candidates = np.where(
        (abs_x[1:-1] > abs_x[:-2]) &
        (abs_x[1:-1] > abs_x[2:]) &
        (abs_x[1:-1] > threshold)
    )[0] + 1

    min_rr_us = int(min_rr_s * 1_000_000)

    peaks = []
    for p in candidates:
        if len(peaks) == 0 or (t[p] - t[peaks[-1]] > min_rr_us):
            peaks.append(p)

    return np.array(peaks)


# ================= STEP 3: SPLIT =================
def split_peaks(peaks):
    n = len(peaks)
    train_end = int(n * 0.7)
    test_end = int(n * 0.85)

    return peaks[:train_end], peaks[train_end:test_end], peaks[test_end:]


# ================= STEP 4: TA-SAFE FIXED TRACES =================
def build_fixed_traces(t, x, peaks, out_dir, prefix):
    os.makedirs(out_dir, exist_ok=True)

    idx = 0

    for i in range(0, len(peaks) - BEATS_PER_TRACE, BEATS_PER_TRACE):

        seg = peaks[i:i + BEATS_PER_TRACE]

        start = seg[0]
        end = seg[-1]

        t_seg = t[start:end + 1]
        x_seg = x[start:end + 1]

        # enforce fixed length
        if len(t_seg) < TRACE_LENGTH:
            continue

        t_seg = t_seg[:TRACE_LENGTH]
        x_seg = x_seg[:TRACE_LENGTH]

        # ================= TA FIX =================
        # convert time → relative → scaled integer
        t0 = t_seg[0]
        rel = (t_seg - t0).astype(np.float64)

        if rel[-1] == 0:
            continue

        rel_norm = rel / rel[-1]

        # SCALE TO INTEGER RANGE (NO FLOATS)
        rel_int = (rel_norm * TA_TIME_SCALE).astype(np.int64)

        # enforce monotonicity
        if not np.all(np.diff(rel_int) >= 0):
            continue

        # pad to fixed length if needed
        if len(rel_int) < TRACE_LENGTH:
            pad = TRACE_LENGTH - len(rel_int)
            rel_int = np.pad(rel_int, (0, pad), mode='edge')
            x_seg = np.pad(x_seg, (0, pad), mode='edge')

        np.savetxt(
            os.path.join(out_dir, f"{prefix}-{idx}.csv"),
            np.column_stack((rel_int, x_seg)),
            delimiter=';',
            fmt=['%d', '%.6f'],
            header='time;ecg',
            comments=''
        )

        idx += 1

    print(f"[{prefix}] {idx} TA-safe fixed traces")


# ================= MAIN =================
def main():

    format_ecg_data(RAW_INPUT_FILE, FORMATTED_FILE, MAX_SAMPLES)

    data = np.genfromtxt(FORMATTED_FILE, delimiter=';', skip_header=1)

    t = data[:, 0].astype(np.int64)
    x = data[:, 1].astype(float)

    peaks = detect_peaks(t, x, MIN_RR_SECONDS)
    print(f"Detected peaks: {len(peaks)}")

    train_p, test_p, val_p = split_peaks(peaks)

    print(f"Train peaks: {len(train_p)}")
    print(f"Test peaks: {len(test_p)}")
    print(f"Val peaks: {len(val_p)}")

    build_fixed_traces(t, x, train_p, OUTPUT_DIR / "train", "train")
    build_fixed_traces(t, x, test_p, OUTPUT_DIR / "test", "test")
    build_fixed_traces(t, x, val_p, OUTPUT_DIR / "val", "val")

    print("\nDONE: TA pipeline fully stabilized")


if __name__ == "__main__":
    main()