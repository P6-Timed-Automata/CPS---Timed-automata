#!/usr/bin/env python3
"""Extract ECG beat traces and split them into train/test/val sets."""

from processData import (
    extract_ecg_intervals_by_beats,
    extract_ecg_intervals_by_time_window,
    extract_ecg_intervals_by_time_and_beats,
    extract_ecg_intervals_by_samples,
)
import os
import random
import shutil
import sys
from pathlib import Path

# Source file is inside src/DataProcessing, so repo root is two levels up.
ROOT_DIR = Path(__file__).resolve().parent.parent.parent

# Add the DataProcessing folder to sys.path so local imports work.
PROCESSING_DIR = Path(__file__).resolve().parent
if str(PROCESSING_DIR) not in sys.path:
    sys.path.insert(0, str(PROCESSING_DIR))


# Configuration
FORMATTED_ECG_FILE = ROOT_DIR / "Data-marco" / \
    "Formatted" / "patient_100_ecg_formatted.csv"
SOURCE_DIR = ROOT_DIR / "Data" / "3-ExtractInterval" / "ecg-experimenrt"
EXTRACTION_DIR = SOURCE_DIR / "beats"
TRAIN_DIR = SOURCE_DIR / "train"
TEST_DIR = SOURCE_DIR / "test"
VAL_DIR = SOURCE_DIR / "val"

TRAIN_RATIO = 0.70
TEST_RATIO = 0.15
VAL_RATIO = 0.15
RANDOM_SEED = 42
# Extraction method: beats, window, window_and_beats, samples
EXTRACTION_METHOD = "samples"
WINDOW_SECONDS = 10
BEATS_PER_TRACE = 1
MIN_RR_SECONDS = 0.3
SAMPLES_PER_TRACE = 1000  # Fixed number of samples per trace for consistent length
OUTPUT_PREFIX = "patient100-beats"
# Limit total number of extracted traces to reduce file count.
MAX_TRACES = 30


def create_directories():
    """Create train, test, val subdirectories."""
    for directory in [TRAIN_DIR, TEST_DIR, VAL_DIR]:
        os.makedirs(directory, exist_ok=True)
        print(f"Created/verified: {directory}")


def extract_beat_traces(force=False):
    """Extract ECG traces by heartbeat peaks into the extraction folder."""
    if not FORMATTED_ECG_FILE.exists():
        raise FileNotFoundError(
            f"Formatted ECG file not found: {FORMATTED_ECG_FILE}")

    if force and os.path.exists(EXTRACTION_DIR):
        shutil.rmtree(EXTRACTION_DIR)

    os.makedirs(EXTRACTION_DIR, exist_ok=True)
    existing_files = [f for f in os.listdir(
        EXTRACTION_DIR) if f.endswith('.csv')]
    if existing_files:
        print(
            f"Extraction folder already contains {len(existing_files)} CSV files. Skipping extraction.")
        return

    print(
        f"Extracting ECG traces to {EXTRACTION_DIR} using method={EXTRACTION_METHOD}...")
    if EXTRACTION_METHOD == "beats":
        extract_ecg_intervals_by_beats(
            input_file=str(FORMATTED_ECG_FILE),
            output_folder=str(EXTRACTION_DIR),
            output_prefix=OUTPUT_PREFIX,
            beats_per_trace=BEATS_PER_TRACE,
            min_rr_seconds=MIN_RR_SECONDS,
            max_traces=MAX_TRACES
        )
    elif EXTRACTION_METHOD == "window":
        extract_ecg_intervals_by_time_window(
            input_file=str(FORMATTED_ECG_FILE),
            output_folder=str(EXTRACTION_DIR),
            output_prefix=OUTPUT_PREFIX,
            window_seconds=WINDOW_SECONDS
        )
    elif EXTRACTION_METHOD == "window_and_beats":
        extract_ecg_intervals_by_time_and_beats(
            input_file=str(FORMATTED_ECG_FILE),
            output_folder=str(EXTRACTION_DIR),
            output_prefix=OUTPUT_PREFIX,
            window_seconds=WINDOW_SECONDS,
            beats_per_trace=BEATS_PER_TRACE,
            min_rr_seconds=MIN_RR_SECONDS,
            max_traces=MAX_TRACES
        )
    elif EXTRACTION_METHOD == "samples":
        extract_ecg_intervals_by_samples(
            input_file=str(FORMATTED_ECG_FILE),
            output_folder=str(EXTRACTION_DIR),
            output_prefix=OUTPUT_PREFIX,
            samples_per_trace=SAMPLES_PER_TRACE,
            max_traces=MAX_TRACES
        )
    else:
        raise ValueError(f"Unknown EXTRACTION_METHOD: {EXTRACTION_METHOD}")


def split_files():
    """Split extracted beat trace files into train/test/val sets."""
    csv_files = [f for f in os.listdir(EXTRACTION_DIR)
                 if f.endswith('.csv') and os.path.isfile(os.path.join(EXTRACTION_DIR, f))]

    if not csv_files:
        raise FileNotFoundError(
            f"No ECG beat trace files found in {EXTRACTION_DIR}")

    random.seed(RANDOM_SEED)
    random.shuffle(csv_files)

    total_files = len(csv_files)
    train_count = int(total_files * TRAIN_RATIO)
    test_count = int(total_files * TEST_RATIO)

    train_files = csv_files[:train_count]
    test_files = csv_files[train_count:train_count + test_count]
    val_files = csv_files[train_count + test_count:]

    for file in train_files:
        shutil.copy2(os.path.join(EXTRACTION_DIR, file),
                     os.path.join(TRAIN_DIR, file))
    for file in test_files:
        shutil.copy2(os.path.join(EXTRACTION_DIR, file),
                     os.path.join(TEST_DIR, file))
    for file in val_files:
        shutil.copy2(os.path.join(EXTRACTION_DIR, file),
                     os.path.join(VAL_DIR, file))

    print(f"\nSplit Summary:")
    print(f"  Total files: {total_files}")
    print(f"  Train files: {len(train_files)} ({TRAIN_RATIO*100:.0f}%)")
    print(f"  Test files: {len(test_files)} ({TEST_RATIO*100:.0f}%)")
    print(f"  Val files: {len(val_files)} ({VAL_RATIO*100:.0f}%)")


if __name__ == "__main__":
    create_directories()
    extract_beat_traces(force=True)

    print(
        f"\nSplitting files with ratios: Train={TRAIN_RATIO*100:.0f}%, Test={TEST_RATIO*100:.0f}%, Val={VAL_RATIO*100:.0f}%")
    split_files()
    print("\nDone! Files have been split into train/test/val subdirectories.")
