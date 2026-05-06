#!/usr/bin/env python3
"""
Discretize ECG data from train/test/val splits.
Creates Naive discretization for each split.
"""

from Discretization.discretizationSetup import csv_to_temp_time_list, map_bins_to_symbols, format_output
from Discretization.naive import equal_width_discretization
from DataProcessing.processData import get_trace_files
import os
import sys
import json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))


BASE_DIR = Path(__file__).resolve().parent.parent

def normalize(traces, target=None):
    lengths = [len(t) for t in traces]
    target = target or int(sum(lengths) / len(lengths))

    normalized = []

    for t in traces:
        if len(t) > target:
            t = t[:target]
        else:
            t = t + [t[-1]] * (target - len(t))

        normalized.append(t)

    return normalized

def filter_equal_length(traces, tolerance=0.2):
    lengths = [len(t) for t in traces]
    target = sorted(lengths)[len(lengths)//2]  # median length

    lower = target * (1 - tolerance)
    upper = target * (1 + tolerance)

    filtered = [t for t in traces if lower <= len(t) <= upper]

    print(f"Target length: {target}")
    print(f"Kept {len(filtered)}/{len(traces)} traces")

    return filtered

def discretize_experiment(experiment_folder, output_base, experiment_name, method="naiv", num_symbols=4, max_traces=None):
    """
    Discretize ECG traces using equal-width (naive) method.

    Args:
        experiment_folder: Path to folder with trace files
        output_base: Base output directory
        experiment_name: Name for output subfolder
        method: Discretization method name ('naiv', 'persist', 'sax')
        num_symbols: Number of symbols to use (4 = a,b,c,d)
        max_traces: Maximum number of traces to process (None = all traces)
    """

    # Create output directory as <output_base>/<method>/<experiment_name>
    output_dir = Path(output_base) / method / experiment_name
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nProcessing: {experiment_folder}")
    print(f"Output: {output_dir}\n")

    # Get all trace files
    trace_files = get_trace_files(experiment_folder, extension=".csv")

    if not trace_files:
        print(f"No CSV files found in {experiment_folder}")
        return

    print(f"Found {len(trace_files)} trace files")

    # Limit number of files if specified
    if max_traces is not None and len(trace_files) > max_traces:
        trace_files = trace_files[:max_traces]
        print(f"Using {len(trace_files)} traces (limited to {max_traces})")

    # Load traces
    traces = csv_to_temp_time_list(trace_files)
    print(f"Loaded {len(traces)} traces")

    # normalize lengths BEFORE discretization
    traces = normalize(traces)
    print("After normalization:", len(traces), "traces")

    # Filter traces to equal length
    traces = filter_equal_length(traces)

    # NAIVE DISCRETIZATION (equal-width)
    print(
        f"\nApplying Naive (equal-width) discretization (s={num_symbols})...")
    for trace_nr in range(1, len(traces) + 1):

        current_traces = traces[:trace_nr] 

        # Discretize
        naive_traces, naive_bins = equal_width_discretization(current_traces, num_symbols)
        naive_symbolic, naive_map, _ = map_bins_to_symbols(
            naive_traces, num_symbols, naive_bins)

        # Dynamic filename
        filename = f"ecg-{trace_nr}trace-{experiment_name}-{method}-s{num_symbols}-trace.txt"
        output_path = output_dir / filename

        format_output(naive_symbolic, str(output_path))

        print(f"  ✓ Saved: {filename}")

    # Save symbol map
    with open(output_dir / "symbol_map.json", 'w') as f:
        json.dump(naive_map, f, indent=2)
    print(f"  ✓ Saved to {filename}")

    print(f"\n✓ Discretization complete for {experiment_name}")


if __name__ == "__main__":
    discretization_output = BASE_DIR / "Data" / "4-DiscretizationData"
    ecg_experiment_base = BASE_DIR / "Data" / \
        "3-ExtractInterval" / "ecg-experimenrt"

    # ===== CONFIGURATION =====
    # Set max_traces to limit number of traces per dataset (None = all)
    MAX_TRAIN_TRACES = 30  # Set to e.g., 20 to use only 20 training traces
    MAX_TEST_TRACES = 30
    MAX_VAL_TRACES = 20
    NUM_SYMBOLS = 4  # Number of symbols (a, b, c, d)

    # Discretize training data
    print("=" * 60)
    print("DISCRETIZING TRAINING DATA")
    print("=" * 60)
    discretize_experiment(
        experiment_folder=str(ecg_experiment_base / "train"),
        output_base=str(discretization_output),
        experiment_name="ecg-train",
        method="naiv",
        num_symbols=NUM_SYMBOLS,
        max_traces=MAX_TRAIN_TRACES
    )

    # Discretize test data
    print("\n" + "=" * 60)
    print("DISCRETIZING TEST DATA")
    print("=" * 60)
    discretize_experiment(
        experiment_folder=str(ecg_experiment_base / "test"),
        output_base=str(discretization_output),
        experiment_name="ecg-test",
        method="naiv",
        num_symbols=NUM_SYMBOLS,
        max_traces=MAX_TEST_TRACES
    )

    # Discretize validation data
    print("\n" + "=" * 60)
    print("DISCRETIZING VALIDATION DATA")
    print("=" * 60)
    discretize_experiment(
        experiment_folder=str(ecg_experiment_base / "val"),
        output_base=str(discretization_output),
        experiment_name="ecg-val",
        method="naiv",
        num_symbols=NUM_SYMBOLS,
        max_traces=MAX_VAL_TRACES
    )

    print("\n" + "=" * 60)
    print("✓ All discretizations complete!")
    print("✓ Output: Data/4-DiscretizationData/naiv/ecg-train/")
    print("✓ Output: Data/4-DiscretizationData/naiv/ecg-test/")
    print("✓ Output: Data/4-DiscretizationData/naiv/ecg-val/")
    print("=" * 60)
