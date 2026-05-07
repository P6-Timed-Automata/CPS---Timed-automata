import numpy as np
import string


def equal_width_discretization(traces, k):

    all_values = []

    # Extract all values across all traces
    for trace in traces:
        values = [val for val, t in trace]
        all_values.extend(values)

    all_values = np.array(all_values)

    # Equal-width bins
    min_val = np.min(all_values)
    max_val = np.max(all_values)
    bins = np.linspace(min_val, max_val, k + 1)

    # Discretize each trace
    discretized_traces = []

    for trace in traces:
        values = np.array([v for v, t in trace])
        times = np.array([t for v, t in trace])

        labels = np.digitize(values, bins) - 1

        # Fix edge cases
        labels[labels == k] = k - 1

        # Reconstruct trace
        discretized_trace = [(int(l), int(t)) for l, t in zip(labels, times)]
        discretized_traces.append(discretized_trace)

    # Print bin intervals
    print("Bin intervals:")
    for i in range(k):
        print(f"Bin {i}: [{bins[i]} → {bins[i+1]})")

    #print("\nAssignments:")
    #for v, l in zip(values, labels):
    #print(f"Value {v} → Bin {l}")


    return discretized_traces, bins


def ecg_trace_to_tag_input(
    input_file,
    n_symbols=5,
    min_rr_ms=300.0,
    global_bins=None
):

    data = np.genfromtxt(
        input_file,
        delimiter=';',
        skip_header=1
    )

    # FIX TIMESTAMPS

    samples = data[:, 0]

    # Adjust this if your ECG has another sampling rate
    times_ms = samples / 1000.0

    values = data[:, 1]

    print(times_ms[:10])

    # Detect peaks

    signal = values - np.median(values)

    abs_sig = np.abs(signal)

    threshold = (
        np.mean(abs_sig)
        + 0.5 * np.std(abs_sig)
    )

    candidates = np.where(
        (abs_sig[1:-1] > abs_sig[:-2]) &
        (abs_sig[1:-1] >= abs_sig[2:]) &
        (abs_sig[1:-1] > threshold)
    )[0] + 1

    peaks = [candidates[0]] if len(candidates) else []

    for idx in candidates[1:]:

        if (
            times_ms[idx]
            - times_ms[peaks[-1]]
        ) >= min_rr_ms:

            peaks.append(idx)

    if len(peaks) < 3:
        return None, global_bins

    # RR intervals

    peak_times = times_ms[np.array(peaks)]

    rr_intervals = np.diff(peak_times)

    # Discretize RR intervals

    labels = np.clip(
        np.digitize(rr_intervals, global_bins) - 1,
        0,
        n_symbols - 1
    )

    # RETURN FORMAT COMPATIBLE WITH PIPELINE

    discretized_trace = [
        (int(label), int(rr))
        for label, rr in zip(labels, rr_intervals)
    ]

    return [discretized_trace], global_bins