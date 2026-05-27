"""Equal-width discretization for time-series traces."""

import numpy as np


def equal_width_discretization(traces, k, verbose=False):
    """
    Assign each sample to one of k equal-width bins spanning the global
    [min, max] of all training values.

    Parameters
    ----------
    traces  : list of [(value, time), ...]
    k       : number of bins (alphabet size)
    verbose : if True, prints bin edges for inspection

    Returns
    -------
    discretized_traces : list of [(bin_label_int, time_int), ...]
                         where bin_label_int is in [0, k-1].
    bins               : array of bin edges, length k+1.
    """
    all_values = np.concatenate([
        np.array([v for v, _ in trace], dtype=float)
        for trace in traces
    ])

    min_val = float(np.min(all_values))
    max_val = float(np.max(all_values))
    bins = np.linspace(min_val, max_val, k + 1)

    discretized_traces = []
    for trace in traces:
        values = np.array([v for v, _ in trace], dtype=float)
        times = np.array([t for _, t in trace])

        # that valid bin indices are 0..k-1, then handle the top edge.
        labels = np.digitize(values, bins) - 1
        labels = np.where(values == bins[-1], k - 1, labels)

        discretized_trace = [(int(l), int(t)) for l, t in zip(labels, times)]
        discretized_traces.append(discretized_trace)

    if verbose:
        print("Bin intervals:")
        for i in range(k):
            closing = "]" if i == k - 1 else ")"
            print(f"  Bin {i}: [{bins[i]:.4f}, {bins[i+1]:.4f}{closing}")

    return discretized_traces, bins