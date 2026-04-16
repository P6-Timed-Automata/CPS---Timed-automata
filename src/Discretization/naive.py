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



