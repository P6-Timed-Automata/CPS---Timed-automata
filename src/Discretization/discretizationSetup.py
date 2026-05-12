import numpy as np
import string
import json
import os


def csv_to_temp_time_list(input_files):

    all_results = []
    for input_file in input_files:
        # Load data
        data = np.genfromtxt(
            input_file,
            delimiter=';',
            dtype=str,
            skip_header=1
        )

        # Extract columns
        times = data[:, 0].astype(int)
        temps = data[:, 1].astype(float)

        # Build list of (temperature, time)
        result = [(float(temp), int(time)) for temp, time in zip(temps, times)]

        all_results.append(result)


    print("tranformed data to a list")

    return all_results


# def format_output(symbolic_res_list,output_path):
#     lines = []
#
#     for symbolic_res in symbolic_res_list:
#         line = " ".join(f"{s}:{v}" for s, v in symbolic_res)
#         lines.append(line)
#
#     output = "\n".join(lines)
#
#     # Ensure the directory exists
#     os.makedirs(os.path.dirname(output_path), exist_ok=True)
#
#     with open(output_path, "w") as f:
#         f.write(output)
#
#     print(f"File saved to {output_path}")


def format_output(symbolic_res_list, output_path):

    lines = []

    for trace in symbolic_res_list:

        formatted = []
        prev_time = None

        for i, (symbol, value) in enumerate(trace):

            if i == 0:
                delay = 0
            else:
                delay = int(float(value) - float(prev_time))
                if delay < 0:
                    delay = 0

            prev_time = value

            formatted.append(f"{symbol}:{delay}")

        lines.append(" ".join(formatted))

    output = "\n".join(lines)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(output_path, "w") as f:
        f.write(output)

    print(f"File saved to {output_path}")

def map_bins_to_symbols(result, s, bins):
    # Create symbols: a, b, c, ...
    symbols = list(string.ascii_lowercase)

    if s > len(symbols):
        raise ValueError("s too large (max 26 supported with simple letters)")

    # Create mapping: 0->'a', 1->'b', ...
    mapping = {i: symbols[i] for i in range(s)}

    #Midpoint symbol map
    symbol_map = None
    if bins is not None:
        symbol_map = {
            symbols[i]: round(((bins[i] + bins[i + 1]) / 2) * 100)
            for i in range(s)
        }

    # Apply mapping to traces
    symbolic_results = []
    for trace in result:
        symbolic_trace = [(mapping[int(label)], int(time)) for label, time in trace]
        symbolic_results.append(symbolic_trace)


    return symbolic_results, symbol_map, mapping

#
#
# def preprocess_test_traces(test_traces, bins, s):
#     """
#     Convert raw test traces into TAG format using:
#     - training bins
#     - s = number of symbols (int)
#     """
#
#     # --------------------------
#     # 1. Create alphabet
#     # --------------------------
#     symbols = list(string.ascii_lowercase)
#
#     if s > len(symbols):
#         raise ValueError("s too large (max 26 supported)")
#
#     symbols = symbols[:s]
#     k = len(bins) - 1
#
#
#
#     # --------------------------
#     # 2. Build mapping (bin -> letter)
#     # --------------------------
#     mapping = {i: symbols[i] for i in range(s)}
#
#     # --------------------------
#     # 3. Discretize using TRAIN bins
#     # --------------------------
#     discretized = []
#
#     for trace in test_traces:
#         values = np.array([v for v, t in trace])
#         times = np.array([t for v, t in trace])
#
#         labels = np.digitize(values, bins) - 1
#         labels = np.clip(labels, 0, k - 1)
#
#         discretized.append([(int(l), int(t)) for l, t in zip(labels, times)])
#
#
#     # --------------------------
#     # 4. Convert to TAG format
#     # --------------------------
#     symbolic_traces = [
#         [f"{mapping[label]}:{t}" for (label, t) in trace]
#         for trace in discretized
#     ]
#
#
#     return symbolic_traces


#
# def preprocess_test_traces(test_traces, bins, s):
#
#     symbols = list(string.ascii_lowercase)[:s]
#     mapping = {i: symbols[i] for i in range(s)}
#
#     k = len(bins) - 1
#
#     symbolic_traces = []
#
#     for trace in test_traces:
#
#         values = np.array([v for v, t in trace])
#         times  = np.array([t for v, t in trace])
#
#         labels = np.digitize(values, bins) - 1
#         labels = np.clip(labels, 0, k - 1)
#
#         symbolic_trace = []
#
#         prev_time = None
#
#         for i, label in enumerate(labels):
#
#             if i == 0:
#                 delay = 0
#             else:
#                 delay = int(float(times[i]) - float(prev_time))
#                 if delay < 0:
#                     delay = 0
#
#             prev_time = times[i]
#
#             symbolic_trace.append(f"{mapping[label]}:{delay}")
#
#         symbolic_traces.append(symbolic_trace)
#
#     return symbolic_traces




def collapse_trace(trace):
    """
    Input:
        [('a', 0), ('a', 300), ('a', 300), ('c', 300)]

    Output:
        [('a', 900), ('c', 300)]
    """

    collapsed = []

    for symbol, delay in trace:

        if not collapsed:
            collapsed.append([symbol, delay])

        elif collapsed[-1][0] == symbol:
            collapsed[-1][1] += delay

        else:
            collapsed.append([symbol, delay])

    return [(s, d) for s, d in collapsed]


def timestamps_to_relative(trace):
    """
    Input:
        [('a', 0), ('a', 300), ('c', 600)]

    Output:
        [('a', 0), ('a', 300), ('c', 300)]
    """

    relative_trace = []

    prev_time = None

    for i, (symbol, time_value) in enumerate(trace):

        if i == 0:
            delay = 0
        else:
            delay = int(float(time_value) - float(prev_time))

            if delay < 0:
                delay = 0

        prev_time = time_value

        relative_trace.append((symbol, delay))

    return relative_trace


def preprocess_test_traces(test_traces, bins, s):

    symbols = list(string.ascii_lowercase)[:s]
    mapping = {i: symbols[i] for i in range(s)}

    k = len(bins) - 1

    symbolic_traces = []

    for trace in test_traces:

        values = np.array([v for v, t in trace])
        times = np.array([t for v, t in trace])

        labels = np.digitize(values, bins) - 1
        labels = np.clip(labels, 0, k - 1)

        # Step 1: symbolic timestamps
        symbolic_timestamp_trace = [
            (mapping[label], times[i])
            for i, label in enumerate(labels)
        ]

        # Step 2: relative delays
        relative_trace = timestamps_to_relative(symbolic_timestamp_trace)

        # Step 3: collapse repeated symbols
        collapsed_trace = collapse_trace(relative_trace)

        # Step 4: format for TAG
        formatted_trace = [
            f"{symbol}:{delay}"
            for symbol, delay in collapsed_trace
        ]

        symbolic_traces.append(formatted_trace)

    return symbolic_traces



def format_output(symbolic_res_list, output_path):

    lines = []

    for trace in symbolic_res_list:

        # 1. convert absolute time → relative delay
        relative_trace = timestamps_to_relative(trace)

        # 2. collapse consecutive identical symbols
        collapsed_trace = collapse_trace(relative_trace)

        # 3. format
        formatted = [
            f"{symbol}:{delay}"
            for symbol, delay in collapsed_trace
        ]

        lines.append(" ".join(formatted))

    output = "\n".join(lines)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(output_path, "w") as f:
        f.write(output)

    print(f"File saved to {output_path}")