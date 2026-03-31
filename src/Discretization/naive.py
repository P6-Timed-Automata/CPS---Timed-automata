import numpy as np
import string

import Discretization.discretizationSetup

#
# def csv_to_temp_time_list(input_file):
#     # Load data
#     data = np.genfromtxt(
#         input_file,
#         delimiter=';',
#         dtype=str,
#         skip_header=1
#     )
#
#     # Extract columns
#     times = data[:, 0].astype(int)
#     temps = data[:, 1].astype(float)
#
#     # Build list of (temperature, time)
#     result = [(temp, time) for temp, time in zip(temps, times)]
#
#     print("tranformed data to a list")
#
#     return result
#
#
# def format_output(symbolic_res_list):
#     lines = []
#
#     for symbolic_res in symbolic_res_list:
#         line = " ".join(f"{s}:{v}" for s, v in symbolic_res)
#         lines.append(line)
#
#     output = "\n".join(lines)
#
#     with open("output.txt", "w") as f:
#         f.write(output)
#
#     print("File saved")
#
# def map_bins_to_symbols(result, k):
#     # Create symbols: a, b, c, ...
#     symbols = list(string.ascii_lowercase)
#
#     if k > len(symbols):
#         raise ValueError("k too large (max 26 supported with simple letters)")
#
#     # Create mapping: 0->'a', 1->'b', ...
#     mapping = {i: symbols[i] for i in range(k)}
#
#     # Apply mapping
#     symbolic_result = [(mapping[int(label)], int(time)) for label, time in result]
#
#     return symbolic_result, mapping

def equal_width_discretization(trace1, trace2, k):
    # Convert to arrays
    t1 = np.array([t for v, t in trace1])
    v1 = np.array([v for v, t in trace1])

    t2 = np.array([t for v, t in trace2])
    v2 = np.array([v for v, t in trace2])

    # Combine values from both traces
    combined_values = np.concatenate([v1, v2])

    # Equal-width bins
    min_val = np.min(combined_values)
    max_val = np.max(combined_values)
    bins = np.linspace(min_val, max_val, k + 1)

    # Discretize both traces using SAME bins
    labels1 = np.digitize(v1, bins) - 1
    labels2 = np.digitize(v2, bins) - 1


    # Fix edge cases
    labels1[labels1 == k] = k - 1
    labels2[labels2 == k] = k - 1


    # Reconstruct traces separately
    #trace1_discretized = list(zip(labels1, t1))
    #trace2_discretized = list(zip(labels2, t2))


    # Print bin intervals
    print("Bin intervals:")
    for i in range(k):
        print(f"Bin {i}: [{bins[i]} → {bins[i+1]})")

    #print("\nAssignments:")
    #for v, l in zip(values, labels):
        #print(f"Value {v} → Bin {l}")

    # Recombine (discretized value + original time)
    #result = list(zip(labels, values))

    trace1_discretized = [(int(l), int(t)) for l, t in zip(labels1, t1)]
    trace2_discretized = [(int(l), int(t)) for l, t in zip(labels2, t2)]

    return trace1_discretized,trace2_discretized, bins

input1_file = '../dataProcessing/formated_data.csv'
data1 = csv_to_temp_time_list(input1_file)

input2_file = '../dataProcessing/formated_data2.csv'
data2 = csv_to_temp_time_list(input2_file)




#data = [(10, 1), (15, 2), (8, 3), (20, 4)]

k = 3

trace1_discretized, trace2_discretized, bins = equal_width_discretization(data1,data2, k)

print("Bins:", bins)
print("Result1:", trace1_discretized)
print("Result2:", trace2_discretized)

symbolic_res1, mapping = map_bins_to_symbols(trace1_discretized, k)
symbolic_res2, __ = map_bins_to_symbols(trace2_discretized, k)

print("Mapping:", mapping)

print("Symbolic result:", symbolic_res1)
print("Symbolic result:", symbolic_res2)

symbolic_res_list = [symbolic_res1, symbolic_res2]

format_output(symbolic_res_list)

# import json

# Compute midpoints: symbol -> rounded midpoint value
# symbol_map = {
#     mapping[i]: round((bins[i] + bins[i + 1]) / 2*100)
#     for i in range(k)
# }
#
# with open('symbol_map.json', 'w') as f:
#     json.dump(symbol_map, f)

print("Symbol map:", symbol_map)