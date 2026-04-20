from scipy.stats import norm
import numpy as np
import string


import matplotlib.pyplot as plt
import scipy.stats as stats


def csv_to_temp_time_list(input_file):
    # Load data
    data = np.genfromtxt(
        input_file,
        delimiter=';',
        dtype=str,
        skip_header=1
    )
½½
    # Extract columns
    times = data[:, 0].astype(int)
    temps = data[:, 1].astype(float)

    # Build list of (temperature, time)
    result = [(temp, time) for temp, time in zip(temps, times)]

    print("tranformed data to a list")

    return result


def format_output(symbolic_res_list):
    lines = []

    for symbolic_res in symbolic_res_list:
        line = " ".join(f"{s}:{v}" for s, v in symbolic_res)
        lines.append(line)

    output = "\n".join(lines)

    with open("output.txt", "w") as f:
        f.write(output)

    print("File saved")

def map_bins_to_symbols(result, k):
    # Create symbols: a, b, c, ...
    symbols = list(string.ascii_lowercase)

    if k > len(symbols):
        raise ValueError("k too large (max 26 supported with simple letters)")

    # Create mapping: 0->'a', 1->'b', ...
    mapping = {i: symbols[i] for i in range(k)}

    # Apply mapping
    symbolic_result = [(mapping[int(label)], int(time)) for label, time in result]

    return symbolic_result, mapping

    # After znorm and PAA
    combined_paa = np.concatenate([paa_v1, paa_v2])

    # # Histogram
    # plt.figure(figsize=(12, 4))
    # plt.subplot(1, 2, 1)
    # plt.hist(combined_paa, bins=20, density=True, alpha=0.6)
    # x = np.linspace(-3, 3, 100)
    # plt.plot(x, stats.norm.pdf(x), 'r-', label='N(0,1)')
    # plt.title("Histogram of PAA means")
    # plt.legend()
    #
    # # Q-Q plot
    # plt.subplot(1, 2, 2)
    # stats.probplot(combined_paa, dist="norm", plot=plt)
    # plt.title("Q-Q Plot")
    #
    # plt.tight_layout()
    # plt.show()

#
# input1_file = '../../Data/FormattedData/experiment_5h/formated_data.csv'
# data1 = csv_to_temp_time_list(input1_file)
#
# input2_file = '../../Data/FormattedData/experiment_5h/formated_data2.csv'
# data2 = csv_to_temp_time_list(input2_file)
#
# w = 10
# k = 3
#
# trace1_discretized, trace2_discretized, bins = sax_discretization(data1, data2, w, k)
#
# print("Bins:", bins)
# print("Result1:", trace1_discretized)
# print("Result2:", trace2_discretized)
#
# symbolic_res1, mapping = map_bins_to_symbols(trace1_discretized, k)
# symbolic_res2, __ = map_bins_to_symbols(trace2_discretized, k)
#
# print("Mapping:", mapping)
#
# print("Symbolic result:", symbolic_res1)
# print("Symbolic result:", symbolic_res2)
#
# symbolic_res_list = [symbolic_res1, symbolic_res2]
#
# format_output(symbolic_res_list)

