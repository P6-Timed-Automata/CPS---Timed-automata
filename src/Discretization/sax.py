from scipy.stats import norm
import numpy as np

from Discretization.discretizationSetup import (
    csv_to_temp_time_list,
    format_output,
    map_bins_to_symbols
)

import matplotlib.pyplot as plt
import scipy.stats as stats


def sax_discretization_multi(data_lists, w, k):
    breakpoints = norm.ppf(np.linspace(0, 1, k + 1)[1:-1])

    def znorm(v):
        sigma = v.std()
        return (v - v.mean()) / sigma if sigma != 0 else np.zeros_like(v)

    def paa(v, t, w):
        v_segs = np.array_split(v, w)
        t_segs = np.array_split(t, w)
        return (
            np.array([seg.mean() for seg in v_segs]),
            np.array([int(seg.mean()) for seg in t_segs])
        )

    discretized = []
    for trace in data_lists:
        v = np.array([val for val, _ in trace])
        t = np.array([time for _, time in trace])
        paa_v, paa_t = paa(znorm(v), t, w)
        labels = np.digitize(paa_v, breakpoints)
        discretized.append([(int(l), int(ts)) for l, ts in zip(labels, paa_t)])

    return discretized, breakpoints

def sax_discretization(trace1, trace2, w, k):
    """
    w: number of PAA segments (output length per trace)
    k: alphabet size (number of symbols)
    """
    v1 = np.array([v for v, t in trace1])
    t1 = np.array([t for v, t in trace1])
    v2 = np.array([v for v, t in trace2])
    t2 = np.array([t for v, t in trace2])

    #Z-normalize
    def znorm(v):
        sigma = v.std()
        return (v - v.mean()) / sigma

    v1_norm = znorm(v1)
    v2_norm = znorm(v2)

    #PAA
    #Reduces n points to w segment means
    def paa(v, t, w):
        v_segs = np.array_split(v, w)
        t_segs = np.array_split(t, w)
        means = np.array([seg.mean() for seg in v_segs])
        midpoints = np.array([int(seg.mean()) for seg in t_segs])
        return means, midpoints

    paa_v1, paa_t1 = paa(v1_norm, t1, w)
    paa_v2, paa_t2 = paa(v2_norm, t2, w)

    #Gaussian breakpoints
    breakpoints = norm.ppf(np.linspace(0, 1, k + 1)[1:-1])

    labels1 = np.digitize(paa_v1, breakpoints)
    labels2 = np.digitize(paa_v2, breakpoints)

    trace1_discretized = [(int(l), int(t)) for l, t in zip(labels1, paa_t1)]
    trace2_discretized = [(int(l), int(t)) for l, t in zip(labels2, paa_t2)]

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

    return trace1_discretized, trace2_discretized, breakpoints

if __name__ == "__main__":
    input1_file = '../../Data/3-ExtractInterval/1day-experiment/roomA/roomA-1day-tid1.csv'
    input2_file = '../../Data/3-ExtractInterval/1day-experiment/roomA/roomA-1day-tid2.csv'

    data1 = csv_to_temp_time_list([input1_file])[0]
    data2 = csv_to_temp_time_list([input2_file])[0]

    w = 10
    k = 3

    trace1_discretized, trace2_discretized, bins = sax_discretization(data1, data2, w, k)

    print("Bins:", bins)
    print("Result1:", trace1_discretized)
    print("Result2:", trace2_discretized)


    #add outer edges so it workks with map_bins_to_symbols
    bins_with_edges = np.concatenate([[-3.0], bins, [3.0]])

    symbolic_res1, symbol_map, mapping = map_bins_to_symbols([trace1_discretized], k, bins_with_edges)
    symbolic_res2, _, _ = map_bins_to_symbols([trace2_discretized], k, bins_with_edges)



    print("Mapping:", mapping)
    print("Symbolic result 1:", symbolic_res1)
    print("Symbolic result 2:", symbolic_res2)

    symbolic_res1 = symbolic_res1[0]
    symbolic_res2 = symbolic_res2[0]

    symbolic_res_list = [symbolic_res1, symbolic_res2]

    # Updated to include output_path
    format_output(symbolic_res_list, output_path="sax_output_test/output.txt")
