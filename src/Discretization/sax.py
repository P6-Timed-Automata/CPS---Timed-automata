from scipy.stats import norm
import numpy as np

from Discretization.discretizationSetup import (
    csv_to_temp_time_list,
    format_output,
    map_bins_to_symbols
)

import matplotlib.pyplot as plt
import scipy.stats as stats


import numpy as np
from scipy.stats import norm

def sax_discretization_multi(data_lists, w, k):

    breakpoints = norm.ppf(np.linspace(0, 1, k + 1)[1:-1])

    # ---- SAFE GLOBAL NORMALIZATION ----
    all_v = np.concatenate([
        np.array([val for val, _ in trace])
        for trace in data_lists
        if len(trace) > 0
    ])

    global_mean = np.mean(all_v)
    global_std = np.std(all_v)

    if global_std == 0 or np.isnan(global_std):
        global_std = 1.0

    def safe_paa(v, t, w):
        """
        Robust PAA:
        - prevents empty segments
        - avoids NaN propagation
        """

        n = len(v)

        if n == 0:
            return np.array([]), np.array([])

        # ensure w does not exceed signal length
        w_eff = min(w, n)

        # split safely
        v_segs = np.array_split(v, w_eff)
        t_segs = np.array_split(t, w_eff)

        paa_v = []
        paa_t = []

        for vs, ts in zip(v_segs, t_segs):

            if len(vs) == 0:
                continue

            # signal PAA (safe)
            v_mean = np.nanmean(vs)
            if np.isnan(v_mean):
                v_mean = 0.0

            # time PAA (use midpoint, NOT mean cast)
            if len(ts) > 0:
                t_mean = float(np.nanmean(ts))
            else:
                t_mean = 0.0

            paa_v.append(v_mean)
            paa_t.append(int(t_mean))

        return np.array(paa_v), np.array(paa_t)

    discretized = []
    all_norm_vals = []

    for trace in data_lists:

        if len(trace) == 0:
            continue

        v = np.array([val for val, _ in trace], dtype=float)
        t = np.array([time for _, time in trace], dtype=float)

        # ---- SAFE NORMALIZATION ----
        norm_v = (v - global_mean) / global_std
        norm_v = np.nan_to_num(norm_v)

        all_norm_vals.extend(norm_v)

        paa_v, paa_t = safe_paa(norm_v, t, w)

        if len(paa_v) == 0:
            continue

        labels = np.digitize(paa_v, breakpoints, right=False)

        discretized.append([
            (int(l), int(ts))
            for l, ts in zip(labels, paa_t)
        ])

    # ---- SAFE BIN CONSTRUCTION ----
    all_norm_vals = np.array(all_norm_vals)

    if len(all_norm_vals) == 0:
        bins = breakpoints
    else:
        bins = np.concatenate((
            [np.min(all_norm_vals)],
            breakpoints,
            [np.max(all_norm_vals)]
        ))

    return discretized, bins, global_mean, global_std

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

def sax_bins_in_original_space(bins_z, global_mean, global_std):
    """
    Convert SAX bin edges from z-normalized space back to the original
    (un-normalized) value space.

    sax_discretization_multi returns bins in z-space because PAA segments
    are compared against Gaussian quantiles after normalization. For
    downstream code that operates in original space (MAE computation,
    symbol_map midpoints for UPPAAL, plotting), the bins must be
    converted back.

    Parameters
    ----------
    bins_z       : array of bin edges in z-space (output of sax_discretization_multi)
    global_mean  : mean used during normalization
    global_std   : std used during normalization

    Returns
    -------
    bins in original (un-normalized) value space, sorted ascending.
    """
    return np.sort(bins_z) * global_std + global_mean