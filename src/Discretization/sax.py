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

    data_min = np.min(all_v)
    data_max = np.max(all_v)

    bins = np.concatenate((
        [data_min],
        (breakpoints * global_std) + global_mean,
        [data_max]
    ))

    return discretized, bins, breakpoints, global_mean, global_std


if __name__ == "__main__":

    k = 3

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