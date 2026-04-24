import os
import string
import numpy as np
from scipy.stats import norm

from TAG.TALearner import TALearner
from Discretization.discretizationSetup import format_output
from Discretization.sax import csv_to_temp_time_list
from DataProcessing.processData import get_trace_files
from GraphGeneration.graphs import plot_discretized_traces


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


def map_bins_to_symbols_multi(traces, k):
    if k > 26:
        raise ValueError("k > 26 not supported with single-letter symbols")
    mapping = {i: string.ascii_lowercase[i] for i in range(k)}
    symbolic = [[(mapping[label], time) for label, time in trace] for trace in traces]
    return symbolic, mapping, mapping


def build_temp_symbol_map(symbol_map, bins, mean, std):
    """
    Builds {letter: median_temperature_celsius} for export_ta.
    symbol_map: {bin_index: letter}
    bins:       SAX Gaussian breakpoints (z-scores), length k-1
    """
    boundaries = np.concatenate([[-np.inf], bins, [np.inf]])
    temp_map = {}
    for bin_idx, letter in symbol_map.items():
        lo = boundaries[bin_idx]
        hi = boundaries[bin_idx + 1]
        if np.isinf(lo):
            z_mid = hi - 1.0
        elif np.isinf(hi):
            z_mid = lo + 1.0
        else:
            z_mid = (lo + hi) / 2.0
        temp_map[letter] = round(z_mid * std + mean, 2)
    return temp_map


# --- Config ---
room = "A"
w = 200 # controls x axis binning
discretization_method = "sax"
period = "7day" #1day, 7day, 30day

alphabet_sizes = range(15, 16, 1)  # SAX alphabet: controls discretization
k_future = 4                       # TAG lookahead: controls TA merging

experiment_folder = f"../Data/3-ExtractInterval/{period}-experiment/room{room}/"

for trace_nr in range(9, 10):
    raw_traces = get_trace_files(folder_path=experiment_folder, max_files=trace_nr)
    data_lists = [csv_to_temp_time_list(f) for f in raw_traces]

    # Compute once per trace_nr — mean/std don't change with alphabet
    all_temps = np.concatenate([np.array([v for v, _ in trace]) for trace in data_lists])
    global_mean = all_temps.mean()
    global_std  = all_temps.std()

    for alphabet in alphabet_sizes:
        traces, breakpoints = sax_discretization_multi(data_lists, w=w, k=alphabet)
        symbolic_trace, symbol_map, mapping = map_bins_to_symbols_multi(traces, k=alphabet)

        # Scale to int after building the map
        temp_symbol_map = build_temp_symbol_map(symbol_map, breakpoints, mean=global_mean, std=global_std)
        temp_symbol_map = {letter: int(round(temp * 100)) for letter, temp in temp_symbol_map.items()}

        discretized_path = (
            f"../Data/4-DiscretizationData/{discretization_method}/{period}/"
            f"{room}-{trace_nr}trace-{period}-{discretization_method}-w{w}-a{alphabet}-kf{k_future}-trace.txt"
        )
        format_output(symbolic_res_list=symbolic_trace, output_path=discretized_path)

        plot_discretized_traces(
            discretized_traces=traces,
            output_folder=f"../Data/Graphs/Discretized/{discretization_method}/{period}/{trace_nr}trace-w{w}-a{alphabet}",
            mapping=mapping
        )

        learner = TALearner(tss_path=discretized_path, display=False, k=k_future)

        title = f"{room}-{trace_nr}trace-{period}-{discretization_method}-w{w}-a{alphabet}-kf{k_future}-ta"
        learner.ta.show(
            title=title,
            savePng=True,
            output_path=f"../Data/5-TaResults/{discretization_method}/{period}"
        )

        learner.ta.export_ta(
            path=f"../Data/6-XMLOutput/{discretization_method}/{period}/"
                 f"{room}-{trace_nr}trace-{period}-{discretization_method}-w{w}-a{alphabet}-kf{k_future}.xml",
            symbol_map=temp_symbol_map  # ← correct map with real °C values
        )

        print(f"Done: trace={trace_nr}, alphabet={alphabet}, k_future={k_future}")