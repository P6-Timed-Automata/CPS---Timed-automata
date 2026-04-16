import os
import string
import numpy as np
from scipy.stats import norm

from TAG.TALearner import TALearner
from Discretization.discretizationSetup import format_output
from Discretization.sax import csv_to_temp_time_list  # single-file version
from DataProcessing.processData import get_trace_files


def sax_discretization_multi(data_lists, w, k):
    """
    Generalized SAX for N traces.
    w: PAA segment count (controls compression, not alphabet size)
    k: alphabet size (Gaussian breakpoints)
    Returns: list of [(label, time), ...] per trace, and breakpoints array
    """
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
    """Multi-trace wrapper matching the return signature expected by main."""
    if k > 26:
        raise ValueError("k > 26 not supported with single-letter symbols")
    mapping = {i: string.ascii_lowercase[i] for i in range(k)}
    symbolic = [[(mapping[label], time) for label, time in trace] for trace in traces]
    return symbolic, mapping, mapping  # symbol_map and mapping are equivalent in SAX


# --- Config ---
room = "A"
w = 50                        # PAA segments — tune to trace length
discretization_method = "sax"
period = "30day"

experiment_folder = f"../Data/3-ExtractInterval/{period}-experiment"

for trace_nr in range(1, 11):
    raw_traces = get_trace_files(folder_path=experiment_folder, max_files=trace_nr)
    data_lists = [csv_to_temp_time_list(f) for f in raw_traces]

    # SAX: k affects discretization, so both loops must be nested
    for k in range(4, 11, 2):
        traces, breakpoints = sax_discretization_multi(data_lists, w=w, k=k)
        symbolic_trace, symbol_map, mapping = map_bins_to_symbols_multi(traces, k)

        discretized_path = (
            f"../Data/4-DiscretizationData/{discretization_method}/{period}/"
            f"{room}-{trace_nr}trace-{period}-{discretization_method}-w{w}-k{k}-trace.txt"
        )
        format_output(symbolic_res_list=symbolic_trace, output_path=discretized_path)

        learner = TALearner(tss_path=discretized_path, display=False, k=k)

        title = f"{room}-{trace_nr}trace-{period}-{discretization_method}-w{w}-k{k}-ta"
        learner.ta.show(
            title=title,
            savePng=True,
            output_path=f"../Data/5-TaResults/{discretization_method}/{period}"
        )

        learner.ta.export_ta(
            path=f"../Data/6-XMLOutput/{discretization_method}/{period}/"
                 f"{room}-{trace_nr}trace-{period}-{discretization_method}-w{w}-k{k}.xml",
            symbol_map=symbol_map
        )

        print(f"Done: trace={trace_nr}, k={k}")