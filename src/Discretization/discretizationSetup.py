"""
Discretization setup utilities.

Trace format conventions:
- Raw trace: [(value, time_seconds), ...] where time is absolute, integer seconds.
- Discretized trace: [(bin_label_int, time_seconds), ...]
- Symbolic trace: [(letter, time_seconds), ...]
- TAG trace string: "a:0 b:300 c:600 ..." where each number is dwell time
  (seconds the symbol persisted before the next different symbol occurred).
"""

import os
import string

import numpy as np


# -----------------------------------------------------------------------------
# CSV loading
# -----------------------------------------------------------------------------

def csv_to_temp_time_list(input_files, time_dtype=int):
    """
    Load CSVs into [(value, time), ...] traces.

    Parameters
    ----------
    input_files : iterable of paths
    time_dtype  : int (default) for temperature data with integer seconds,
                  float for ECG or sub-second sampling.

    Returns
    -------
    list of traces; each trace is a list of (value: float, time: time_dtype).

    Load CSVs into [(value, time), ...] traces.

    Note: the CSV column order is `time;value`, but this function returns
    (value, time) tuples to match the (value, time) convention used by
    downstream discretization and TAG code. The swap is intentional.
    """
    all_results = []
    for input_file in input_files:
        data = np.genfromtxt(input_file, delimiter=';', dtype=str, skip_header=1)
        # genfromtxt returns 1D for single-row files; force 2D so indexing works.
        data = np.atleast_2d(data)
        # Skip rows where the first column is not numeric (like header)
        numeric_data = []
        for row in data:
            try:
                float(row[0])  # check if first column is a number
                numeric_data.append(row)
            except ValueError:
                continue

        numeric_data = np.array(numeric_data)
        times = numeric_data[:, 0].astype(float).astype(time_dtype)
        values = numeric_data[:, 1].astype(float)

        # times = data[:, 0].astype(float).astype(time_dtype)
        # values = data[:, 1].astype(float)
        all_results.append(list(zip(values.tolist(), times.tolist())))

    return all_results





# -----------------------------------------------------------------------------
# Symbol alphabet and mapping
# -----------------------------------------------------------------------------

_ALPHABET = list(string.ascii_lowercase)


def _alphabet(n_symbols):
    if n_symbols > len(_ALPHABET):
        raise ValueError(
            f"n_symbols={n_symbols} exceeds available alphabet of "
            f"{len(_ALPHABET)} letters."
        )
    return _ALPHABET[:n_symbols]


def map_bins_to_symbols(discretized_traces, bins, value_scale=100):
    """
    Convert integer-labeled traces to letter-labeled traces and produce
    a symbol -> scaled bin midpoint mapping for UPPAAL.

    Parameters
    ----------
    discretized_traces : list of [(label_int, time), ...]
    bins               : array of bin edges, length n_symbols + 1
    value_scale        : multiplier for the bin midpoints in symbol_map.
                         Default 100 converts e.g. 22.41 °C to 2241 for
                         use as an integer in UPPAAL declarations.

    Returns
    -------
    symbolic_traces : list of [(letter, time), ...]
    symbol_map      : dict letter -> scaled midpoint int
    label_to_letter : dict int label -> letter
    """
    n_symbols = len(bins) - 1
    letters = _alphabet(n_symbols)
    label_to_letter = {i: letters[i] for i in range(n_symbols)}

    symbol_map = {
        letters[i]: round(((bins[i] + bins[i + 1]) / 2) * value_scale)
        for i in range(n_symbols)
    }

    symbolic_traces = [
        [(label_to_letter[int(label)], int(time)) for label, time in trace]
        for trace in discretized_traces
    ]

    return symbolic_traces, symbol_map, label_to_letter


# -----------------------------------------------------------------------------
# Test-time discretization (uses TRAINING bins, no leakage)
# -----------------------------------------------------------------------------

# Sentinel label for out-of-training-range values. Maps to a symbol that
# no training trace contains, so any test trace touching it is guaranteed
# to fail with an alphabet inconsistency (Cornanguer thesis §4.3.2).
OUT_OF_RANGE_LABEL = -1
OUT_OF_RANGE_SYMBOL = '?'


def preprocess_test_traces(test_traces, bins, mark_out_of_range=True):
    """
    Discretize test traces using training bins.

    Parameters
    ----------
    test_traces        : list of [(value, time), ...]
    bins               : training bin edges
    mark_out_of_range  : if True, values outside [bins[0], bins[-1]] become
                         OUT_OF_RANGE_SYMBOL; if False, they get clipped into
                         the nearest training bin (Cornanguer's clip behavior).
                         For anomaly detection, keep True.

    Returns
    -------
    list of TAG-format strings, e.g. [["a:600" "b:300 "c:120"],[] ...]
    """
    n_symbols = len(bins) - 1
    letters = _alphabet(n_symbols)

    formatted_traces = []
    for trace in test_traces:
        values = np.array([v for v, _ in trace])
        times = np.array([t for _, t in trace])

        # np.digitize returns 0 for v < bins[0], i for bins[i-1] <= v < bins[i],
        # n_symbols + 1 for v >= bins[-1]. Subtract 1 to get bin index in [0, n_symbols-1]
        # for in-range values; out-of-range become -1 or n_symbols.
        labels = np.digitize(values, bins) - 1

        # In-range: clamp v == bins[-1] (which gives label n_symbols) down by 1
        # since the top bin should include its right edge.
        labels = np.where(values == bins[-1], n_symbols - 1, labels)

        if mark_out_of_range:
            # Anything still outside [0, n_symbols-1] becomes the sentinel.
            in_range = (labels >= 0) & (labels < n_symbols)
            symbols = np.where(
                in_range,
                [letters[lbl] if 0 <= lbl < n_symbols else OUT_OF_RANGE_SYMBOL
                 for lbl in labels],
                OUT_OF_RANGE_SYMBOL,
            )
        else:
            labels = np.clip(labels, 0, n_symbols - 1)
            symbols = [letters[lbl] for lbl in labels]

        # Convert to TAG format with collapsing + relative delays
        symbol_time_pairs = list(zip(symbols, times.tolist()))


        formatted = _format_trace(symbol_time_pairs)

        # keep output as list instead of single string
        formatted_traces.append(formatted.split())

    return formatted_traces



def sax_preprocess_traces(test_traces, w, breakpoints, mean, std, mark_out_of_range=True):
    n_symbols = len(breakpoints) + 1
    letters = _alphabet(n_symbols)

    formatted_traces = []

    for trace in test_traces:
        if len(trace) == 0:
            formatted_traces.append([])
            continue

        values = np.array([v for v, _ in trace], dtype=float)
        times = np.array([t for _, t in trace], dtype=float)

        # ---- SAME NORMALIZATION AS TRAIN ----
        norm_v = (values - mean) / std
        norm_v = np.nan_to_num(norm_v)

        # ---- SAME PAA AS TRAIN ----
        n = len(norm_v)
        w_eff = min(w, n)

        v_segs = np.array_split(norm_v, w_eff)
        t_segs = np.array_split(times, w_eff)

        paa_v = []
        paa_t = []

        for vs, ts in zip(v_segs, t_segs):
            if len(vs) == 0:
                continue
            paa_v.append(np.nanmean(vs))
            paa_t.append(int(np.nanmean(ts)))

        paa_v = np.array(paa_v)
        paa_t = np.array(paa_t)

        if len(paa_v) == 0:
            formatted_traces.append([])
            continue

        # ---- SAX DISCRETIZATION (CRITICAL FIX) ----
        labels = np.digitize(paa_v, breakpoints, right=False) # SAME AS TRAIN

        # ---- OUT OF RANGE HANDLING (MATCH YOUR PIPELINE) ----
        if mark_out_of_range:
            symbols = []
            for lbl in labels:
                if 0 <= lbl < n_symbols:
                    symbols.append(letters[lbl])
                else:
                    symbols.append(OUT_OF_RANGE_SYMBOL)
        else:
            labels = np.clip(labels, 0, n_symbols - 1)
            symbols = [letters[lbl] for lbl in labels]

        # ---- FORMAT EXACTLY LIKE YOUR EXISTING PIPELINE ----
        symbol_time_pairs = list(zip(symbols, paa_t.tolist()))
        formatted = _format_trace(symbol_time_pairs)

        formatted_traces.append(formatted.split())


    return formatted_traces


# -----------------------------------------------------------------------------
# TAG trace formatting
# -----------------------------------------------------------------------------

def _format_trace(symbol_time_pairs):
    """
    Convert [(symbol, absolute_time), ...] into a TAG trace string
    "a:dwell_time b:dwell_time c:dwell_time" where dwell_time is the number
    of seconds the symbol persisted before the next different symbol.

    The last symbol's dwell time is the gap to the final timestamp.
    """
    if not symbol_time_pairs:
        return ""


    parts = []
    run_start_idx = 0
    run_symbol = symbol_time_pairs[0][0]

    for i in range(1, len(symbol_time_pairs)):
        sym, t = symbol_time_pairs[i]
        if sym != run_symbol:
            # End of the previous run. Dwell time = how long we stayed in run_symbol.
            run_start_time = symbol_time_pairs[run_start_idx][1]
            dwell = max(0, int(t - run_start_time))
            parts.append(f"{run_symbol}:{dwell}")
            run_symbol = sym
            run_start_idx = i

    # Last run: its dwell time is from its start to the final timestamp.
    last_time = symbol_time_pairs[-1][1]
    run_start_time = symbol_time_pairs[run_start_idx][1]
    final_dwell = max(0, int(last_time - run_start_time))
    parts.append(f"{run_symbol}:{final_dwell}")

    return " ".join(parts)


def format_output(symbolic_traces, output_path):
    """
    Write a list of [(symbol, absolute_time), ...] traces to disk in
    TAG format, one trace per line.
    """
    lines = [_format_trace(trace) for trace in symbolic_traces]

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        f.write("\n".join(lines))
