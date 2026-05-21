"""
shared/pipeline.py
==================
Core pipeline: discretize → format (collapsed) → TAG → evaluate.

Used identically by all experiments so results are directly comparable.

Key design decisions
--------------------
* Consecutive same-symbol events are COLLAPSED before TAG so guards
  represent thermal dwell times, not sampling intervals.
* SAX uses GLOBAL normalisation fitted on training data only, applied
  consistently to both train and test.
* Test traces are discretised using TRAINING bins/breakpoints so there
  is no data leakage.
"""

import os
import string
import tempfile

import numpy as np
from scipy.stats import norm as scipy_norm

from Discretization.naive import equal_width_discretization
from Discretization.sax import sax_discretization_multi
from Discretization.persist import (
    Persist, get_best_bins, discretize_traces_with_bins, flatten_traces_to_ts,
)
from Discretization.discretizationSetup import map_bins_to_symbols
from TAG.TALearner import TALearner


# ---------------------------------------------------------------------------
# Data format conversion
# ---------------------------------------------------------------------------

def to_list_format(traces):
    """
    Convert [(times_arr, temps_arr), ...] → [[(temp, time), ...], ...]
    matching the format expected by discretization functions.
    """
    return [
        [(float(v), int(t)) for t, v in zip(times, temps)]
        for times, temps in traces
    ]


# ---------------------------------------------------------------------------
# Collapsed format helpers
# ---------------------------------------------------------------------------

def _collapse_symbolic(trace_symbols):
    """
    Collapse consecutive same-symbol (symbol, time) pairs, accumulating delays.
    Input : [(symbol, time), ...]  where time is absolute timestamp
    Output: ["a:600", "b:300", ...]  collapsed timed strings
    """
    if not trace_symbols:
        return []

    collapsed = []
    cur_sym, prev_time = trace_symbols[0]
    accumulated = 0

    for i, (sym, t) in enumerate(trace_symbols):
        if i == 0:
            continue
        delay = max(0, int(float(t) - float(prev_time)))
        prev_time = t
        if sym == cur_sym:
            accumulated += delay
        else:
            collapsed.append(f"{cur_sym}:{accumulated}")
            cur_sym, accumulated = sym, delay

    collapsed.append(f"{cur_sym}:{accumulated}")
    return collapsed


def _write_collapsed(symbolic_traces, output_path):
    """Write a list of (symbol, time) traces to file in collapsed TAG format."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    lines = []
    for trace in symbolic_traces:
        collapsed = _collapse_symbolic(trace)
        if collapsed:
            lines.append(" ".join(collapsed))
    with open(output_path, "w") as f:
        f.write("\n".join(lines))

def _preprocess_test(test_data_list, bins, n_symbols, mark_out_of_range=True):
    """
    Discretise test traces with training bins. Values outside [bins[0], bins[-1]]
    become OUT_OF_RANGE_SYMBOL='?' instead of being clipped, so any OOD sample
    causes alphabet mismatch in the learned TA.
    """
    OUT_OF_RANGE_SYMBOL = '?'
    alphabet = list(string.ascii_lowercase)[:n_symbols]
    mapping  = {i: alphabet[i] for i in range(n_symbols)}
    k        = len(bins) - 1

    result = []
    for trace in test_data_list:
        values = np.array([v for v, _ in trace])
        times  = np.array([t for _, t in trace], dtype=float)

        # digitize: -1 for v<bins[0], k for v>=bins[-1], i for in-range bin i.
        labels = np.digitize(values, bins) - 1
        # The top edge belongs to the top bin, not to "out of range above".
        labels = np.where(values == bins[-1], k - 1, labels)

        if mark_out_of_range:
            in_range = (labels >= 0) & (labels < k)
            sym_time = [
                (mapping[int(l)] if ok else OUT_OF_RANGE_SYMBOL, int(times[i]))
                for i, (l, ok) in enumerate(zip(labels, in_range))
            ]
        else:
            labels   = np.clip(labels, 0, k - 1)
            sym_time = [(mapping[int(l)], int(times[i])) for i, l in enumerate(labels)]

        result.append(_collapse_symbolic(sym_time))

    return result
# def _preprocess_test(test_data_list, bins, n_symbols):
#     """
#     Discretise test traces using training bins and return collapsed timed strings.
#
#     Parameters
#     ----------
#     test_data_list : [[(temp, time), ...], ...]
#     bins           : bin edges (length n_symbols + 1)
#     n_symbols      : number of symbols
#
#     Returns
#     -------
#     list of timed-string lists, e.g. [["a:600", "b:300"], ...]
#     """
#     alphabet = list(string.ascii_lowercase)[:n_symbols]
#     mapping  = {i: alphabet[i] for i in range(n_symbols)}
#     k        = len(bins) - 1
#
#     result = []
#     for trace in test_data_list:
#         values = np.array([v for v, _ in trace])
#         times  = np.array([t for _, t in trace], dtype=float)
#
#         labels = np.digitize(values, bins) - 1
#         labels = np.clip(labels, 0, k - 1)
#
#         # Build (symbol, time) pairs then collapse
#         sym_time = [(mapping[l], int(times[i])) for i, l in enumerate(labels)]
#         result.append(_collapse_symbolic(sym_time))
#
#     return result


def _preprocess_test_sax(test_data_list, w, n_symbols, breakpoints,
                         global_mean, global_std,
                         value_range=None, mark_out_of_range=True):
    """
    SAX test preprocessing. If value_range=(min_train, max_train) is given and
    mark_out_of_range=True, PAA segments whose raw mean falls outside the
    training value range are emitted as '?' rather than digitized.
    """
    OUT_OF_RANGE_SYMBOL = '?'
    alphabet = list(string.ascii_lowercase)[:n_symbols]
    mapping  = {i: alphabet[i] for i in range(n_symbols)}

    result = []
    for trace in test_data_list:
        v = np.array([val for val, _ in trace], dtype=float)
        t = np.array([tim for _, tim in trace], dtype=float)

        v_norm = (v - global_mean) / global_std if global_std != 0 else np.zeros_like(v)

        # PAA in both raw and z-space — raw is needed for OOD detection,
        # z-space for digitization against Gaussian breakpoints.
        v_segs_raw  = np.array_split(v,      w)
        v_segs_norm = np.array_split(v_norm, w)
        t_segs      = np.array_split(t,      w)

        paa_v_raw  = np.array([seg.mean() for seg in v_segs_raw])
        paa_v_norm = np.array([seg.mean() for seg in v_segs_norm])
        paa_t      = np.array([int(seg.mean()) for seg in t_segs])

        labels = np.digitize(paa_v_norm, breakpoints, right=False)
        labels = np.clip(labels, 0, n_symbols - 1)

        sym_time = []
        for i, l in enumerate(labels):
            oor = (mark_out_of_range and value_range is not None
                   and (paa_v_raw[i] < value_range[0]
                        or paa_v_raw[i] > value_range[1]))
            sym = OUT_OF_RANGE_SYMBOL if oor else mapping[int(l)]
            sym_time.append((sym, int(paa_t[i])))

        result.append(_collapse_symbolic(sym_time))

    return result


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run_pipeline(
        method:           str,
        params:           dict,
        train_traces:     list,     # [(times_arr, temps_arr), ...]
        test_pos_traces:  list,
        test_neg_traces:  list,
        tag_k:            int   = 2,
        tmp_path:         str   = None,
        neg_modes:        list  = None,   # mode index per neg trace
        save_ta_path:     str   = None,   # folder for TA PNG
        ta_title:         str   = None,
):
    """
    Full pipeline for one method configuration.

    Returns
    -------
    dict with keys:
        method, params, n_states, n_edges, learner,
        overall   : {TP, FP, TN, FN, precision, recall, f1, PAR, NAR}
        per_mode  : {mode_name: rejection_rate} if neg_modes provided
    """
    if tmp_path is None:
        # Secure a unique path to isolate concurrent SLURM executions on the same node
        fd, tmp_path = tempfile.mkstemp(suffix=".txt", prefix="pipeline_tmp_")
        os.close(fd)  # Close the descriptor so downstream components can open the file path

    # Convert to list format
    train_list   = to_list_format(train_traces)
    pos_list     = to_list_format(test_pos_traces)
    neg_list     = to_list_format(test_neg_traces)

    # ------------------------------------------------------------------
    # Discretise training data
    # ------------------------------------------------------------------
    if method == "naive":
        traces_disc, bins = equal_width_discretization(train_list, k=params["bins"])
        n_symbols         = len(bins) - 1
        sym_train, _, _   = map_bins_to_symbols(traces_disc, bins)
        _write_collapsed(sym_train, tmp_path)

        pos_strings = _preprocess_test(pos_list, bins, n_symbols)
        neg_strings = _preprocess_test(neg_list, bins, n_symbols)

    elif method.startswith("sax"):
        w            = params["w"]
        k            = params["bins"]
        breakpoints  = scipy_norm.ppf(np.linspace(0, 1, k + 1)[1:-1])

        # Global stats from training
        all_v        = np.concatenate([np.array([v for v, _ in tr]) for tr in train_list])
        global_mean  = float(all_v.mean())
        global_std   = float(all_v.std()) if all_v.std() != 0 else 1.0
        value_range = (float(all_v.min()), float(all_v.max()))

        traces_disc, bins_z, _, _ = sax_discretization_multi(
            train_list, w=w, k=k
        )
        bins      = np.sort(bins_z) * global_std + global_mean
        n_symbols = k
        sym_train, _, _ = map_bins_to_symbols(traces_disc,
                                              np.sort(bins_z))
        _write_collapsed(sym_train, tmp_path)

        pos_strings = _preprocess_test_sax(pos_list, w, n_symbols,
                                           breakpoints, global_mean, global_std,
                                           value_range=value_range)
        neg_strings = _preprocess_test_sax(neg_list, w, n_symbols,
                                           breakpoints, global_mean, global_std,
                                           value_range=value_range)
        # pos_strings = _preprocess_test_sax(pos_list, w, n_symbols,
        #                                    breakpoints, global_mean, global_std)
        # neg_strings = _preprocess_test_sax(neg_list, w, n_symbols,
        #                                    breakpoints, global_mean, global_std)

    elif method == "persist":
        ts           = flatten_traces_to_ts(train_list)
        persist_obj  = Persist(ts, break_min=2, break_max=params["bins"],
                               skip=np.array([4, 4]))
        bins         = get_best_bins(persist_obj, ts)
        n_symbols    = len(bins) - 1

        traces_disc  = discretize_traces_with_bins(train_list, bins)
        sym_train, _, _ = map_bins_to_symbols(traces_disc, bins)
        _write_collapsed(sym_train, tmp_path)

        pos_strings  = _preprocess_test(pos_list, bins, n_symbols)
        neg_strings  = _preprocess_test(neg_list, bins, n_symbols)

    else:
        raise ValueError(f"Unknown method: {method}")

    # ------------------------------------------------------------------
    # Learn TA
    # ------------------------------------------------------------------
    learner  = TALearner(tss_path=tmp_path, display=False, k=tag_k)
    n_states = len(learner.ta.states)
    n_edges  = len(learner.ta.edges)

    if save_ta_path and ta_title:
        learner.ta.show(title=ta_title, savePng=True, output_path=save_ta_path)

    # ------------------------------------------------------------------
    # Evaluate overall
    # ------------------------------------------------------------------
    overall = learner.ta.evaluate_classifier(
        positive_tss=pos_strings,
        negative_tss=neg_strings,
        timed=True,
    )

    # ------------------------------------------------------------------
    # Per-mode rejection rates
    # ------------------------------------------------------------------
    per_mode = {}
    if neg_modes is not None:
        from collections import defaultdict
        from Generators import NEG_MODE_NAMES

        mode_indices = defaultdict(list)
        for i, m in enumerate(neg_modes):
            mode_indices[m].append(i)

        for mode_idx, indices in mode_indices.items():
            mode_neg = [neg_strings[i] for i in indices]
            if not mode_neg:
                continue
            # Count how many are correctly rejected
            n_rejected = sum(
                1 for ts in mode_neg
                if not any(
                    learner.ta._Automaton__exist_path(ts, timed=True)
                    for _ in [None]
                )
            )
            # Use evaluate_classifier for clean per-mode metrics
            mode_metrics = learner.ta.evaluate_classifier(
                positive_tss=pos_strings,
                negative_tss=mode_neg,
                timed=True,
            )
            name = NEG_MODE_NAMES.get(mode_idx, str(mode_idx))
            per_mode[name] = {
                "NAR":        mode_metrics["NAR"],    # negative acceptance rate (lower = better)
                "rejection":  100.0 - mode_metrics["NAR"],  # correctly rejected %
                "precision":  mode_metrics["precision"],
                "recall":     mode_metrics["recall"],
                "f1":         mode_metrics["f1"],
            }

    # Clean up temp file
    if os.path.exists(tmp_path):
        os.remove(tmp_path)

    return {
        "method":   method,
        "params":   params,
        "n_states": n_states,
        "n_edges":  n_edges,
        "learner":  learner,
        "overall":  overall,
        "per_mode": per_mode,
    }