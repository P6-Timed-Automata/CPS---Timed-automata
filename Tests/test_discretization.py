"""
Tests for Discretization.discretizationSetup.

Each test class targets a specific bug from the cleanup round or a specific
property that must hold for downstream TAG learning to work.
"""

import os
import tempfile

import numpy as np
import pytest

from Discretization.discretizationSetup import (
    _alphabet,
    _format_trace,
    csv_to_temp_time_list,
    format_output,
    map_bins_to_symbols,
    OUT_OF_RANGE_SYMBOL,
    preprocess_test_traces,
)

from Discretization.persist import (
    Persist,
    get_best_bins,
    discretize_traces_with_bins,
    flatten_traces_to_ts,
)


# =============================================================================
# Trace formatting
# =============================================================================

class TestTraceFormatting:
    """
    A TAG trace string `a:dwell_a b:dwell_b c:dwell_c` must encode:
      - dwell_a = time from start of run-of-a to start of run-of-b
      - dwell_b = time from start of run-of-b to start of run-of-c
      - dwell_last = time from start of last run to final timestamp
    """

    def test_single_symbol(self):
        # One sample at t=0: zero dwell, just the symbol.
        trace = [('a', 0)]
        assert _format_trace(trace) == "a:0"

    def test_single_run_of_same_symbol(self):
        # a at 0, a at 300, a at 600 — single run of 'a' lasting 600 s.
        trace = [('a', 0), ('a', 300), ('a', 600)]
        assert _format_trace(trace) == "a:600"

    def test_two_different_symbols_back_to_back(self):
        # a at 0, then c at 300 — 'a' lasted 300 s before c.
        # c has no successor so its dwell is to the final timestamp = 0.
        trace = [('a', 0), ('c', 300)]
        assert _format_trace(trace) == "a:300 c:0"

    def test_dwell_time_is_until_next_different_symbol(self):
        """
        The critical case the old buggy code got wrong.
        a at [0, 300, 600], then c at 900.
        - 'a' run lasted from t=0 until c started at t=900 → dwell 900
        - 'c' has no successor, last timestamp is 900 → dwell 0
        """
        trace = [('a', 0), ('a', 300), ('a', 600), ('c', 900)]
        assert _format_trace(trace) == "a:900 c:0"

    def test_three_runs(self):
        # a:0..300, b:300..600, c:600..900
        trace = [('a', 0), ('b', 300), ('c', 600), ('c', 900)]
        assert _format_trace(trace) == "a:300 b:300 c:300"

    def test_empty_trace(self):
        assert _format_trace([]) == ""

    def test_dwell_never_negative(self):
        """timestamps shouldn't produce negative dwell times."""
        trace = [('a', 100), ('b', 50)]   # time went backwards
        result = _format_trace(trace)
        assert "-" not in result, f"Negative dwell in: {result}"

    def test_matches_cornanguer_tv_log_convention(self):
        """
        Cornanguer's TV-log trace format is:
          News:0 COM:3570 Interstitials:30 Children_programs:7 ...
        This means News at t=0 was immediately followed by COM at t=0
        (zero dwell), which then lasted 3570 s before Interstitials.
        """
        trace = [('News', 0), ('COM', 0), ('Interstitials', 3570),
                 ('Children_programs', 3600)]
        result = _format_trace(trace)
        # News dwelt 0 s, COM dwelt 3570 s, Interstitials dwelt 30 s,
        # Children_programs is last so dwell to itself = 0.
        assert result == "News:0 COM:3570 Interstitials:30 Children_programs:0"


# =============================================================================
# Alphabet handling
# =============================================================================

class TestAlphabet:

    def test_alphabet_size_within_limit(self):
        assert _alphabet(3) == ['a', 'b', 'c']
        assert _alphabet(26) == list("abcdefghijklmnopqrstuvwxyz")

    def test_alphabet_size_zero(self):
        assert _alphabet(0) == []

    def test_alphabet_size_too_large_raises(self):
        with pytest.raises(ValueError, match="exceeds available alphabet"):
            _alphabet(27)


# =============================================================================
# Bin-to-symbol mapping
# =============================================================================

class TestMapBinsToSymbols:

    def test_basic_mapping(self):
        traces = [[(0, 0), (1, 300), (2, 600)]]
        bins = np.array([0.0, 1.0, 2.0, 3.0])   # 3 bins
        symbolic, symbol_map, label_to_letter = map_bins_to_symbols(traces, bins)

        assert symbolic == [[('a', 0), ('b', 300), ('c', 600)]]
        assert label_to_letter == {0: 'a', 1: 'b', 2: 'c'}

    def test_symbol_map_uses_midpoints_scaled(self):
        traces = [[(0, 0)]]
        bins = np.array([10.0, 20.0, 30.0])    # bin 0: [10, 20), bin 1: [20, 30]
        _, symbol_map, _ = map_bins_to_symbols(traces, bins, value_scale=100)

        # Bin 0 midpoint = 15.0 → 1500. Bin 1 midpoint = 25.0 → 2500.
        assert symbol_map == {'a': 1500, 'b': 2500}

    def test_symbol_map_value_scale_default_is_100(self):
        traces = [[(0, 0)]]
        bins = np.array([22.0, 24.0])
        _, symbol_map, _ = map_bins_to_symbols(traces, bins)
        assert symbol_map == {'a': 2300}   # midpoint 23.0 × 100

    def test_symbol_map_value_scale_one_preserves_units(self):
        traces = [[(0, 0)]]
        bins = np.array([22.0, 24.0])
        _, symbol_map, _ = map_bins_to_symbols(traces, bins, value_scale=1)
        assert symbol_map == {'a': 23}

    def test_no_redundant_size_parameter(self):
        """Alphabet size derived from bins, not passed separately."""
        bins = np.array([0.0, 1.0, 2.0])
        _, _, label_to_letter = map_bins_to_symbols([[(0, 0)]], bins)
        assert len(label_to_letter) == 2   # len(bins) - 1


# =============================================================================
# Test-trace discretization with training bins
# =============================================================================

class TestPreprocessTestTraces:

    def test_in_range_values_get_correct_symbols(self):
        bins = np.array([0.0, 10.0, 20.0, 30.0])   # bins a, b, c
        traces = [[(5.0, 0), (15.0, 100), (25.0, 200)]]
        result = preprocess_test_traces(traces, bins)
        # In TAG format: a dwelt 100, b dwelt 100, c is last → 0
        assert result == ["a:100 b:100 c:0"]

    def test_top_edge_value_included_in_top_bin(self):
        """
        Value exactly equal to bins[-1] should belong to the top bin,
        not be marked out-of-range.
        """
        bins = np.array([0.0, 10.0, 20.0])
        traces = [[(20.0, 0)]]  # exactly bins[-1]
        result = preprocess_test_traces(traces, bins)
        assert result == ["b:0"]   # top bin, not '?'

    def test_value_below_training_min_marked_sentinel(self):
        """
        Critical for offset anomaly detection. An anomaly trace with values
        below training range must NOT be silently clipped into the bottom bin.
        """
        bins = np.array([20.0, 22.0, 24.0])   # training range is [20, 24]
        traces = [[(15.0, 0), (16.0, 100)]]   # both below training min
        result = preprocess_test_traces(traces, bins)
        assert OUT_OF_RANGE_SYMBOL in result[0], \
            f"Out-of-range value silently mapped; got {result[0]}"

    def test_value_above_training_max_marked_sentinel(self):
        """The +15 °C offset anomaly case."""
        bins = np.array([20.0, 22.0, 24.0])   # training range is [20, 24]
        traces = [[(22.0, 0), (40.0, 100)]]   # 40 is way above training
        result = preprocess_test_traces(traces, bins)
        assert OUT_OF_RANGE_SYMBOL in result[0], \
            f"Out-of-range value silently mapped; got {result[0]}"

    def test_legacy_mode_clips_out_of_range(self):
        """
        With mark_out_of_range=False, out-of-range values get clipped into
        the nearest bin (Cornanguer-original behavior).
        """
        bins = np.array([20.0, 22.0, 24.0])
        traces = [[(15.0, 0)]]   # below training min
        result = preprocess_test_traces(traces, bins, mark_out_of_range=False)
        # Clipped to bin 0 → 'a'
        assert result == ["a:0"]
        assert OUT_OF_RANGE_SYMBOL not in result[0]

    def test_no_data_leakage_from_test_to_bins(self):
        """
        Bin edges must NOT change based on test data — passing different
        test traces with same bins should give the same in-range labels.
        """
        bins = np.array([0.0, 10.0, 20.0])
        result_a = preprocess_test_traces([[(5.0, 0)]], bins)
        result_b = preprocess_test_traces([[(5.0, 0), (1000.0, 100)]], bins)
        # First sample should be classified the same way regardless of
        # what else is in the trace.
        assert result_a[0].startswith("a:")
        assert result_b[0].startswith("a:")


# =============================================================================
# CSV loading
# =============================================================================

class TestCsvLoading:

    def _write_csv(self, path, rows, header="time;value"):
        with open(path, "w") as f:
            f.write(header + "\n")
            for row in rows:
                f.write(";".join(str(x) for x in row) + "\n")

    def test_integer_times_preserved(self, tmp_path):
        csv = tmp_path / "trace.csv"
        self._write_csv(csv, [(0, 22.5), (300, 22.7), (600, 23.1)])
        result = csv_to_temp_time_list([str(csv)])
        assert result == [[(22.5, 0), (22.7, 300), (23.1, 600)]]

    def test_subsecond_times_with_float_dtype(self, tmp_path):
        """
        ECG case: sub-second timestamps. With default int dtype these would
        truncate to all-same values, which destroys the timing information.
        """
        csv = tmp_path / "ecg.csv"
        self._write_csv(csv, [(0.004, 0.1), (0.008, 0.2), (0.012, 0.15)])

        # Default int truncates — this is the documented behavior, but
        # should be opt-in for sub-second data.
        result_int = csv_to_temp_time_list([str(csv)], time_dtype=int)
        all_times = [t for trace in result_int for _, t in trace]
        # All sub-second times collapsed to 0 — this is what we DON'T want
        # for ECG. The test documents this gotcha.
        assert all(t == 0 for t in all_times)

        # Float preserves them.
        result_float = csv_to_temp_time_list([str(csv)], time_dtype=float)
        times = [t for _, t in result_float[0]]
        assert times == [0.004, 0.008, 0.012]

    def test_value_order_in_returned_tuples(self, tmp_path):
        """The returned tuples are (value, time), not (time, value)."""
        csv = tmp_path / "trace.csv"
        self._write_csv(csv, [(100, 99.9)])
        (val, t), = csv_to_temp_time_list([str(csv)])[0]
        assert val == 99.9
        assert t == 100


# =============================================================================
# format_output
# =============================================================================

class TestFormatOutput:

    def test_writes_one_line_per_trace(self, tmp_path):
        out = tmp_path / "out.txt"
        traces = [
            [('a', 0), ('b', 300)],
            [('c', 0), ('d', 600)],
        ]
        format_output(traces, str(out))
        content = out.read_text()
        assert content == "a:300 b:0\nc:600 d:0"

    def test_creates_output_directory(self, tmp_path):
        nested = tmp_path / "deeply" / "nested" / "out.txt"
        format_output([[('a', 0)]], str(nested))
        assert nested.exists()

    def test_uses_same_dwell_logic_as_format_trace(self, tmp_path):
        """
        format_output must use _format_trace internally, not a separate
        implementation (Issue 1 — eliminating the duplicate definition).
        """
        out = tmp_path / "out.txt"
        trace = [('a', 0), ('a', 300), ('a', 600), ('c', 900)]
        format_output([trace], str(out))
        # Same expectation as TestTraceFormatting.test_dwell_time_is_until_next_different_symbol
        assert out.read_text() == "a:900 c:0"


import numpy as np
import pytest

from Discretization.persist import (
    Persist,
    get_best_bins,
    discretize_traces_with_bins,
    flatten_traces_to_ts,
)


class TestPersistBins:
    """Tests for Persist's get_best_bins output."""

    def test_bins_are_strictly_monotonic(self):
        """
        get_best_bins must produce strictly increasing bin edges, otherwise
        np.digitize downstream produces nonsense.
        """
        # Build a small but realistic training set: 5 traces of 50 samples each,
        # values ranging from 20 to 25 with some structure.
        rng = np.random.default_rng(0)
        traces = [
            [(20.0 + 5.0 * rng.random(), i * 100) for i in range(50)]
            for _ in range(5)
        ]
        ts = flatten_traces_to_ts(traces)

        persist_obj = Persist(
            x=ts, break_min=3, break_max=3,
            divergence="w", candidates="EW",
        )
        bins = get_best_bins(persist_obj, ts)

        assert np.all(np.diff(bins) > 0), \
            f"Bins not strictly monotonic: {bins}"

    def test_bins_span_data_range(self):
        """First bin edge should be at or below data min, last at or above max."""
        rng = np.random.default_rng(0)
        traces = [
            [(20.0 + 5.0 * rng.random(), i * 100) for i in range(50)]
            for _ in range(5)
        ]
        ts = flatten_traces_to_ts(traces)

        persist_obj = Persist(
            x=ts, break_min=3, break_max=3,
            divergence="w", candidates="EW",
        )
        bins = get_best_bins(persist_obj, ts)

        assert bins[0] <= float(np.min(ts))
        assert bins[-1] >= float(np.max(ts))


class TestDiscretizeWithBins:
    """Tests for discretize_traces_with_bins."""

    def test_training_data_in_range_produces_valid_labels(self):
        """All labels should be in [0, n_symbols-1] for in-range training data."""
        traces = [[(21.0, 0), (22.0, 100), (23.0, 200)]]
        bins = np.array([20.0, 22.0, 24.0])  # 2 bins
        discretized = discretize_traces_with_bins(traces, bins)
        labels = [lbl for trace in discretized for lbl, _ in trace]
        assert all(0 <= lbl <= 1 for lbl in labels), \
            f"Labels out of [0, 1]: {labels}"

    def test_clip_masks_out_of_range_values(self):
        """
        Document the current clip-behavior: values outside [bins[0], bins[-1]]
        get silently mapped to bin 0 or bin n-1. This is the bug we'd want to
        fix if discretize_traces_with_bins is ever applied to test data.
        """
        traces = [[(15.0, 0), (50.0, 100)]]   # both outside [20, 24]
        bins = np.array([20.0, 22.0, 24.0])
        discretized = discretize_traces_with_bins(traces, bins)
        labels = [lbl for trace in discretized for lbl, _ in trace]
        # Below-min clipped to 0, above-max clipped to len(bins)-2 = 1
        assert labels == [0, 1]
        # If you ever add a mark_out_of_range parameter to fix this,
        # update this test to assert the sentinel symbol instead.



class TestDiscretizeWithBins:
    """Tests for discretize_traces_with_bins."""

    def test_training_data_in_range_produces_valid_labels(self):
        """All labels should be in [0, n_symbols-1] for in-range training data."""
        traces = [[(21.0, 0), (22.0, 100), (23.0, 200)]]
        bins = np.array([20.0, 22.0, 24.0])  # 2 bins
        discretized = discretize_traces_with_bins(traces, bins)
        labels = [lbl for trace in discretized for lbl, _ in trace]
        assert all(0 <= lbl <= 1 for lbl in labels), \
            f"Labels out of [0, 1]: {labels}"

    def test_clip_masks_out_of_range_values(self):
        """
        Document the current clip-behavior: values outside [bins[0], bins[-1]]
        get silently mapped to bin 0 or bin n-1.
        """
        traces = [[(15.0, 0), (50.0, 100)]]   # both outside [20, 24]
        bins = np.array([20.0, 22.0, 24.0])
        discretized = discretize_traces_with_bins(traces, bins)
        labels = [lbl for trace in discretized for lbl, _ in trace]
        # Below-min clipped to 0, above-max clipped to len(bins)-2 = 1
        assert labels == [0, 1]
        # If you ever add a mark_out_of_range parameter to fix this,
        # update this test to assert the sentinel symbol instead.
