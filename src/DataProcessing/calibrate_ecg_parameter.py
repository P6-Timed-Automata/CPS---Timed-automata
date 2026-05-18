import numpy as np
from scipy.signal import find_peaks,  peak_widths
from pathlib import Path

from DataProcessing.processData import (
    get_trace_files
)


from Discretization.discretizationSetup import (
    csv_to_temp_time_list
)

def calibrate_ecg_parameters(traces, spike_prominence=0.2):
    """
    Extract structural verification parameters for timed automata.

    Parameters:
    - spike_window  : typical spike duration
    - flat_window   : typical flat duration
    - spike_threshold : threshold for spike detection
    - flat_threshold  : threshold for flat detection

    IDEA
    ----
    We process MANY ECG traces and estimate:
        "What is the normal spike duration?"
        "What is the normal flat duration?"

    using robust statistics (median / percentiles)
    instead of manually hardcoding values.
    """

    # --------------------------------------------------
    # STORAGE FOR GLOBAL STATISTICS
    # --------------------------------------------------

    # stores all detected spike durations from all traces
    all_spike_durations = []

    # stores all detected flat durations from all traces
    all_flat_durations = []

    # stores all deviations from baseline
    # used later for estimating flat threshold
    all_noise = []

    # --------------------------------------------------
    # PROCESS EACH TRACE
    # --------------------------------------------------
    for trace in traces:

        # split trace into signal values and timestamps
        values = np.array([v for v, t in trace])
        times  = np.array([t for v, t in trace])

        # skip traces that are too short
        if len(values) < 3:
            continue

        # --------------------------------------------------
        # BASELINE ESTIMATION
        # --------------------------------------------------
        # ECG spends most time near resting value.
        #
        # Median is used instead of mean because:
        # - spikes heavily affect the mean
        # - median is robust against outliers
        baseline = np.median(values)

        # --------------------------------------------------
        # SPIKE DETECTION
        # --------------------------------------------------
        # find_peaks detects local maxima.
        #
        # "prominence" means:
        # how much a peak stands out compared
        # to surrounding signal.
        #
        # Small noise bumps are ignored.

        peaks, _ = find_peaks(values, prominence=spike_prominence)

        # --------------------------------------------------
        # SPIKE WIDTH ESTIMATION
        # --------------------------------------------------
        if len(peaks) > 0:
            # peak_widths estimates how wide each spike is.
            # rel_height=0.25 means:
            # measure width close to the top of the spike,
            # not the full noisy base.
            # This better approximates the sharp ECG QRS core.

            widths, _, _, _ = peak_widths(
                values,
                peaks,
                rel_height=0.25
            )


            # convert sample widths -> real time widths
            # dt = average time difference between samples
            dt = np.median(np.diff(times)) if len(times) > 1 else 1.0

            # spike duration in time units
            spike_durations = widths * dt

            # accumulate durations globally
            all_spike_durations.extend(spike_durations)

        # --------------------------------------------------
        # FLAT DETECTION
        # --------------------------------------------------

        # noise = distance from resting baseline
        # small values:
        #     signal is near baseline (flat)
        # large values:
        #     signal is active/spiking

        # distance from baseline
        noise = np.abs(values - baseline)

        # collect all noise values globally
        all_noise.extend(noise)

        # --------------------------------------------------
        # FLAT THRESHOLD ESTIMATION
        # --------------------------------------------------
        # 70th percentile means:
        # most baseline variations are still considered flat,
        # while larger deviations are considered activity.

        # more tolerant threshold
        flat_thresh = np.percentile(noise, 70)

        # boolean mask:
        # True  -> flat region
        # False -> active region
        is_flat = noise <= flat_thresh

        # --------------------------------------------------
        # FIND CONTINUOUS FLAT SEGMENTS
        # --------------------------------------------------
        # Convert True/False transitions into intervals.

        # find continuous flat regions
        edges = np.diff(is_flat.astype(int))

        starts = np.where(edges == 1)[0]
        ends   = np.where(edges == -1)[0]

        # handle edge case:
        # trace starts already flat
        if is_flat[0]:
            starts = np.insert(starts, 0, 0)

        # handle edge case:
        # trace ends still flat
        if is_flat[-1]:
            ends = np.append(ends, len(values)-1)

        # --------------------------------------------------
        # FLAT DURATION ESTIMATION
        # --------------------------------------------------
        for s, e in zip(starts, ends):

            duration = times[e] - times[s]

            # ignore tiny fragments
            if duration > 5:
                all_flat_durations.append(duration)

    # --------------------------------------------------
    # FINAL PARAMETERS
    # --------------------------------------------------

    # median spike duration across all traces
    spike_window = (
        float(np.median(all_spike_durations))
        if all_spike_durations else 1.0
    )

    # median flat duration across all traces
    flat_window = (
        float(np.median(all_flat_durations))
        if all_flat_durations else 1.0
    )

    spike_threshold = float(spike_prominence)

    # global flat threshold estimated from all traces
    flat_threshold = (
        float(np.percentile(all_noise, 70))
        if len(all_noise) > 0 else 0.1
    )

    # --------------------------------------------------
    # RETURN VERIFICATION PARAMETERS
    # --------------------------------------------------
    return {
        "spike_window": spike_window,
        "flat_window": flat_window,
        "spike_threshold": spike_threshold,
        "flat_threshold": flat_threshold
    }

BASE_DIR = Path(__file__).resolve().parent.parent.parent

ecg_folder = BASE_DIR / "Data" / "3-ExtractInterval" /"ecg" /"1beat-experiment"/"1beat-train"

ecg_raw_traces = get_trace_files(folder_path = ecg_folder)
ecg_traces = csv_to_temp_time_list(input_files=ecg_raw_traces)


params = calibrate_ecg_parameters(ecg_traces)
print(params)