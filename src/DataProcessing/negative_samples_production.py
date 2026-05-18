import random
import numpy as np
import os
import matplotlib.pyplot as plt
from pathlib import Path
import shutil

from DataProcessing.processData import (
    get_trace_files
)

from Discretization.discretizationSetup import (
    csv_to_temp_time_list,
)


# =============================================================================
# ECG NEGATIVE INJECTION
# Mirrors build_injected_negatives in Metrics_temp_run.py but adapted for ECG
# time/value scale. Four controlled anomaly modes preserve real-data baseline
# characteristics while introducing detectable, known-class anomalies.
#
# Default magnitudes assume mV-scale ECG (typical MIT-BIH). If your traces are
# in raw ADC counts or otherwise scaled, multiply magnitudes by your scale
# factor when calling build_injected_negatives_ecg.
# =============================================================================

def _inject_spike_temp(trace, magnitude, duration_samples, rng):
    """Brief amplitude excursion at a random position."""
    values = np.array([float(v) for v, _ in trace], dtype=float)
    times  = np.array([float(t) for _, t in trace], dtype=float)
    n = len(values)
    if n < duration_samples + 2:
        return list(zip(values.tolist(), times.tolist()))
    start = int(rng.integers(0, n - duration_samples))
    sign  = int(rng.choice([-1, 1]))
    values[start:start + duration_samples] += sign * magnitude
    return list(zip(values.tolist(), times.tolist()))


def _inject_shifted_temp(trace, shift_fraction):
    """Phase-shift: rotate values in time (clock-skew / wrong-time-of-day)."""
    values = np.array([float(v) for v, _ in trace], dtype=float)
    times  = np.array([float(t) for _, t in trace], dtype=float)
    n = len(values)
    shift_idx = int(n * shift_fraction)
    if shift_idx <= 0 or shift_idx >= n:
        return list(zip(values.tolist(), times.tolist()))
    rotated = np.concatenate([values[shift_idx:], values[:shift_idx]])
    return list(zip(rotated.tolist(), times.tolist()))


def _inject_stuck_temp(trace, duration_samples):
    """Flatten a segment (sensor freeze)."""
    values = np.array([float(v) for v, _ in trace], dtype=float)
    times  = np.array([float(t) for _, t in trace], dtype=float)
    n = len(values)
    if n < duration_samples + 2:
        return list(zip(values.tolist(), times.tolist()))
    # Place the flat region around 1/3 of the way in (avoids start/end edges).
    start = max(0, n // 3 - duration_samples // 2)
    end   = min(n, start + duration_samples)
    values[start:end] = values[start]
    return list(zip(values.tolist(), times.tolist()))


def _inject_offset_temp(trace, magnitude):
    """Constant additive shift (calibration drift / electrode bias)."""
    values = np.array([float(v) for v, _ in trace], dtype=float) + magnitude
    times  = np.array([float(t) for _, t in trace], dtype=float)
    return list(zip(values.tolist(), times.tolist()))


def build_injected_negatives_temp(
        positive_traces,
        out_folder,
        file_prefix,
        n_per_mode=15,
        spike_magnitude=5.0,          # 5 °C spike — clearly above natural ~2 °C diurnal swing
        spike_duration_samples=2,     # ~10 min at 5-min sampling
        phase_shift_fraction=0.25,    # 6 h shift of a 24 h trace (clearly mistimed)
        stuck_duration_samples=60,    # ~5 h flat at 5-min sampling
        offset_magnitude=3.0,         # 3 °C baseline drift
        seed=42,
):
    """
    Temperature counterpart to build_injected_negatives_ecg.
    Magnitudes tuned for 5-min-sampled real room data with ~2 °C diurnal swing.
    Writes to mode subfolders so the runner's per-mode rejection metrics work.

    Modes (indices match NEG_MODE_NAMES in Generators.py):
        0 spikes   - brief ±magnitude excursion
        1 shifted  - rotate values in time
        2 stuck    - flatten a centered segment
        3 offset   - constant additive shift
    """
    from Generators import NEG_MODE_NAMES

    if not positive_traces:
        raise ValueError("No positive traces to inject into.")

    rng = np.random.default_rng(seed)

    injection_specs = [
        (0, lambda t: _inject_spike_temp(
            t, spike_magnitude, spike_duration_samples, rng)),
        (1, lambda t: _inject_shifted_temp(t, phase_shift_fraction)),
        # (2, lambda t: _inject_stuck_temp(t, stuck_duration_samples)),  # disabled for temperature
        (3, lambda t: _inject_offset_temp(t, offset_magnitude)),
    ]

    negatives, modes = [], []
    for mode_int, inject_fn in injection_specs:
        for _ in range(n_per_mode):
            idx = int(rng.integers(0, len(positive_traces)))
            negatives.append(inject_fn(positive_traces[idx]))
            modes.append(mode_int)

    out_folder = Path(out_folder)
    out_folder.mkdir(parents=True, exist_ok=True)

    mode_counters = {m: 0 for m in NEG_MODE_NAMES}
    for trace, mode_int in zip(negatives, modes):
        mode_name = NEG_MODE_NAMES[mode_int]
        mode_subfolder = out_folder / mode_name
        mode_subfolder.mkdir(parents=True, exist_ok=True)
        mode_counters[mode_int] += 1
        filename = f"neg-{file_prefix}-{mode_name}-tid{mode_counters[mode_int]}.csv"
        with open(mode_subfolder / filename, "w") as f:
            f.write("time;value\n")
            for value, time in trace:
                f.write(f"{time:.0f};{value:.5f}\n")

    counts = {NEG_MODE_NAMES[m]: c for m, c in mode_counters.items() if c > 0}
    print(f"Generated {len(negatives)} temperature negatives -> {out_folder}")
    print(f"  Mode counts: {counts}")
    return negatives, modes

def _inject_spike_ecg(trace, magnitude, duration_samples, rng,
                      offset_min=30, offset_max=80):
    """
    Inject an additional spike NEAR the R-peak (mimics an ectopic beat / PVC).

    For R-peak-centered traces, the extra spike is placed at sample offset
    (offset_min..offset_max) from the trace center, on either side. At 360 Hz,
    30-80 samples ≈ 85-220 ms — a plausible PVC coupling interval that
    produces a visibly distinct "two-spike" trace.

    Replaces the previous baseline-region placement, which was visually too
    similar to a normal positive trace.
    """
    values = np.array([float(v) for v, _ in trace], dtype=float)
    times  = np.array([float(t) for _, t in trace], dtype=float)
    n = len(values)

    if n < duration_samples + 2:
        return list(zip(values.tolist(), times.tolist()))

    center = n // 2
    sign   = int(rng.choice([-1, 1]))             # spike before or after R
    offset = int(rng.integers(offset_min, offset_max + 1))

    start = center + sign * offset - duration_samples // 2
    start = max(0, min(start, n - duration_samples))

    # Positive-going deflection (R-peak-like). Same magnitude convention as
    # before, but now added to a non-baseline region so it stacks visibly.
    values[start:start + duration_samples] += magnitude

    return list(zip(values.tolist(), times.tolist()))


def _inject_shifted_ecg(trace, shift_fraction):
    """Rotate values in time (beat misalignment / arrhythmia-style mis-timing)."""
    values = np.array([float(v) for v, _ in trace], dtype=float)
    times  = np.array([float(t) for _, t in trace], dtype=float)
    n = len(values)
    shift_idx = int(n * shift_fraction)
    if shift_idx <= 0 or shift_idx >= n:
        return list(zip(values.tolist(), times.tolist()))
    rotated = np.concatenate([values[shift_idx:], values[:shift_idx]])
    return list(zip(rotated.tolist(), times.tolist()))


def _inject_stuck_ecg(trace, duration_samples):
    """
    Flatten a narrow centered segment over the R-peak — a "missing QRS" /
    dropped-beat event. P-wave and T-wave remain intact; only the most
    distinctive feature is replaced with baseline.

    For R-peak-centered traces, duration_samples should cover ≈ QRS+S width.
    At 360 Hz: 40-60 samples ≈ 110-170 ms is appropriate. The previous
    180-sample width (500 ms ≈ 30 % of a 1-beat trace) erased far more than
    a clinically realistic conduction failure.
    """
    values = np.array([float(v) for v, _ in trace], dtype=float)
    times  = np.array([float(t) for _, t in trace], dtype=float)
    n = len(values)
    if n < duration_samples + 2:
        return list(zip(values.tolist(), times.tolist()))

    mid   = n // 2
    start = max(0, mid - duration_samples // 2)
    end   = min(n, start + duration_samples)
    # Flatten to the baseline level just before the QRS, not to values[start]
    # (which would be a Q-wave or pre-R sample and look mid-amplitude).
    baseline = float(values[max(0, start - 5)])
    values[start:end] = baseline
    return list(zip(values.tolist(), times.tolist()))


def _inject_offset_ecg(trace, magnitude):
    """Constant additive shift (baseline drift / electrode bias)."""
    values = np.array([float(v) for v, _ in trace], dtype=float) + magnitude
    times  = np.array([float(t) for _, t in trace], dtype=float)
    return list(zip(values.tolist(), times.tolist()))


def build_injected_negatives_ecg(
        positive_traces,
        out_folder,
        file_prefix,
        n_per_mode=15,
        spike_magnitude=2.0,
        spike_duration_samples=20,
        spike_offset_min=30,            # was: spike_avoid_center_fraction=0.4
        spike_offset_max=80,            # NEW
        phase_shift_fraction=0.33,
        stuck_duration_samples=60,
        offset_magnitude=0.5,
        seed=42,
):
    """
    Generate ECG negative test traces by injecting controlled anomalies into
    real positive traces. Four modes, saved to mode subfolders so the runner
    can produce per-mode rejection rates (matching exp 5.1/5.2/5.4 layout).

    Modes (indices match NEG_MODE_NAMES in Generators.py):
        0 spikes   - brief amplitude excursion at a random position
        1 shifted  - rotate values in time (beat misalignment)
        2 stuck    - flatten a centered segment (asystole)
        3 offset   - constant additive shift (baseline drift)

    Each mode samples n_per_mode positives with replacement, so the same
    positive can be perturbed across multiple modes.

    Parameters
    ----------
    positive_traces : list of [(value, time), ...]
        Real ECG positives, e.g. from csv_to_temp_time_list().
    out_folder : Path or str
        Output root; mode subfolders are created beneath it.
    file_prefix : str
        Filename prefix for generated CSVs.
    n_per_mode : int
        Negatives per mode. Total = 4 * n_per_mode.

    The five magnitude/duration parameters are tuned for ~1 mV-amplitude
    ECG at 360 Hz with 3-beat traces. Adjust if your scale differs:
      - ECG in ADC counts (e.g. ×200) → multiply spike_magnitude and
        offset_magnitude by ~200.
      - Different sampling rate → scale spike_duration_samples and
        stuck_duration_samples by your_fs / 360.
      - Different trace length / n_beats → keep phase_shift_fraction
        around 1/n_beats.

    Returns
    -------
    (negatives, modes)
        negatives : list of [(value, time), ...]
        modes     : list of int, one per trace, 0-3
    Also writes CSVs to disk under out_folder/<mode_name>/.
    """
    from Generators import NEG_MODE_NAMES

    if not positive_traces:
        raise ValueError("No positive traces to inject into.")

    rng = np.random.default_rng(seed)

    injection_specs = [
        (0, lambda t: _inject_spike_ecg(
            t, spike_magnitude, spike_duration_samples, rng,
            spike_offset_min, spike_offset_max)),
        (1, lambda t: _inject_shifted_ecg(t, phase_shift_fraction)),
        (2, lambda t: _inject_stuck_ecg(t, stuck_duration_samples)),
        (3, lambda t: _inject_offset_ecg(t, offset_magnitude)),
    ]

    negatives, modes = [], []
    for mode_int, inject_fn in injection_specs:
        for _ in range(n_per_mode):
            idx = int(rng.integers(0, len(positive_traces)))
            negatives.append(inject_fn(positive_traces[idx]))
            modes.append(mode_int)

    # Write to disk, organized by mode subfolder so the runner can do
    # per-mode evaluation with load_traces_by_mode.
    out_folder = Path(out_folder)
    out_folder.mkdir(parents=True, exist_ok=True)

    mode_counters = {m: 0 for m in NEG_MODE_NAMES}
    for trace, mode_int in zip(negatives, modes):
        mode_name = NEG_MODE_NAMES[mode_int]
        mode_subfolder = out_folder / mode_name
        mode_subfolder.mkdir(parents=True, exist_ok=True)
        mode_counters[mode_int] += 1
        filename = f"neg-{file_prefix}-{mode_name}-tid{mode_counters[mode_int]}.csv"
        with open(mode_subfolder / filename, "w") as f:
            f.write("time;value\n")
            for value, time in trace:
                f.write(f"{time:.0f};{value:.5f}\n")

    counts = {NEG_MODE_NAMES[m]: c for m, c in mode_counters.items() if c > 0}
    print(f"Generated {len(negatives)} ECG negatives -> {out_folder}")
    print(f"  Mode counts: {counts}")
    return negatives, modes

def plot_and_save_traces(
        traces,
        output_folder,
        symbol_map=None,
        positive_traces=None,
        title_prefix=None
):

    os.makedirs(output_folder, exist_ok=True)

    converted_neg = convert_traces_for_plotting(traces, symbol_map)

    converted_pos = None
    if positive_traces is not None:
        converted_pos = convert_traces_for_plotting(
            positive_traces,
            symbol_map
        )

    # ---- categorical plotting setup ----
    if symbol_map is not None:

        inv_map = {v: k for k, v in symbol_map.items()}

        symbols = [inv_map[v] for v in sorted(inv_map.keys())]

        plot_map = {
            s: i for i, s in enumerate(symbols)
        }

    else:
        symbols = None
        plot_map = None

    # ---- plot traces ----
    for i, (neg_times, neg_values) in enumerate(converted_neg):

        plt.figure(figsize=(12, 4))

        # ---------- NEGATIVE ----------
        order = np.argsort(neg_times)
        neg_times = neg_times[order]
        neg_values = neg_values[order]

        if symbol_map is not None:
            neg_values = np.array([
                plot_map[inv_map[v]]
                for v in neg_values
            ])

        plt.step(
            neg_times / 3600,
            neg_values,
            where="post",
            linewidth=2,
            alpha=0.9,
            label="Negative"
        )

        # ---------- POSITIVE ----------
        if converted_pos is not None and i < len(converted_pos):

            pos_times, pos_values = converted_pos[i]

            order = np.argsort(pos_times)
            pos_times = pos_times[order]
            pos_values = pos_values[order]

            if symbol_map is not None:
                pos_values = np.array([
                    plot_map[inv_map[v]]
                    for v in pos_values
                ])

            plt.step(
                pos_times / 3600,
                pos_values,
                where="post",
                linestyle="--",
                linewidth=2,
                alpha=0.8,
                label="Positive"
            )

        # ---------- y-axis ----------
        if symbol_map is not None:
            plt.yticks(
                range(len(symbols)),
                symbols
            )

        base_title = f"Trace {i}"

        if title_prefix is not None:
            base_title = f"{base_title} | {title_prefix}"

        plt.title(base_title)
        plt.xlabel("Time (hours)")
        plt.ylabel("Symbol")
        plt.grid(True, alpha=0.3)
        plt.legend()

        path = os.path.join(
            output_folder,
            f"trace-{i}-{title_prefix}.png"
        )

        plt.savefig(path, dpi=150)
        plt.close()

    print(f"Saved {len(converted_neg)} overlay traces to {output_folder}")

def convert_traces_for_plotting(traces, symbol_map=None):
    """
    Converts symbolic traces into numeric format for plotting only.
    Does NOT affect TA pipeline.
    """

    plot_traces = []

    inv_map = None
    if symbol_map is not None:
        inv_map = {v: k for k, v in symbol_map.items()}

    for trace in traces:
        times = []
        values = []

        for event in trace:
            s, t = event.split(":")

            # time
            t = float(t)


            # convert symbol to numeric index for plotting
            if symbol_map is not None and isinstance(symbol_map.get(s, None), int):
                v = symbol_map[s]
            else:
                # fallback: hash-style stable mapping
                v = abs(hash(s)) % 20

            times.append(t)
            values.append(v)

        plot_traces.append((np.array(times), np.array(values)))
    return plot_traces




def generate_negative_samples(
        positive_traces,
        out_folder,
        file_prefix,
        time_jitter=0.5,
        value_jitter=1.5,
        swap_prob=0.25,
        mutate_prob=0.25,
        spike_prob=0.10,
        seed=None,
):
    """
      Convert real observed traces (positive samples) into synthetic negative samples.

      Negative traces:
          - keep SAME length as positives
          - stay realistic (not random noise)
          - violate timing, order, and symbol/bin consistency

      Args:
          positive_traces (list[list[str]]):
              Each trace is a sequence like ["a:10", "b:5", "c:7"]

          out_folder (str, optional):
              If set, saves visual comparisons of positive vs negative traces.

          time_jitter (float):
              Strength of timestamp perturbation.

              Higher values → larger timing distortion.

          swap_prob (float):
              Probability of swapping adjacent events.

              Introduces local ordering errors.

          mutate_prob (float):
              Probability of changing a symbol/bin.

              Uses symbol_map for realistic replacements.

          spike_prob (float):
              Probability that a mutation becomes a large bin "spike"
              instead of a local change.

          seed (int, optional):
              Random seed for reproducibility.

          symbol_map (dict):
              Maps symbols to ordered bins for structured mutations.

      Returns:
          list[list[str]]:
              Negative traces with same structure and length as inputs.
      """

    # ---------------------------------------------------------------
    # Reproducibility
    # ---------------------------------------------------------------
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    negative_traces = []

    # ===============================================================
    # PROCESS EACH TRACE
    # ===============================================================
    for trace in positive_traces:

        # -----------------------------------------------------------
        # Split values and times
        # -----------------------------------------------------------
        values = np.array([
            float(v) for v, _ in trace
        ])

        times = np.array([
            float(t) for _, t in trace
        ])

        original_values = values.copy()
        original_times = times.copy()



        times = original_times.copy()

        # ===========================================================
        # 2. VALUE MUTATION
        # ===========================================================
        for i in range(len(values)):

            if random.random() < mutate_prob:

                # ---------------------------------------------------
                # LARGE SPIKE
                # ---------------------------------------------------
                if random.random() < spike_prob:

                    noise = np.random.normal(
                        0,
                        value_jitter * 4
                    )

                # ---------------------------------------------------
                # SMALL LOCAL CHANGE
                # ---------------------------------------------------
                else:

                    noise = np.random.normal(
                        0,
                        value_jitter
                    )

                values[i] += noise

        # ===========================================================
        # 3. LOCAL SWAPS
        # ===========================================================
        for i in range(len(values)):

            if random.random() < swap_prob:

                direction = random.choice([-1, 1])

                j = i + direction

                if 0 <= j < len(values):

                    # swap values
                    values[i], values[j] = (
                        values[j],
                        values[i]
                    )

                    # swap timestamps
                    # times[i], times[j] = (
                    #     times[j],
                    #     times[i]
                    # )

        # ===========================================================
        # 4. REBUILD RAW TRACE
        # ===========================================================
        neg_trace = list(zip(values, times))

        # ensure actually changed
        changed = (
                not np.array_equal(values, original_values)
                or
                not np.array_equal(times, original_times)
        )

        if changed:
            negative_traces.append(neg_trace)

    # ---------------------------------------------------------------
    # SAVE RAW NEGATIVE FILES
    # ---------------------------------------------------------------
    out_folder = Path(out_folder)
    out_folder.mkdir(parents=True, exist_ok=True)

    for i, trace in enumerate(negative_traces):

        file_path = out_folder / f"neg-{file_prefix}-tid{i}.csv"

        with open(file_path, "w") as f:
            # HEADER
            f.write("time_seconds;temperature\n")

            for value, time in trace:

                # RAW FORMAT
                f.write(f"{time:.0f};{value:.5f}\n")

    print("Negative samples generated")

    return negative_traces




def split_dataset(
        input_folder,
        output_folder,
        prefix,
        train_ratio=0.7,
        seed=42
):
    """
    Random but representative split of dataset.

    Ensures:
        - shuffled distribution
        - no ordering bias
        - reproducible split
    """

    input_folder = Path(input_folder)
    output_folder = Path(output_folder)


    random.seed(seed)

    train_dir = output_folder / f"{prefix}-train"
    test_dir = output_folder /  f"{prefix}-test" / "positive"

    train_dir.mkdir(parents=True, exist_ok=True)
    test_dir.mkdir(parents=True, exist_ok=True)

    files = list(input_folder.rglob("*"))
    files = [f for f in files if f.is_file()]
    random.shuffle(files)

    split_idx = int(len(files) * train_ratio)

    train_files = files[:split_idx]
    test_files = files[split_idx:]

    for f in train_files:
        shutil.copy2(f, train_dir / f.name)

    for f in test_files:
        shutil.copy2(f, test_dir / f.name)

    print(f"Train: {len(train_files)} files")
    print(f"Test: {len(test_files)} files")


BASE_DIR = Path(__file__).resolve().parents[2]
SEED = 42

room = "A"
period = "1day"
beats = "1beat"

file_prefix = f"room{room}-{period}"

file_ecg_prefix = f"{beats}"

all_traces_folder = BASE_DIR / "Data" / "3-ExtractInterval" /  f"{period}-experiment" / f"room{room}"
output_folder = BASE_DIR / "Data" / "3-ExtractInterval" /  f"{period}-experiment"
test_positive_folder = BASE_DIR / "Data" / "3-ExtractInterval" / f"{period}-experiment"/f"{room}-test/positive"
test_negative_folder = BASE_DIR / "Data" / "3-ExtractInterval" / f"{period}-experiment"/f"{room}-test/negative"


all_ecg_traces_folder = BASE_DIR / "Data" / "3-ExtractInterval" /  "ecg" / "1beat"
output_ecg_folder = BASE_DIR / "Data" / "3-ExtractInterval" / "ecg" / f"{beats}-experiment1"

test_ecg_positive_folder = BASE_DIR / "Data" / "3-ExtractInterval" /  "ecg" / f"{beats}-experiment1"/f"{beats}-test/positive"
test_ecg_negative_folder = BASE_DIR / "Data" / "3-ExtractInterval" / "ecg" / f"{beats}-experiment1"/f"{beats}-test/negative"



# =============================================================================
# ECG negative generation workflow
# Run once to populate the negative folder, then run Metrics_ecg_run.py.
# =============================================================================

# Step 1 (one-time): split your ECG positives into train/test
# split_dataset(
#     input_folder  = all_ecg_traces_folder,
#     output_folder = output_ecg_folder,
#     prefix        = beats,
# )

#Step 2: generate negatives from the test positives
# test_positive_files = get_trace_files(folder_path=test_ecg_positive_folder)
# test_positive_lists = csv_to_temp_time_list(input_files=test_positive_files)
# build_injected_negatives_ecg(
#     positive_traces = test_positive_lists,
#     out_folder      = test_ecg_negative_folder,
#     file_prefix     = file_ecg_prefix,
#     n_per_mode      = 43,
# )

# =============================================================================
# Temperature negative generation
# Positives are already split into A-train / A-test/positive; skip split_dataset.
# =============================================================================

# test_positive_files = get_trace_files(folder_path=test_positive_folder)
# test_positive_lists = csv_to_temp_time_list(input_files=test_positive_files)
# build_injected_negatives_temp(
#     positive_traces = test_positive_lists,
#     out_folder      = test_negative_folder,
#     file_prefix     = file_prefix,
#     n_per_mode      = 22,        # 4 × 22 = 88 (close to your previous 86)
# )

#
# split_dataset(input_folder = all_ecg_traces_folder,
#               output_folder = output_ecg_folder,
#               prefix = beats)
#
#
#
# test_positive_raw_traces = get_trace_files(folder_path = test_ecg_positive_folder)
# test_positive_raw_lists = csv_to_temp_time_list(input_files=test_positive_raw_traces)
#
# test_negative_raw_lists = generate_negative_samples( positive_traces = test_positive_raw_lists, seed= SEED, out_folder=test_ecg_negative_folder, file_prefix =file_ecg_prefix )
#

#
