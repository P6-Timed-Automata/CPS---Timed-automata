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
beats = "3beat"

file_prefix = f"room{room}-{period}"

file_ecg_prefix = f"{beats}"

all_traces_folder = BASE_DIR / "Data" / "3-ExtractInterval" /  f"{period}-experiment" / f"room{room}"
output_folder = BASE_DIR / "Data" / "3-ExtractInterval" /  f"{period}-experiment"
test_positive_folder = BASE_DIR / "Data" / "3-ExtractInterval" / f"{period}-experiment"/f"{room}-test/positive"
test_negative_folder = BASE_DIR / "Data" / "3-ExtractInterval" / f"{period}-experiment"/f"{room}-test/negative"


all_ecg_traces_folder = BASE_DIR / "Data" / "3-ExtractInterval" /  "ecg" / "1beat"
output_ecg_folder = BASE_DIR / "Data" / "3-ExtractInterval" / "ecg" / f"{beats}-experiment"

test_ecg_positive_folder = BASE_DIR / "Data" / "3-ExtractInterval" /  "ecg" / f"{beats}-experiment"/f"{beats}-test/positive"
test_ecg_negative_folder = BASE_DIR / "Data" / "3-ExtractInterval" / "ecg" / f"{beats}-experiment"/f"{beats}-test/negative"




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
#
