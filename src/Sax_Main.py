import os
import string
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
from scipy.stats import norm

from TAG.TALearner import TALearner
from Discretization.discretizationSetup import format_output
from Discretization.sax import csv_to_temp_time_list
from DataProcessing.processData import get_trace_files


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


def check_normality(data_lists, w, trace_nr, output_dir):
    """
    For each trace in data_lists: z-normalize, apply PAA, then plot histogram
    vs N(0,1) and a Q-Q plot. Also runs D'Agostino-Pearson normality test.
    Saves one figure per trace to output_dir.
    """
    os.makedirs(output_dir, exist_ok=True)

    def znorm(v):
        sigma = v.std()
        return (v - v.mean()) / sigma if sigma != 0 else np.zeros_like(v)

    for i, trace in enumerate(data_lists):
        v = znorm(np.array([val for val, _ in trace]))
        paa_segs = np.array_split(v, w)
        paa_v = np.array([seg.mean() for seg in paa_segs])

        stat, p = stats.normaltest(paa_v)
        result_label = f"D'Agostino-Pearson: stat={stat:.3f}, p={p:.4f} ({'normal' if p > 0.05 else 'NOT normal'} at α=0.05)"

        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        fig.suptitle(f"Trace {trace_nr}, file {i+1} — {result_label}", fontsize=10)

        # Histogram vs N(0,1)
        axes[0].hist(paa_v, bins=20, density=True, alpha=0.6, label="PAA means")
        x = np.linspace(paa_v.min() - 1, paa_v.max() + 1, 200)
        axes[0].plot(x, stats.norm.pdf(x), 'r-', label='N(0,1)')
        axes[0].set_title("Histogram of z-normed PAA means")
        axes[0].legend()

        # Q-Q plot
        stats.probplot(paa_v, dist="norm", plot=axes[1])
        axes[1].set_title("Q-Q Plot")

        plt.tight_layout()
        save_path = os.path.join(output_dir, f"normality-trace{trace_nr}-file{i+1}.png")
        plt.savefig(save_path)
        plt.close()
        print(f"  Normality check saved: {save_path} | {result_label}")

# --- Config ---
room = "A"
w_values = [50, 100, 500, 1000, 2000, 4000, 8609]  # PAA segment counts to test
discretization_method = "sax"
period = "30day"

experiment_folder = f"../Data/3-ExtractInterval/{period}-experiment"

for trace_nr in range(1, 2):
    raw_traces = get_trace_files(folder_path=experiment_folder, max_files=trace_nr)
    data_lists = [csv_to_temp_time_list(f) for f in raw_traces]

    for w in w_values:
        normality_output = f"../Data/5-TaResults/{discretization_method}/{period}/normality-checks/w{w}"
        check_normality(data_lists, w=w, trace_nr=trace_nr, output_dir=normality_output)

        for k in range(2, 13, 2):
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

            print(f"Done: trace={trace_nr}, w={w}, k={k}")