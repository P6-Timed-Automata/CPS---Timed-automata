import os
import time
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from TAG.TALearner import TALearner
from Discretization.discretizationSetup import (
    csv_to_temp_time_list,
    format_output,
    map_bins_to_symbols,
    preprocess_test_traces,
)
from Discretization.naive import equal_width_discretization
from Discretization.sax import sax_discretization_multi
from Discretization.persist import (
    Persist, get_best_bins, discretize_traces_with_bins, flatten_traces_to_ts
)
from DataProcessing.processData import get_trace_files

# =============================================================================
# GENERATORS
# =============================================================================
def generate_synthetic_trace(duration_hours=24, sample_interval=300, base_temp=22.0,
                             amplitude=3.0, phase_shift_h=0.0, noise_std=0.1, rng=None):
    if rng is None:
        rng = np.random.default_rng()
    times = np.arange(0, duration_hours * 3600, sample_interval, dtype=float)
    hours = times / 3600
    trough_hour = 4.0 + phase_shift_h
    cycle = -np.cos(2 * np.pi * (hours - trough_hour) / 24)
    temps = base_temp + (amplitude / 2) * cycle
    temps += rng.normal(0, noise_std, size=len(times))
    return times, temps


def generate_trace_set(n_traces, base_temp=22.0, amplitude=3.0,
                       base_temp_std=0.0, amplitude_std=0.2,
                       phase_std_h=0.25, noise_std=0.1, seed=42):
    rng = np.random.default_rng(seed)
    traces = []
    for _ in range(n_traces):
        bt = base_temp + rng.normal(0, base_temp_std)
        am = max(0.5, amplitude + rng.normal(0, amplitude_std))
        ph = rng.normal(0, phase_std_h)
        t, v = generate_synthetic_trace(base_temp=bt, amplitude=am,
                                        phase_shift_h=ph, noise_std=noise_std, rng=rng)
        traces.append((t, v))
    return traces


def generate_negative_trace_set(n_traces=20, seed=99):
    rng = np.random.default_rng(seed)
    neg_traces = []
    for i in range(n_traces):
        mode = i % 4
        t, v = generate_synthetic_trace(duration_hours=24, noise_std=0.05, rng=rng)
        if mode == 0:
            spike_indices = rng.choice(len(v), size=3, replace=False)
            v[spike_indices] += rng.uniform(5.0, 10.0)
        elif mode == 1:
            t, v = generate_synthetic_trace(duration_hours=24, phase_shift_h=12.0, rng=rng)
        elif mode == 2:
            v[40:120] = v[40]
        else:
            v += 15.0
        neg_traces.append((t, v))
    return neg_traces


def save_traces_as_csv(traces, output_folder, prefix="synthetic"):
    output_folder = Path(output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)
    for i, (t, v) in enumerate(traces):
        path = output_folder / f"{prefix}_tid{i + 1}.csv"
        with open(path, 'w') as f:
            f.write("time_s;temperature\n")
            for ti, vi in zip(t, v):
                f.write(f"{int(ti)};{vi:.5f}\n")


# =============================================================================
# DISCRETIZATION WRAPPERS
# =============================================================================
def discretize_naive(data_lists_train, data_lists_test, data_lists_neg,
                     symbols, training_txt_path):
    traces_train, bins = equal_width_discretization(data_lists_train, symbols)
    symbolic_train, _, _ = map_bins_to_symbols(traces_train, symbols, bins)
    format_output(symbolic_train, training_txt_path)

    symbolic_pos_str = preprocess_test_traces(data_lists_test, bins, symbols)
    symbolic_neg_str = preprocess_test_traces(data_lists_neg,  bins, symbols)
    return symbolic_pos_str, symbolic_neg_str, symbols


def preprocess_sax_traces(data_lists, breakpoints, sax_w, symbols,
                          norm_mode="per_trace", global_mean=None, global_std=None):
    import string
    alphabet = list(string.ascii_lowercase)[:symbols]
    mapping  = {i: alphabet[i] for i in range(symbols)}

    symbolic_traces = []
    for trace in data_lists:
        v = np.array([val for val, _ in trace])
        t = np.array([time for _, time in trace])

        if norm_mode == "global":
            v_norm = (v - global_mean) / global_std if global_std != 0 else np.zeros_like(v)
        else:  # per_trace
            sigma  = v.std()
            v_norm = (v - v.mean()) / sigma if sigma != 0 else np.zeros_like(v)

        v_segs = np.array_split(v_norm, sax_w)
        t_segs = np.array_split(t, sax_w)
        paa_v  = np.array([seg.mean() for seg in v_segs])
        paa_t  = np.array([int(seg.mean()) for seg in t_segs])

        labels = np.digitize(paa_v, breakpoints, right=False)
        labels = np.clip(labels, 0, symbols - 1)

        symbolic_traces.append([f"{mapping[l]}:{int(ts)}" for l, ts in zip(labels, paa_t)])

    return symbolic_traces


def discretize_sax(data_lists_train, data_lists_test, data_lists_neg,
                   symbols, sax_w, training_txt_path, norm_mode="global"):
    from scipy.stats import norm
    breakpoints = norm.ppf(np.linspace(0, 1, symbols + 1)[1:-1])

    all_train_vals = np.concatenate([
        np.array([v for v, _ in trace]) for trace in data_lists_train
    ])
    global_mean = float(all_train_vals.mean())
    global_std  = float(all_train_vals.std())

    traces_train, bins_z = sax_discretization_multi(data_lists_train, w=sax_w, k=symbols)
    symbolic_train, _, _ = map_bins_to_symbols(traces_train, symbols, bins_z)
    format_output(symbolic_train, training_txt_path)

    kwargs = dict(breakpoints=breakpoints, sax_w=sax_w, symbols=symbols,
                  norm_mode=norm_mode, global_mean=global_mean, global_std=global_std)

    symbolic_pos_str = preprocess_sax_traces(data_lists_test, **kwargs)
    symbolic_neg_str = preprocess_sax_traces(data_lists_neg,  **kwargs)
    return symbolic_pos_str, symbolic_neg_str, symbols


def discretize_persist(data_lists_train, data_lists_test, data_lists_neg,
                       symbols, training_txt_path):
    ts = flatten_traces_to_ts(data_lists_train)
    persist_obj  = Persist(ts, break_min=2, break_max=10,
                           skip=np.array([4, 4]))
    bins         = get_best_bins(persist_obj, ts)
    actual_k     = len(bins) - 1

    traces_train = discretize_traces_with_bins(data_lists_train, bins)
    symbolic_train, _, _ = map_bins_to_symbols(traces_train, actual_k, bins)
    format_output(symbolic_train, training_txt_path)

    symbolic_pos_str = preprocess_test_traces(data_lists_test, bins, actual_k)
    symbolic_neg_str = preprocess_test_traces(data_lists_neg,  bins, actual_k)
    return symbolic_pos_str, symbolic_neg_str, actual_k


# =============================================================================
# SINGLE EXPERIMENT
# =============================================================================
def run_experiment(label, disc_method, BASE_DIR,
                   train_files, test_files, neg_files,
                   symbols, k_val, sax_w=20):

    period = "synthetic"
    suffix = label.lower().replace(" ", "_")

    # Routing for SAX variants
    actual_disc_method = disc_method
    sax_norm = "global"
    if disc_method == "sax_per_trace":
        actual_disc_method = "sax"
        sax_norm = "per_trace"
    elif disc_method == "sax_global":
        actual_disc_method = "sax"
        sax_norm = "global"

    training_txt_path = (
        BASE_DIR / "Data" / "4-DiscretizationData" / disc_method / period
        / f"{suffix}-training-s{symbols}.txt"
    )
    TA_output_path = (
        BASE_DIR / "Data" / "5-TaResults" / disc_method / period / suffix
    )
    os.makedirs(TA_output_path, exist_ok=True)
    os.makedirs(training_txt_path.parent, exist_ok=True)

    data_lists_train = csv_to_temp_time_list(input_files=train_files)
    data_lists_test  = csv_to_temp_time_list(input_files=test_files)
    data_lists_neg   = csv_to_temp_time_list(input_files=neg_files)

    if actual_disc_method == "naiv":
        symbolic_pos_str, symbolic_neg_str, actual_k = discretize_naive(
            data_lists_train, data_lists_test, data_lists_neg,
            symbols, training_txt_path
        )
    elif actual_disc_method == "sax":
        symbolic_pos_str, symbolic_neg_str, actual_k = discretize_sax(
            data_lists_train, data_lists_test, data_lists_neg,
            symbols, sax_w, training_txt_path, norm_mode=sax_norm
        )
    elif actual_disc_method == "persist":
        symbolic_pos_str, symbolic_neg_str, actual_k = discretize_persist(
            data_lists_train, data_lists_test, data_lists_neg,
            symbols, training_txt_path
        )
    else:
        raise ValueError(f"Unknown method: {disc_method}")

    learner = TALearner(tss_path=str(training_txt_path), display=False, k=k_val)

    metrics = learner.ta.evaluate_classifier(
        positive_tss=symbolic_pos_str,
        negative_tss=symbolic_neg_str,
        timed=True,
        save_path=str(TA_output_path / "metrics.csv"),
        run_id=f"{suffix}_s{actual_k}_k{k_val}"
    )

    fn_symbol = fn_timing = 0
    for ts in symbolic_pos_str:
        timed_ok   = learner.ta.inconsistency_nb([ts], timed=True,  show=False, p=False) == 0
        untimed_ok = learner.ta.inconsistency_nb([ts], timed=False, show=False, p=False) == 0
        if not timed_ok and not untimed_ok:
            fn_symbol += 1
        elif not timed_ok and untimed_ok:
            fn_timing += 1

    metrics.update({'label': label, 'disc_method': disc_method,
                    'fn_symbol': fn_symbol, 'fn_timing': fn_timing})

    learner.ta.show(title=label, savePng=True, output_path=str(TA_output_path))
    return metrics


# =============================================================================
# PLOTTING
# =============================================================================
def plot_results(all_results, output_path):
    method_display_names = {
        "naiv": "Naive",
        "sax_global": "SAX (Global)",
        "sax_per_trace": "SAX (PT)",
        "persist": "Persist"
    }
    methods      = ["naiv", "sax_global", "sax_per_trace", "persist"]
    conditions   = ["Clean", "Noisy"]
    cond_colors  = {'Clean': 'steelblue', 'Noisy': 'darkorange'}

    results_dict = {}
    for r in all_results:
        m = r['disc_method']
        cond = 'Clean' if 'clean' in r['label'].lower() else 'Noisy'
        results_dict.setdefault(m, {})[cond] = r

    x      = np.arange(len(methods))
    width  = 0.35

    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    metric_keys = [('precision', 'Precision'), ('recall', 'Recall'), ('f1', 'F1')]

    for ax, (key, title) in zip(axes[:3], metric_keys):
        for ci, cond in enumerate(conditions):
            vals = [results_dict.get(m, {}).get(cond, {}).get(key, 0) for m in methods]
            offset = (ci - 0.5) * width
            bars = ax.bar(x + offset, vals, width, label=cond,
                          color=cond_colors[cond], alpha=0.85)
            for bar, v in zip(bars, vals):
                ax.text(bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + 0.01, f"{v:.2f}",
                        ha='center', va='bottom', fontsize=7)
        ax.set_xticks(x)
        ax.set_xticklabels([method_display_names[m] for m in methods])
        ax.set_ylim(0, 1.2)
        ax.set_title(title)
        ax.legend(fontsize=8)
        ax.grid(True, linestyle='--', alpha=0.4, axis='y')

    ax = axes[3]
    for ci, cond in enumerate(conditions):
        fn_sym = [results_dict.get(m, {}).get(cond, {}).get('fn_symbol', 0) for m in methods]
        fn_tim = [results_dict.get(m, {}).get(cond, {}).get('fn_timing', 0) for m in methods]
        offset = (ci - 0.5) * width
        ax.bar(x + offset, fn_sym, width, color=cond_colors[cond], alpha=0.85,
               label=f"{cond} — symbol")
        ax.bar(x + offset, fn_tim, width, color=cond_colors[cond], alpha=0.4,
               bottom=fn_sym, label=f"{cond} — timing")

    ax.set_xticks(x)
    ax.set_xticklabels([method_display_names[m] for m in methods])
    ax.set_title("FN breakdown")
    ax.legend(fontsize=7)
    ax.grid(True, linestyle='--', alpha=0.4, axis='y')

    fig.suptitle("Comparison of Discretization and Normalization Methods", fontsize=12)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


# =============================================================================
# MAIN
# =============================================================================
if __name__ == "__main__":
    total_start = time.perf_counter()
    BASE_DIR = Path(__file__).resolve().parent.parent

    symbols = 5
    k_val   = 2
    sax_w   = 20
    n_train, n_test, n_neg = 15, 10, 10

    out_root     = BASE_DIR / "Data" / "synthetic_data"
    graph_folder = BASE_DIR / "Data" / "Graphs" / "synthetic_comparison"
    os.makedirs(graph_folder, exist_ok=True)

    all_clean = generate_trace_set(n_traces=n_train + n_test, seed=42, noise_std=0.05)
    all_noisy = generate_trace_set(n_traces=n_train + n_test, seed=42, noise_std=0.3,
                                   base_temp_std=2.0, amplitude_std=1.0, phase_std_h=1.0)
    neg_traces = generate_negative_trace_set(n_traces=n_neg, seed=99)

    clean_train, clean_test = all_clean[:n_train], all_clean[n_train:]
    noisy_train, noisy_test = all_noisy[:n_train], all_noisy[n_train:]

    save_traces_as_csv(clean_train, out_root / "clean_train", prefix="clean_train")
    save_traces_as_csv(clean_test,  out_root / "clean_test",  prefix="clean_test")
    save_traces_as_csv(noisy_train, out_root / "noisy_train", prefix="noisy_train")
    save_traces_as_csv(noisy_test,  out_root / "noisy_test",  prefix="noisy_test")
    save_traces_as_csv(neg_traces,  out_root / "negative",    prefix="neg")

    clean_train_files = get_trace_files(out_root / "clean_train")
    clean_test_files  = get_trace_files(out_root / "clean_test")
    noisy_train_files = get_trace_files(out_root / "noisy_train")
    noisy_test_files  = get_trace_files(out_root / "noisy_test")
    neg_files         = get_trace_files(out_root / "negative")

    all_results = []
    methods_to_test = ["naiv", "sax_global", "sax_per_trace", "persist"]

    for disc in methods_to_test:
        print(f"\nMethod: {disc.upper()}")
        all_results.append(run_experiment(f"clean_{disc}", disc, BASE_DIR, clean_train_files,
                                          clean_test_files, neg_files, symbols, k_val, sax_w))
        all_results.append(run_experiment(f"noisy_{disc}", disc, BASE_DIR, noisy_train_files,
                                          noisy_test_files, neg_files, symbols, k_val, sax_w))

    plot_results(all_results, str(graph_folder / "method_comparison.png"))
    print(f"\nTotal runtime: {time.perf_counter() - total_start:.2f}s")