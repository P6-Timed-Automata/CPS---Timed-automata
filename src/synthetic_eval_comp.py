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
# SINGLE EXPERIMENT
# =============================================================================
def run_experiment(label, BASE_DIR, train_files, test_files, neg_files,
                   symbols, k_val, suffix):
    period                = "synthetic"
    discretization_method = "naiv"

    training_txt_path = (
        BASE_DIR / "Data" / "4-DiscretizationData" / discretization_method / period
        / f"{suffix}-training-s{symbols}.txt"
    )
    TA_output_path = (
        BASE_DIR / "Data" / "5-TaResults" / discretization_method / period / suffix
    )
    os.makedirs(TA_output_path, exist_ok=True)
    os.makedirs(training_txt_path.parent, exist_ok=True)

    data_lists_train = csv_to_temp_time_list(input_files=train_files)
    data_lists_test  = csv_to_temp_time_list(input_files=test_files)
    data_lists_neg   = csv_to_temp_time_list(input_files=neg_files)

    traces_train, bins = equal_width_discretization(data_lists_train, symbols)
    symbolic_train, symbol_map, _ = map_bins_to_symbols(traces_train, symbols, bins)
    format_output(symbolic_train, training_txt_path)

    symbolic_pos_str = preprocess_test_traces(data_lists_test, bins, symbols)
    symbolic_neg_str = preprocess_test_traces(data_lists_neg,  bins, symbols)

    learner = TALearner(tss_path=str(training_txt_path), display=False, k=k_val)

    metrics = learner.ta.evaluate_classifier(
        positive_tss=symbolic_pos_str,
        negative_tss=symbolic_neg_str,
        timed=True,
        save_path=str(TA_output_path / "metrics.csv"),
        run_id=f"s{symbols}_k{k_val}"
    )

    # Split FN into symbol vs timing failures
    fn_symbol = fn_timing = 0
    for ts in symbolic_pos_str:
        timed_ok   = learner.ta.inconsistency_nb([ts], timed=True,  show=False, p=False) == 0
        untimed_ok = learner.ta.inconsistency_nb([ts], timed=False, show=False, p=False) == 0
        if not timed_ok and not untimed_ok:
            fn_symbol += 1
        elif not timed_ok and untimed_ok:
            fn_timing += 1

    metrics.update({'label': label, 'fn_symbol': fn_symbol,
                    'fn_timing': fn_timing, 'bins': bins})

    print(f"[{label}]  TP={metrics['TP']} FP={metrics['FP']} "
          f"TN={metrics['TN']} FN={metrics['FN']}  "
          f"(FN: {fn_symbol} symbol, {fn_timing} timing)")
    print(f"  Precision={metrics['precision']:.3f}  "
          f"Recall={metrics['recall']:.3f}  F1={metrics['f1']:.3f}")

    return metrics


# =============================================================================
# PLOTTING
# =============================================================================
def plot_comparison(results, output_path):
    labels     = [r['label'] for r in results]
    precisions = [r['precision'] for r in results]
    recalls    = [r['recall']    for r in results]
    f1s        = [r['f1']        for r in results]
    fn_sym     = [r['fn_symbol'] for r in results]
    fn_tim     = [r['fn_timing'] for r in results]
    x     = np.arange(len(labels))
    width = 0.25

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    ax1.bar(x - width, precisions, width, label='Precision', color='steelblue',  alpha=0.85)
    ax1.bar(x,         recalls,    width, label='Recall',    color='darkorange', alpha=0.85)
    ax1.bar(x + width, f1s,        width, label='F1',        color='seagreen',   alpha=0.85)
    for i, (p, r, f) in enumerate(zip(precisions, recalls, f1s)):
        ax1.text(i - width, p + 0.01, f"{p:.2f}", ha='center', fontsize=8)
        ax1.text(i,         r + 0.01, f"{r:.2f}", ha='center', fontsize=8)
        ax1.text(i + width, f + 0.01, f"{f:.2f}", ha='center', fontsize=8)
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels)
    ax1.set_ylim(0, 1.15)
    ax1.set_ylabel("Score")
    ax1.set_title("Classifier metrics: clean vs noisy")
    ax1.legend()
    ax1.grid(True, linestyle='--', alpha=0.4, axis='y')

    ax2.bar(x, fn_sym, width * 2, label='FN: symbol sequence', color='crimson', alpha=0.85)
    ax2.bar(x, fn_tim, width * 2, label='FN: timing guard',    color='salmon',  alpha=0.85,
            bottom=fn_sym)
    for i, (s, t) in enumerate(zip(fn_sym, fn_tim)):
        if s + t > 0:
            ax2.text(i, s + t + 0.1, str(s + t), ha='center', fontsize=9)
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels)
    ax2.set_ylabel("False negatives")
    ax2.set_title("FN breakdown: why positives were rejected")
    ax2.legend()
    ax2.grid(True, linestyle='--', alpha=0.4, axis='y')

    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved comparison plot to {output_path}")


def plot_trace_overlay(clean_traces, noisy_traces, output_path):
    fig, axes = plt.subplots(1, 2, figsize=(13, 4), sharey=True)
    for ax, traces, title in zip(
        axes,
        [clean_traces, noisy_traces],
        [f"Clean (n={len(clean_traces)})", f"Noisy (n={len(noisy_traces)})"]
    ):
        colors = plt.cm.viridis(np.linspace(0.1, 0.9, len(traces)))
        for i, (t, v) in enumerate(traces):
            ax.plot(t / 3600, v, color=colors[i], linewidth=0.7, alpha=0.5)
        t_grid = traces[0][0]
        mean_v = np.mean([np.interp(t_grid, t, v) for t, v in traces], axis=0)
        ax.plot(t_grid / 3600, mean_v, color='black', linewidth=2,
                linestyle='--', label='Mean')
        ax.set_title(title)
        ax.set_xlabel("Time (hours)")
        ax.set_ylabel("Temperature (°C)")
        ax.legend()
        ax.grid(True, linestyle='--', alpha=0.3)
    fig.suptitle("Training data: clean vs noisy", fontsize=12)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved trace overlay to {output_path}")


# =============================================================================
# MAIN
# =============================================================================
if __name__ == "__main__":
    total_start = time.perf_counter()
    BASE_DIR = Path(__file__).resolve().parent.parent

    # Only 5 bins as 11 made it problematic (only ,2 por someting in each bin)
    symbols = 5
    k_val   = 2
    n_train = 100
    n_test  = 20
    n_neg   = 20

    out_root     = BASE_DIR / "Data" / "synthetic_data"
    graph_folder = BASE_DIR / "Data" / "Graphs" / "synthetic_comparison"
    os.makedirs(graph_folder, exist_ok=True)

    print("Generating datasets...")

    # FIX 2: generate train+test from ONE seed, then split
    # This ensures train and test are drawn from the same distribution
    all_clean = generate_trace_set(
        n_traces    = n_train + n_test,
        noise_std   = 0.05,
        base_temp_std  = 0.0,
        amplitude_std  = 0.1,
        phase_std_h    = 0.1,   # small but nonzero — realistic clean data
        seed        = 42
    )
    all_noisy = generate_trace_set(
        n_traces    = n_train + n_test,
        noise_std   = 0.3,
        base_temp_std  = 2.0,   # different temperature regimes
        amplitude_std  = 1.0,   # varying swing
        phase_std_h    = 1.0,   # timing shifts up to ~1h
        seed        = 42        # same seed — difference is purely the parameters
    )
    neg_traces = generate_negative_trace_set(n_traces=n_neg, seed=99)

    # Split into train / test
    clean_train, clean_test = all_clean[:n_train], all_clean[n_train:]
    noisy_train, noisy_test = all_noisy[:n_train], all_noisy[n_train:]

    # Save
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

    plot_trace_overlay(clean_train, noisy_train,
                       str(graph_folder / "training_data_overview.png"))

    print("\n" + "="*50)
    print("EXPERIMENT 1: Clean data (low variation)")
    print("="*50)
    clean_metrics = run_experiment(
        "Clean", BASE_DIR,
        train_files=clean_train_files, test_files=clean_test_files,
        neg_files=neg_files, symbols=symbols, k_val=k_val, suffix="clean"
    )

    print("\n" + "="*50)
    print("EXPERIMENT 2: Noisy data (high variation)")
    print("="*50)
    noisy_metrics = run_experiment(
        "Noisy", BASE_DIR,
        train_files=noisy_train_files, test_files=noisy_test_files,
        neg_files=neg_files, symbols=symbols, k_val=k_val, suffix="noisy"
    )

    plot_comparison([clean_metrics, noisy_metrics],
                    str(graph_folder / "clean_vs_noisy_comparison.png"))

    print("\n" + "="*50)
    print("SUMMARY")
    print(f"{'':20s}  {'Clean':>8}  {'Noisy':>8}")
    for key, label in [('precision','Precision'), ('recall','Recall'),
                        ('f1','F1'), ('fn_symbol','FN (symbol)'), ('fn_timing','FN (timing)')]:
        cv = clean_metrics[key]
        nv = noisy_metrics[key]
        fmt = ".3f" if isinstance(cv, float) else "d"
        print(f"  {label:20s}  {cv:>8{fmt}}  {nv:>8{fmt}}")
    print("="*50)

    print(f"\nTotal runtime: {time.perf_counter() - total_start:.2f}s")