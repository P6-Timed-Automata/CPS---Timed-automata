import os
import time
import numpy as np
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


def generate_trace_set(n_traces=20, base_temp=22.0, amplitude=3.0,
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


if __name__ == "__main__":
    total_start = time.perf_counter()
    BASE_DIR = Path(__file__).resolve().parent.parent

    symbols = 11
    k_val   = 2
    n_train = 15   # traces used to build the TA
    n_test  = 5    # held-out positives — NOT seen during training

    period                = "synthetic"
    discretization_method = "naiv"
    room                  = "synthetic_eval"

    out_root   = BASE_DIR / "Data" / "synthetic_data"
    pos_folder = out_root / "clean"
    neg_folder = out_root / "negative"

    training_txt_path = (
        BASE_DIR / "Data" / "4-DiscretizationData" / discretization_method / period
        / f"{room}-training-s{symbols}.txt"
    )
    TA_output_path = BASE_DIR / "Data" / "5-TaResults" / discretization_method / period
    os.makedirs(TA_output_path, exist_ok=True)
    os.makedirs(training_txt_path.parent, exist_ok=True)

    # Generate
    print("Generating synthetic datasets...")
    all_clean  = generate_trace_set(n_traces=n_train + n_test, noise_std=0.05)
    neg_traces = generate_negative_trace_set(n_traces=20)
    save_traces_as_csv(all_clean,  pos_folder, prefix="clean")
    save_traces_as_csv(neg_traces, neg_folder, prefix="neg")

    # Load and split — train files never appear in evaluation
    pos_files   = get_trace_files(folder_path=pos_folder)
    neg_files   = get_trace_files(folder_path=neg_folder)
    train_files = pos_files[:n_train]
    test_files  = pos_files[n_train:]   # held-out positives
    print(f"Train={len(train_files)} pos | Test={len(test_files)} held-out pos "
          f"| Neg={len(neg_files)}")

    data_lists_train = csv_to_temp_time_list(input_files=train_files)
    data_lists_test  = csv_to_temp_time_list(input_files=test_files)
    data_lists_neg   = csv_to_temp_time_list(input_files=neg_files)

    # Discretize — bins from training data only
    traces_train, bins = equal_width_discretization(data_lists_train, symbols)
    symbolic_train, symbol_map, _ = map_bins_to_symbols(traces_train, symbols, bins)
    format_output(symbolic_train, training_txt_path)

    # Convert test sets using the same training bins
    symbolic_pos_str = preprocess_test_traces(data_lists_test, bins, symbols)
    symbolic_neg_str = preprocess_test_traces(data_lists_neg,  bins, symbols)

    # Train
    print(f"\nTraining TALearner (k={k_val})...")
    learner = TALearner(tss_path=str(training_txt_path), display=False, k=k_val)

    # Debug: check a few traces both timed and untimed to see what's happening
    print("\n--- Debug (timed=True) ---")
    for label, tss in [("pos", symbolic_pos_str[:3]), ("neg", symbolic_neg_str[:3])]:
        for i, ts in enumerate(tss):
            r = learner.ta.inconsistency_nb([ts], timed=True, show=False, p=False)
            print(f"  {label}[{i}]: {'ACCEPTED' if r == 0 else 'REJECTED'}")

    print("\n--- Debug (timed=False, symbol-only) ---")
    for label, tss in [("pos", symbolic_pos_str[:3]), ("neg", symbolic_neg_str[:3])]:
        for i, ts in enumerate(tss):
            r = learner.ta.inconsistency_nb([ts], timed=False, show=False, p=False)
            print(f"  {label}[{i}]: {'ACCEPTED' if r == 0 else 'REJECTED'}")

    # Evaluate
    print("\nEvaluating...")
    metrics = learner.ta.evaluate_classifier(
        positive_tss=symbolic_pos_str,
        negative_tss=symbolic_neg_str,
        timed=True,
        save_path=str(TA_output_path / "metrics.csv"),
        run_id=f"s{symbols}_k{k_val}"
    )

    print("\n" + "=" * 45)
    print(f"  TP={metrics['TP']}  FP={metrics['FP']}  "
          f"TN={metrics['TN']}  FN={metrics['FN']}")
    print(f"  Precision : {metrics['precision']:.4f}")
    print(f"  Recall    : {metrics['recall']:.4f}")
    print(f"  F1        : {metrics['f1']:.4f}")
    print(f"  PAR : {metrics['PAR']:.1f}%  (should be high)")
    print(f"  NAR : {metrics['NAR']:.1f}%  (should be low)")
    print("=" * 45)

    learner.ta.show(title=f"{room}-s{symbols}-k{k_val}",
                    savePng=True, output_path=str(TA_output_path))

    print(f"\nTotal runtime: {time.perf_counter() - total_start:.2f}s")