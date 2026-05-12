import numpy as np
from pathlib import Path



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



if __name__ == "__main__":
    BASE_DIR = Path(__file__).resolve().parent.parent.parent

    n_train = 300
    n_test = 100
    n_neg = 100

    out_root = BASE_DIR / "Data" / "synthetic_data"

    all_clean = generate_trace_set(
        n_traces=n_train + n_test,
        seed=42,
        noise_std=0.05
    )

    all_noisy = generate_trace_set(
        n_traces=n_train + n_test,
        seed=42,
        noise_std=0.3,
        base_temp_std=2.0,
        amplitude_std=1.0,
        phase_std_h=1.0
    )

    neg_traces = generate_negative_trace_set(
        n_traces=n_neg,
        seed=99
    )

    clean_train, clean_test = all_clean[:n_train], all_clean[n_train:]
    noisy_train, noisy_test = all_noisy[:n_train], all_noisy[n_train:]

    save_traces_as_csv(clean_train, out_root / "clean_train", prefix="clean_train")
    save_traces_as_csv(clean_test,  out_root / "clean_test",  prefix="clean_test")

    save_traces_as_csv(noisy_train, out_root / "noisy_train", prefix="noisy_train")
    save_traces_as_csv(noisy_test,  out_root / "noisy_test",  prefix="noisy_test")

    save_traces_as_csv(neg_traces, out_root / "negative", prefix="neg")

    print("Synthetic datasets generated.")