import numpy as np
import matplotlib.pyplot as plt
import os
from pathlib import Path

# ---------------------------------------------------------------------------
# Synthetic 24h room temperature trace generator
#
# Models a realistic diurnal pattern:
#   - Night low (~3-6h) → morning rise → midday plateau → evening drop
# Each trace gets small per-trace variation in amplitude and phase so the
# synthetic dataset is realistic but controlled.
# ---------------------------------------------------------------------------

def generate_synthetic_trace(
        duration_hours  = 24,
        sample_interval = 300,        # seconds between samples (5 min)
        base_temp       = 22.0,       # mean temperature (°C)
        amplitude       = 3.0,        # peak-to-trough swing (°C)
        phase_shift_h   = 0.0,        # shift the cycle left/right (hours)
        noise_std       = 0.1,        # pointwise gaussian noise (°C)
        rng             = None,
):
    """
    Generates one synthetic 24h temperature trace as a sinusoidal
    diurnal cycle with gaussian noise.

    Returns:
        times : array of timestamps in seconds
        temps : array of temperatures in °C
    """
    if rng is None:
        rng = np.random.default_rng()

    times = np.arange(0, duration_hours * 3600, sample_interval, dtype=float)
    hours = times / 3600

    # Sinusoid: minimum at ~4am, maximum at ~4pm
    # Phase offset so trough is at hour 4 → shift by 4h from standard cosine
    trough_hour = 4.0 + phase_shift_h
    cycle = -np.cos(2 * np.pi * (hours - trough_hour) / 24)  # [-1, 1]
    temps = base_temp + (amplitude / 2) * cycle
    temps += rng.normal(0, noise_std, size=len(times))

    return times, temps


def generate_trace_set(
        n_traces        = 20,
        duration_hours  = 24,
        sample_interval = 300,
        base_temp       = 22.0,
        amplitude       = 3.0,
        # Per-trace variation — controls how much traces differ from each other
        base_temp_std   = 0.0,        # variation in mean level across traces
        amplitude_std   = 0.2,        # variation in swing across traces
        phase_std_h     = 0.25,       # variation in peak/trough timing (hours)
        noise_std       = 0.1,        # pointwise noise per trace
        seed            = 42,
):
    """
    Generates a set of synthetic traces with controlled per-trace variation.

    Args:
        n_traces       : number of traces to generate
        base_temp_std  : std of per-trace mean temperature offset
                         (set > 0 to simulate seasonal/regime variation)
        amplitude_std  : std of per-trace amplitude variation
        phase_std_h    : std of per-trace phase shift in hours
        noise_std      : pointwise gaussian noise added to each sample

    Returns:
        list of (times, temps) pairs
    """
    rng    = np.random.default_rng(seed)
    traces = []

    for _ in range(n_traces):
        bt = base_temp + rng.normal(0, base_temp_std)
        am = max(0.5, amplitude + rng.normal(0, amplitude_std))
        ph = rng.normal(0, phase_std_h)

        t, v = generate_synthetic_trace(
            duration_hours  = duration_hours,
            sample_interval = sample_interval,
            base_temp       = bt,
            amplitude       = am,
            phase_shift_h   = ph,
            noise_std       = noise_std,
            rng             = rng,
        )
        traces.append((t, v))

    return traces


def save_traces_as_csv(traces, output_folder):
    """Save traces as semicolon-delimited CSVs matching your existing format."""
    output_folder = Path(output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)
    for i, (t, v) in enumerate(traces):
        path = output_folder / f"synthetic_tid{i+1}.csv"
        with open(path, 'w') as f:
            f.write("time_s;temperature\n")
            for ti, vi in zip(t, v):
                f.write(f"{int(ti)};{vi:.5f}\n")
    print(f"Saved {len(traces)} traces to {output_folder}")


def plot_trace_sets(clean_traces, noisy_traces, output_path):
    """
    Side-by-side overlay of clean vs noisy synthetic trace sets
    so you can visually confirm the variation levels look right
    before feeding them to TAG.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)

    for ax, traces, title in zip(
        axes,
        [clean_traces, noisy_traces],
        [f"Clean  (n={len(clean_traces)})", f"Noisy  (n={len(noisy_traces)})"]
    ):
        colors = plt.cm.viridis(np.linspace(0.1, 0.9, len(traces)))
        for i, (t, v) in enumerate(traces):
            ax.plot(t / 3600, v, color=colors[i], linewidth=0.7, alpha=0.6)

        # Plot mean
        t_grid   = traces[0][0]
        mean_v   = np.mean([np.interp(t_grid, t, v) for t, v in traces], axis=0)
        ax.plot(t_grid / 3600, mean_v, color='black', linewidth=1.8,
                linestyle='--', label='Mean')

        ax.set_title(title)
        ax.set_xlabel("Time (hours)")
        ax.set_ylabel("Temperature (°C)")
        ax.legend(fontsize=9)
        ax.grid(True, linestyle='--', alpha=0.3)

    fig.suptitle("Synthetic trace sets — clean vs noisy", fontsize=12)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved {output_path}")


# ---------------------------------------------------------------------------
# Main — generate both a clean and a noisy set
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    out_root = Path("../Data/synthetic_data")

    # --- Clean set: small noise, traces are nearly identical ---
    # This is the "optimal" case — TAG should learn a tight TA
    clean_traces = generate_trace_set(
        n_traces       = 20,
        base_temp      = 22.0,
        amplitude      = 3.0,
        base_temp_std  = 0.0,   # no regime variation
        amplitude_std  = 0.1,   # tiny swing variation
        phase_std_h    = 0.1,   # tiny timing variation
        noise_std      = 0.05,  # very low pointwise noise
    )
    save_traces_as_csv(clean_traces, out_root / "clean")

    # --- Noisy set: larger variation, mimics your real data issues ---
    noisy_traces = generate_trace_set(
        n_traces       = 300,
        base_temp      = 22.0,
        amplitude      = 3.0,
        base_temp_std  = 2.0,   # traces from different temperature regimes
        amplitude_std  = 1.0,   # swing varies a lot between days
        phase_std_h    = 1.0,   # peak/trough timing shifts by up to ~1h
        noise_std      = 0.3,   # higher pointwise noise
    )
    save_traces_as_csv(noisy_traces, out_root / "noisy")

    plot_trace_sets(
        clean_traces, noisy_traces,
        output_path = str(out_root / "synthetic_comparison.png")
    )