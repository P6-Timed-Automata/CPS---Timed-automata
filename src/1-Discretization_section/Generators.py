"""
====================
Synthetic temperature trace generation used by all experiments.

Traces follow a sinusoidal 24h diurnal pattern (trough ~4am, peak ~4pm)
with controllable noise, amplitude variation, and phase shift.

Negative traces come in four modes that each violate a different property
the TA would learn:
  0 — Spikes     : sudden temperature jumps (rate violation)
  1 — Shifted    : peak/trough 12h out of phase (temporal violation)
  2 — Stuck      : signal freezes mid-recording (guard violation)
  3 — Offset     : constant +15°C shift (value violation)
"""

from pathlib import Path
import numpy as np

NEG_MODE_NAMES = {0: "spikes", 1: "shifted", 2: "stuck", 3: "offset"}


# ---------------------------------------------------------------------------
# Core generators
# ---------------------------------------------------------------------------

def generate_trace(
        duration_hours  = 24,
        sample_interval = 300,
        base_temp       = 22.0,
        amplitude       = 3.0,
        phase_shift_h   = 0.0,
        noise_std       = 0.1,
        rng             = None,
):
    """Return (times_seconds, temperatures) as numpy arrays."""
    if rng is None:
        rng = np.random.default_rng()
    times  = np.arange(0, duration_hours * 3600, sample_interval, dtype=float)
    hours  = times / 3600
    cycle  = -np.cos(2 * np.pi * (hours - (4.0 + phase_shift_h)) / 24)
    temps  = base_temp + (amplitude / 2) * cycle
    temps += rng.normal(0, noise_std, size=len(times))
    return times, temps


def generate_trace_set(
        n_traces        = 20,
        base_temp       = 22.0,
        amplitude       = 3.0,
        base_temp_std   = 0.0,
        amplitude_std   = 0.2,
        phase_std_h     = 0.25,
        noise_std       = 0.1,
        seed            = 42,
):
    """
    Generate a set of positive (normal) traces with controlled per-trace
    variation. Returns list of (times, temps) pairs.
    """
    rng    = np.random.default_rng(seed)
    traces = []
    for _ in range(n_traces):
        bt = base_temp + rng.normal(0, base_temp_std)
        am = max(0.5, amplitude + rng.normal(0, amplitude_std))
        ph = rng.normal(0, phase_std_h)
        t, v = generate_trace(
            base_temp=bt, amplitude=am, phase_shift_h=ph,
            noise_std=noise_std, rng=rng,
        )
        traces.append((t, v))
    return traces


def generate_negative_set(n_traces: int = 20, seed: int = 99):
    """
    Generate negative (anomalous) traces cycling through 4 failure modes.
    Returns (traces, modes) where modes[i] is the mode index for trace i.
    """
    rng    = np.random.default_rng(seed)
    traces = []
    modes  = []

    for i in range(n_traces):
        mode = i % 4
        t, v = generate_trace(noise_std=0.05, rng=rng)

        if mode == 0:   # spikes
            idx = rng.choice(len(v), size=3, replace=False)
            v[idx] += rng.uniform(5.0, 10.0, size=3)
        elif mode == 1: # shifted 12h
            t, v = generate_trace(phase_shift_h=12.0, noise_std=0.05, rng=rng)
        elif mode == 2: # stuck
            v[40:120] = v[40]
        else:           # offset
            v += 15.0

        traces.append((t, v))
        modes.append(mode)

    return traces, modes


# ---------------------------------------------------------------------------
# CSV I/O
# ---------------------------------------------------------------------------

def save_traces(traces, folder, prefix="trace"):
    """
    Save a list of (times, temps) traces to a single folder.
    Used for positive traces (clean, noisy, test sets).
    """
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    for i, (t, v) in enumerate(traces):
        path = folder / f"{prefix}_tid{i + 1}.csv"
        with open(path, "w") as f:
            f.write("time_s;temperature\n")
            for ti, vi in zip(t, v):
                f.write(f"{int(ti)};{vi:.5f}\n")
    print(f"  Saved {len(traces)} traces -> {folder}")


def save_traces_by_mode(traces, modes, folder, prefix="neg"):
    """
    Save negative traces into mode-specific subfolders so that mode
    information is preserved on disk and survives across experiments.

    Folder structure created:
        folder/
            spikes/   neg_spikes_tid1.csv, neg_spikes_tid2.csv, ...
            shifted/  neg_shifted_tid1.csv, ...
            stuck/    neg_stuck_tid1.csv, ...
            offset/   neg_offset_tid1.csv, ...

    Parameters
    ----------
    traces : list of (times_array, temps_array)
        Output from generate_negative_set().
    modes : list of int
        Parallel list of mode indices (0-3) from generate_negative_set().
    folder : str or Path
        Root folder to save into (e.g. data_root / "negative").
    prefix : str
        Filename prefix before the mode name and index.
    """
    folder        = Path(folder)
    mode_counters = {mode: 0 for mode in NEG_MODE_NAMES}

    for (t, v), mode in zip(traces, modes):
        mode_name   = NEG_MODE_NAMES[mode]
        mode_folder = folder / mode_name
        mode_folder.mkdir(parents=True, exist_ok=True)

        mode_counters[mode] += 1
        filename = f"{prefix}_{mode_name}_tid{mode_counters[mode]}.csv"
        path     = mode_folder / filename

        with open(path, "w") as f:
            f.write("time_s;temperature\n")
            for ti, vi in zip(t, v):
                f.write(f"{int(ti)};{vi:.5f}\n")

    counts = {NEG_MODE_NAMES[m]: c for m, c in mode_counters.items() if c > 0}
    print(f"  Saved {len(traces)} negative traces by mode -> {folder}")
    print(f"  Mode counts: {counts}")


def load_traces(folder):
    """
    Load all CSV traces from a flat folder.
    Returns list of (times, temps) pairs.
    Used for positive traces.
    """
    folder = Path(folder)
    paths  = sorted(folder.glob("*.csv"))
    traces = []
    for p in paths:
        data = np.genfromtxt(p, delimiter=";", skip_header=1)
        traces.append((data[:, 0], data[:, 1]))
    return traces


def load_traces_by_mode(folder):
    """
    Load negative traces from mode-specific subfolders created by
    save_traces_by_mode(). Returns (traces, modes) in the same format
    as generate_negative_set() so the two are interchangeable.

    Parameters
    ----------
    folder : str or Path
        Root negative folder containing mode subfolders.

    Returns
    -------
    traces : list of (times_array, temps_array)
    modes  : list of int  (mode index 0-3 for each trace)
    """
    folder = Path(folder)
    traces = []
    modes  = []

    for mode_int, mode_name in NEG_MODE_NAMES.items():
        mode_folder = folder / mode_name
        if not mode_folder.exists():
            print(f"  Warning: mode subfolder not found: {mode_folder}")
            continue
        for p in sorted(mode_folder.glob("*.csv")):
            data = np.genfromtxt(p, delimiter=";", skip_header=1)
            traces.append((data[:, 0], data[:, 1]))
            modes.append(mode_int)

    if not traces:
        raise FileNotFoundError(
            f"No mode subfolders found under {folder}. "
            f"Expected subfolders: {list(NEG_MODE_NAMES.values())}. "
            f"Make sure save_traces_by_mode() was used when saving."
        )

    counts = {}
    for m in modes:
        counts[NEG_MODE_NAMES[m]] = counts.get(NEG_MODE_NAMES[m], 0) + 1
    print(f"  Loaded {len(traces)} negative traces from {folder}")
    print(f"  Mode counts: {counts}")
    return traces, modes