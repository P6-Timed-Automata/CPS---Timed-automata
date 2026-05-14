import numpy as np
import os
import string
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.interpolate import CubicSpline

from Discretization.naive import equal_width_discretization
from Discretization.persist import Persist, get_best_bins, flatten_traces_to_ts, discretize_traces_with_bins
from Discretization.sax import sax_discretization_multi


def make_mapping(k):
    return {i: string.ascii_lowercase[i] for i in range(k)}


# --- 3 trace definitions ---
# Each is a list of (time_seconds, temperature) anchors
TRACES = {
    "tid1": np.array([
        (0,     21.78),   # Normal day — matches real trace closely
        (27000, 21.05),
        (29400, 21.80),
        (61200, 23.60),
        (86100, 21.85),
    ]),
    "tid2": np.array([
        (0,     21.60),   # Slightly cooler day — heating kicks in earlier
        (25200, 20.90),
        (27600, 21.65),
        (59400, 23.30),
        (86100, 21.60),
    ]),
    "tid3": np.array([
        (0,     22.00),   # Slightly warmer day — heating kicks in later
        (28800, 21.30),
        (31200, 22.10),
        (63000, 23.90),
        (86100, 22.10),
    ]),
}


def generate_idealized_trace(anchors, output_path, sampling_interval=300):
    cs           = CubicSpline(anchors[:, 0], anchors[:, 1])
    timestamps   = np.arange(0, 86100 + sampling_interval, sampling_interval)
    temperatures = cs(timestamps)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    result = np.column_stack((timestamps, temperatures))
    np.savetxt(
        output_path,
        result,
        delimiter=';',
        header='time_seconds;temperature',
        fmt=['%.0f', '%.5f'],
        comments=''
    )
    print(f"Saved {output_path} ({len(result)} rows)")
    return timestamps, temperatures


def plot_raw(timestamps, temperatures, title, output_path):
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(timestamps / 3600, temperatures, linewidth=1.5, color='steelblue')
    ax.set_title(title)
    ax.set_xlabel("Time (hours)")
    ax.set_ylabel("Temperature (°C)")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_all_raw(all_traces, output_path):
    """Overlay of all 3 raw traces for easy comparison."""
    fig, ax = plt.subplots(figsize=(12, 5))
    colors = ['steelblue', 'tomato', 'seagreen']
    for (tid, (timestamps, temperatures)), color in zip(all_traces.items(), colors):
        ax.plot(timestamps / 3600, temperatures, linewidth=1.5, label=tid, color=color)
    ax.set_title("All Idealized Traces (Raw)")
    ax.set_xlabel("Time (hours)")
    ax.set_ylabel("Temperature (°C)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"Saved overlay to {output_path}")


def plot_discretized(timestamps, labels, mapping, title, output_path):
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.step(timestamps / 3600, labels, where='post', linewidth=1.5, color='steelblue')
    ax.set_title(title)
    ax.set_xlabel("Time (hours)")
    ax.set_ylabel("Symbol")
    ax.set_yticks(list(mapping.keys()))
    ax.set_yticklabels(list(mapping.values()))
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"Saved discretized plot to {output_path}")


if __name__ == "__main__":
    BASE_DIR  = Path(__file__).resolve().parent.parent
    OUT_DATA  = BASE_DIR / "Data" / "3-ExtractInterval" / "idealized"
    OUT_GRAPH = BASE_DIR / "Data" / "Graphs" / "idealized"

    symbols = 11
    w       = 200

    # --- Generate all traces and collect for overlay ---
    raw_results = {}
    data_lists  = []

    for tid, anchors in TRACES.items():
        timestamps, temperatures = generate_idealized_trace(
            anchors=anchors,
            output_path=str(OUT_DATA / f"roomA-idealized-{tid}.csv")
        )
        raw_results[tid] = (timestamps, temperatures)
        data_lists.append([(float(t), int(s)) for t, s in zip(temperatures, timestamps)])

        # Individual raw plot
        plot_raw(
            timestamps=timestamps,
            temperatures=temperatures,
            title=f"Idealized Trace {tid} (Raw)",
            output_path=str(OUT_GRAPH / "raw" / f"idealized_{tid}_raw.png")
        )

    # Overlay of all raw traces
    plot_all_raw(raw_results, output_path=str(OUT_GRAPH / "raw" / "idealized_overlay_raw.png"))

    # --- Discretize and plot each trace individually ---
    for i, (tid, trace) in enumerate(zip(TRACES.keys(), data_lists)):
        single = [trace]
        timestamps = np.array([t for _, t in trace])

        # Naive
        naive_traces, _ = equal_width_discretization(single, symbols)
        plot_discretized(
            timestamps=np.array([t for _, t in naive_traces[0]]),
            labels=np.array([l for l, _ in naive_traces[0]]),
            mapping=make_mapping(symbols),
            title=f"Idealized {tid} — Naive (k={symbols})",
            output_path=str(OUT_GRAPH / "naive" / f"idealized_{tid}_naive.png")
        )

        # Persist
        ts   = flatten_traces_to_ts(single)
        p    = Persist(x=ts, break_min=2, break_max=10, divergence="w", candidates="EW", skip=np.array([4, 4]))
        bins = get_best_bins(p, ts)
        persist_traces  = discretize_traces_with_bins(single, bins)
        persist_symbols = len(bins) - 1
        plot_discretized(
            timestamps=np.array([t for _, t in persist_traces[0]]),
            labels=np.array([l for l, _ in persist_traces[0]]),
            mapping=make_mapping(persist_symbols),
            title=f"Idealized {tid} — Persist (k={persist_symbols})",
            output_path=str(OUT_GRAPH / "persist" / f"idealized_{tid}_persist.png")
        )

        # SAX
        sax_traces, _ = sax_discretization_multi(single, w=w, k=symbols)
        plot_discretized(
            timestamps=np.array([t for _, t in sax_traces[0]]),
            labels=np.array([l for l, _ in sax_traces[0]]),
            mapping=make_mapping(symbols),
            title=f"Idealized {tid} — SAX (w={w}, k={symbols})",
            output_path=str(OUT_GRAPH / "sax" / f"idealized_{tid}_sax.png")
        )

        print(f"Done: {tid}")