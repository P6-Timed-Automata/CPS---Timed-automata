import numpy as np
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------

def load_trace(path):
    """Load a semicolon-delimited time_seconds;temperature CSV."""
    data = np.genfromtxt(path, delimiter=';', skip_header=1)
    return data[:, 0], data[:, 1]


def load_reference(path, temp_scale=0.01):
    """
    Load a UPPAAL simulation output file representing a step function.

    Format:
        #### Simulations (N)
        # var #1
        t0,v0
        t1,v0      <- same time as next row = instantaneous jump
        t1,v1
        ...

    Args:
        temp_scale : multiply raw values by this to get °C (default 0.01: 2093 -> 20.93)
    """
    times, values = [], []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            parts = line.split(',')
            if len(parts) != 2:
                continue
            try:
                times.append(float(parts[0]))
                values.append(float(parts[1]) * temp_scale)
            except ValueError:
                continue
    return np.array(times), np.array(values)


# ---------------------------------------------------------------------------
# Step-function helpers
# ---------------------------------------------------------------------------

def eval_step_function(t_query, t_steps, v_steps):
    """
    Evaluate a step function at arbitrary query times (left-continuous).
    Handles duplicate timestamps (UPPAAL-style jump edges) correctly.
    """
    idx = np.searchsorted(t_steps, t_query, side='right') - 1
    idx = np.clip(idx, 0, len(v_steps) - 1)
    return v_steps[idx]


def discretized_to_step(discretized_trace, bins_celsius):
    """
    Convert a discretized trace to a step-function signal in °C.

    Args:
        discretized_trace : list of (bin_index, time_seconds) tuples
        bins_celsius      : bin edges in °C (length k+1); bin i maps to
                            the centre of [bins[i], bins[i+1]]

    Returns:
        t (1D array), v (1D array)
    """
    bin_centers = (bins_celsius[:-1] + bins_celsius[1:]) / 2
    times  = np.array([t for _, t in discretized_trace], dtype=float)
    labels = np.array([l for l, _ in discretized_trace], dtype=int)
    labels = np.clip(labels, 0, len(bin_centers) - 1)
    return times, bin_centers[labels]


# ---------------------------------------------------------------------------
# Per-method bin-edge converters
# ---------------------------------------------------------------------------

def naive_bins_celsius(bins):
    """
    Naive / Persist: bins are already in °C — pass through unchanged.
    """
    return bins


def sax_bins_celsius(breakpoints, original_trace_values, outer_z=3.5):
    """
    Convert SAX z-space Gaussian breakpoints to °C bin edges.

    SAX z-normalises each trace before assigning labels, so breakpoints
    live in z-space.  To reconstruct temperatures we invert the z-norm:
        temp = z * std + mean

    Args:
        breakpoints           : 1-D array of k-1 Gaussian breakpoints (z-space)
        original_trace_values : raw °C values of the trace that was discretized
        outer_z               : z-score used as ±outer boundary (default 3.5)

    Returns:
        bin edges in °C (length k+1)
    """
    v = np.asarray(original_trace_values, dtype=float)
    mean, std = v.mean(), v.std()
    if std == 0:
        std = 1.0  # degenerate trace — avoid division by zero

    z_edges = np.concatenate([[-outer_z], np.sort(breakpoints), [outer_z]])
    return z_edges * std + mean


# ---------------------------------------------------------------------------
# RMSE
# ---------------------------------------------------------------------------

def compute_rmse(t_signal, v_signal, t_ref, v_ref):
    """
    RMSE of a signal vs the reference, evaluated over the overlap window
    at the signal's own time points.
    """
    t_start = max(t_signal[0], t_ref[0])
    t_end   = min(t_signal[-1], t_ref[-1])
    if t_start >= t_end:
        return np.nan

    mask           = (t_signal >= t_start) & (t_signal <= t_end)
    t_common       = t_signal[mask]
    v_signal_clip  = v_signal[mask]
    v_ref_at_signal = eval_step_function(t_common, t_ref, v_ref)
    return float(np.sqrt(np.mean((v_signal_clip - v_ref_at_signal) ** 2)))


# ---------------------------------------------------------------------------
# Main comparison function
# ---------------------------------------------------------------------------

def compare_trace_to_reference(
        trace_csv,
        reference_csv,
        output_path=None,
        temp_scale=0.01,
        discretized_traces=None,
):
    """
    Compare a continuous trace and optional discretized variants against a
    UPPAAL step-function reference.  Produces an overlay plot plus one
    residual panel per signal.

    Args:
        trace_csv          : semicolon-delimited time_seconds;temperature CSV
        reference_csv      : UPPAAL simulation output (#### header, comma t,v)
        output_path        : save path for plot (None = interactive show)
        temp_scale         : scale factor for reference raw values -> °C
        discretized_traces : dict of {label: (disc_trace, bins_celsius)} where
                               disc_trace   — list of (bin_index, time_seconds)
                               bins_celsius — bin edges already in °C
                             Use the helper converters above:
                               naive/persist -> naive_bins_celsius(bins)
                               sax           -> sax_bins_celsius(breakpoints, raw_values)

    Returns:
        dict {signal_label: rmse}
    """
    t_trace, v_trace = load_trace(trace_csv)
    t_ref,   v_ref   = load_reference(reference_csv, temp_scale)

    disc_signals = {}  # label -> (t, v) ready to plot
    if discretized_traces:
        for label, (disc_trace, bins_celsius) in discretized_traces.items():
            t_d, v_d = discretized_to_step(disc_trace, bins_celsius)
            disc_signals[label] = (t_d, v_d)

    # RMSE set 1: each discretization vs raw trace
    rmse_vs_raw = {}
    for label, (t_d, v_d) in disc_signals.items():
        v_raw_interp = np.interp(t_d, t_trace, v_trace)
        rmse_vs_raw[label] = float(np.sqrt(np.mean((v_d - v_raw_interp) ** 2)))

    # RMSE set 2: raw trace + each discretization vs UPPAAL reference
    rmse_vs_uppaal = {'Raw trace': compute_rmse(t_trace, v_trace, t_ref, v_ref)}
    for label, (t_d, v_d) in disc_signals.items():
        rmse_vs_uppaal[label] = compute_rmse(t_d, v_d, t_ref, v_ref)

    # Keep a combined dict for the return value
    rmse_results = {**rmse_vs_uppaal, **{f"{k} (vs raw)": v for k, v in rmse_vs_raw.items()}}

    colors = plt.cm.tab10(np.linspace(0, 0.7, max(len(disc_signals), 1)))

    # --- Figure 1: raw trace vs discretizations ---
    fig1, (ax1_top, ax1_bot) = plt.subplots(
        2, 1, figsize=(13, 7), sharex=True,
        gridspec_kw={'height_ratios': [3, 1]},
    )
    ax1_top.plot(t_trace / 3600, v_trace, color='steelblue', linewidth=0.9,
                 alpha=0.85, label='Raw trace')
    for i, (label, (t_d, v_d)) in enumerate(disc_signals.items()):
        ax1_top.step(t_d / 3600, v_d, where='post',
                     color=colors[i], linewidth=1.0, alpha=0.85,
                     label=f"{label}  RMSE={rmse_vs_raw[label]:.4f} °C")
    ax1_top.set_ylabel("Temperature (°C)")
    ax1_top.set_title("Discretizations vs raw trace")
    ax1_top.legend(fontsize=8, loc='lower right', framealpha=0.9)
    ax1_top.grid(True, alpha=0.3)

    # Residuals vs raw trace (interpolated onto disc time grid)
    ax1_bot.axhline(0, color='black', linewidth=0.8, linestyle='--')
    for i, (label, (t_d, v_d)) in enumerate(disc_signals.items()):
        v_raw_at_disc = np.interp(t_d, t_trace, v_trace)
        res = v_d - v_raw_at_disc
        ax1_bot.plot(t_d / 3600, res, color=colors[i], linewidth=0.8, label=label)
        ax1_bot.fill_between(t_d / 3600, res, alpha=0.15, color=colors[i])
    ax1_bot.set_ylabel("Residual (°C)")
    ax1_bot.set_xlabel("Time (hours)")
    ax1_bot.legend(fontsize=7, loc='lower right', framealpha=0.9)
    ax1_bot.grid(True, alpha=0.3)
    fig1.tight_layout()

    # --- Figure 2: UPPAAL reference vs raw trace + discretizations ---
    fig2, (ax2_top, ax2_bot) = plt.subplots(
        2, 1, figsize=(13, 7), sharex=True,
        gridspec_kw={'height_ratios': [3, 1]},
    )
    ax2_top.step(t_ref / 3600, v_ref, where='post',
                 color='black', linewidth=1.4, zorder=10, label='Reference (UPPAAL)')
    ax2_top.plot(t_trace / 3600, v_trace, color='steelblue', linewidth=0.9,
                 alpha=0.85,
                 label=f"Raw trace  RMSE={rmse_vs_uppaal['Raw trace']:.4f} °C")
    for i, (label, (t_d, v_d)) in enumerate(disc_signals.items()):
        ax2_top.step(t_d / 3600, v_d, where='post',
                     color=colors[i], linewidth=1.0, alpha=0.85,
                     label=f"{label}  RMSE={rmse_vs_uppaal[label]:.4f} °C")
    ax2_top.set_ylabel("Temperature (°C)")
    ax2_top.set_title("UPPAAL reference vs raw trace and discretizations")
    ax2_top.legend(fontsize=8, loc='upper right', framealpha=0.9)
    ax2_top.grid(True, alpha=0.3)

    # Residuals vs UPPAAL reference
    ax2_bot.axhline(0, color='black', linewidth=0.8, linestyle='--')
    v_raw_res = v_trace - eval_step_function(t_trace, t_ref, v_ref)
    ax2_bot.plot(t_trace / 3600, v_raw_res, color='steelblue', linewidth=0.8, label='Raw trace')
    ax2_bot.fill_between(t_trace / 3600, v_raw_res, alpha=0.15, color='steelblue')
    for i, (label, (t_d, v_d)) in enumerate(disc_signals.items()):
        res = v_d - eval_step_function(t_d, t_ref, v_ref)
        ax2_bot.plot(t_d / 3600, res, color=colors[i], linewidth=0.8, label=label)
        ax2_bot.fill_between(t_d / 3600, res, alpha=0.15, color=colors[i])
    ax2_bot.set_ylabel("Residual (°C)")
    ax2_bot.set_xlabel("Time (hours)")
    ax2_bot.legend(fontsize=7, loc='lower right', framealpha=0.9)
    ax2_bot.grid(True, alpha=0.3)
    fig2.tight_layout()

    # --- Figure 3: UPPAAL reference alone ---
    fig3, ax3 = plt.subplots(figsize=(13, 4))
    ax3.step(t_ref / 3600, v_ref, where='post', color='black', linewidth=1.4)
    ax3.set_ylabel("Temperature (°C)")
    ax3.set_xlabel("Time (hours)")
    ax3.set_title("UPPAAL reference")
    ax3.grid(True, alpha=0.3)
    fig3.tight_layout()

    if output_path:
        base, ext = output_path.rsplit('.', 1)
        path1 = f"{base}_discretizations.{ext}"
        path2 = f"{base}_uppaal.{ext}"
        path3 = f"{base}_reference.{ext}"
        fig1.savefig(path1, dpi=150)
        fig2.savefig(path2, dpi=150)
        fig3.savefig(path3, dpi=150)
        print(f"Saved to {path1}")
        print(f"Saved to {path2}")
        print(f"Saved to {path3}")
        plt.close(fig1)
        plt.close(fig2)
        plt.close(fig3)
    else:
        plt.show()

    print("\nRMSE vs raw trace:")
    for label, rmse in rmse_vs_raw.items():
        print(f"  {label:30s}  {rmse:.4f} °C")
    print("\nRMSE vs UPPAAL reference:")
    for label, rmse in rmse_vs_uppaal.items():
        print(f"  {label:30s}  {rmse:.4f} °C")

    return rmse_results


# ---------------------------------------------------------------------------
# Usage
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from Discretization.discretizationSetup import csv_to_temp_time_list
    from Discretization.naive import equal_width_discretization
    from Discretization.sax import sax_discretization_multi
    from Discretization.persist import Persist, get_best_bins, discretize_traces_with_bins, flatten_traces_to_ts

    trace_path = "../../data/3-ExtractInterval/1day-experiment/roomA/roomA-1day-tid6.csv"
    data_lists = csv_to_temp_time_list(input_files=[trace_path])
    raw_values = np.array([v for v, _ in data_lists[0]])

    # --- Naive ---
    naive_traces, naive_bins = equal_width_discretization(data_lists, k=5)

    # --- SAX ---
    sax_traces, sax_breakpoints = sax_discretization_multi(data_lists, w=20, k=5)

    # --- Persist ---
    ts = flatten_traces_to_ts(data_lists)
    persist_obj  = Persist(ts, break_min=5, break_max=5)
    persist_bins = get_best_bins(persist_obj, ts)
    persist_traces = discretize_traces_with_bins(data_lists, persist_bins)

    rmse_results = compare_trace_to_reference(
        trace_csv     = trace_path,
        reference_csv = "../../Data/7-ExtractedUppaalGraphData/rmseTest.csv",
        output_path   = "comparison.png",
        temp_scale    = 0.01,
        discretized_traces = {
            'naive  (k=5)':   (naive_traces[0],   naive_bins_celsius(naive_bins)),
            'sax    (w=20,k=5)': (sax_traces[0],  sax_bins_celsius(sax_breakpoints, raw_values)),
            'persist (k=5)':  (persist_traces[0], naive_bins_celsius(persist_bins)),
        }
    )