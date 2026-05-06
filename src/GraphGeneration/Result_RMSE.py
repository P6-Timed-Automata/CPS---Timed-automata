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
    Parse a UPPAAL simulation file containing one or more step-function traces.

    Returns:
        list of (t, v) array pairs, one per simulation
    """
    simulations = []
    current_t, current_v = [], []

    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith('####'):
                continue
            if line.startswith('#'):
                # New simulation block — save previous if any
                if current_t:
                    simulations.append((np.array(current_t), np.array(current_v)))
                current_t, current_v = [], []
                continue
            parts = line.split(',')
            if len(parts) != 2:
                continue
            try:
                current_t.append(float(parts[0]))
                current_v.append(float(parts[1]) * temp_scale)
            except ValueError:
                continue

    if current_t:
        simulations.append((np.array(current_t), np.array(current_v)))

    print(f"Loaded {len(simulations)} UPPAAL simulations")
    return simulations


# ---------------------------------------------------------------------------
# Step-function helpers
# ---------------------------------------------------------------------------

def eval_step_function(t_query, t_steps, v_steps):
    """Evaluate a step function at arbitrary query times (left-continuous)."""
    idx = np.searchsorted(t_steps, t_query, side='right') - 1
    idx = np.clip(idx, 0, len(v_steps) - 1)
    return v_steps[idx]


def build_common_grid(simulations, n_points=1000):
    """
    Build a dense common time grid spanning the overlap of all simulations.
    """
    t_start = max(t[0]  for t, _ in simulations)
    t_end   = min(t[-1] for t, _ in simulations)
    return np.linspace(t_start, t_end, n_points)


def evaluate_all_simulations(simulations, t_grid):
    """
    Evaluate every simulation on t_grid.

    Returns:
        matrix of shape (n_simulations, n_points)
    """
    return np.array([
        eval_step_function(t_grid, t, v)
        for t, v in simulations
    ])


def discretized_to_step(discretized_trace, bins_celsius):
    """Convert a discretized trace to a step-function signal in °C."""
    bin_centers = (bins_celsius[:-1] + bins_celsius[1:]) / 2
    times  = np.array([t for _, t in discretized_trace], dtype=float)
    labels = np.array([l for l, _ in discretized_trace], dtype=int)
    labels = np.clip(labels, 0, len(bin_centers) - 1)
    return times, bin_centers[labels]


# ---------------------------------------------------------------------------
# Bin-edge converters
# ---------------------------------------------------------------------------

def naive_bins_celsius(bins):
    """Naive / Persist bins are already in °C."""
    return bins


def sax_bins_celsius(breakpoints, original_trace_values, outer_z=3.5):
    """Convert SAX z-space breakpoints to °C using the original trace stats."""
    v    = np.asarray(original_trace_values, dtype=float)
    mean, std = v.mean(), v.std()
    if std == 0:
        std = 1.0
    z_edges = np.concatenate([[-outer_z], np.sort(breakpoints), [outer_z]])
    return z_edges * std + mean


# ---------------------------------------------------------------------------
# RMSE and MAE helpers
# ---------------------------------------------------------------------------

def rmse_signal_vs_grid(t_signal, v_signal, sim_matrix, t_grid):
    """RMSE of a signal against every simulation row. Returns 1D array (n_sims,)."""
    v_interp  = np.interp(t_grid, t_signal, v_signal)
    residuals = sim_matrix - v_interp[np.newaxis, :]
    return np.sqrt(np.mean(residuals ** 2, axis=1))


def mae_signal_vs_grid(t_signal, v_signal, sim_matrix, t_grid):
    """MAE of a signal against every simulation row. Returns 1D array (n_sims,)."""
    v_interp  = np.interp(t_grid, t_signal, v_signal)
    residuals = sim_matrix - v_interp[np.newaxis, :]
    return np.mean(np.abs(residuals), axis=1)


def rmse_signal_vs_raw(t_signal, v_signal, t_raw, v_raw):
    """RMSE of a discretized signal vs the raw trace (interpolated)."""
    v_raw_interp = np.interp(t_signal, t_raw, v_raw)
    return float(np.sqrt(np.mean((v_signal - v_raw_interp) ** 2)))


def mae_signal_vs_raw(t_signal, v_signal, t_raw, v_raw):
    """MAE of a discretized signal vs the raw trace (interpolated)."""
    v_raw_interp = np.interp(t_signal, t_raw, v_raw)
    return float(np.mean(np.abs(v_signal - v_raw_interp)))


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------

def _plot_sim_band(ax, t_grid, sim_matrix, n_std=1, alpha_band=0.25):
    """Shade mean ± n_std of all simulations."""
    mean = sim_matrix.mean(axis=0)
    std  = sim_matrix.std(axis=0)
    ax.plot(t_grid / 3600, mean, color='black', linewidth=1.4,
            label='UPPAAL mean')
    ax.fill_between(t_grid / 3600, mean - n_std * std, mean + n_std * std,
                    color='black', alpha=alpha_band,
                    label=f'UPPAAL ±{n_std} std (~{int(n_std*68)}% of sims)')


# ---------------------------------------------------------------------------
# Main comparison function
# ---------------------------------------------------------------------------

def compare_trace_to_reference(
    trace_csv,
    reference_csv,
    output_path=None,
    temp_scale=0.01,
    discretized_traces=None,
    n_grid=2000,
    n_std=1,
):
    """
    Compare a continuous trace and optional discretized variants against
    multiple UPPAAL simulations.  Produces four figures:

      1. _discretizations : discretizations vs raw trace  (RMSE vs raw)
      2. _uppaal_band     : mean ± std band + raw trace + discretizations
      3. _rmse_dist       : RMSE distribution (violin) per signal vs simulations
      4. _reference       : UPPAAL simulations only (all + mean ± std)

    Args:
        trace_csv          : semicolon-delimited time_seconds;temperature CSV
        reference_csv      : UPPAAL multi-simulation file
        output_path        : base save path, e.g. "comparison.png"
        temp_scale         : raw UPPAAL value -> °C scale factor
        discretized_traces : dict {label: (disc_trace, bins_celsius)}
        n_grid             : resolution of the common evaluation grid
    """
    t_trace, v_trace = load_trace(trace_csv)
    simulations      = load_reference(reference_csv, temp_scale)

    t_grid     = build_common_grid(simulations, n_points=n_grid)
    sim_matrix = evaluate_all_simulations(simulations, t_grid)  # (n_sims, n_grid)
    sim_mean   = sim_matrix.mean(axis=0)
    sim_std    = sim_matrix.std(axis=0)

    # Build disc signals
    disc_signals = {}
    if discretized_traces:
        for label, (disc_trace, bins_celsius) in discretized_traces.items():
            disc_signals[label] = discretized_to_step(disc_trace, bins_celsius)

    colors = plt.cm.tab10(np.linspace(0, 0.7, max(len(disc_signals), 1)))

    # --- RMSE and MAE vs raw trace (for figure 1) ---
    rmse_vs_raw = {
        label: rmse_signal_vs_raw(t_d, v_d, t_trace, v_trace)
        for label, (t_d, v_d) in disc_signals.items()
    }
    mae_vs_raw = {
        label: mae_signal_vs_raw(t_d, v_d, t_trace, v_trace)
        for label, (t_d, v_d) in disc_signals.items()
    }

    # --- RMSE and MAE distributions vs UPPAAL simulations (for figures 2 & 3) ---
    rmse_dist = {'Raw trace': rmse_signal_vs_grid(t_trace, v_trace, sim_matrix, t_grid)}
    mae_dist  = {'Raw trace': mae_signal_vs_grid(t_trace, v_trace, sim_matrix, t_grid)}
    for label, (t_d, v_d) in disc_signals.items():
        rmse_dist[label] = rmse_signal_vs_grid(t_d, v_d, sim_matrix, t_grid)
        mae_dist[label]  = mae_signal_vs_grid(t_d, v_d, sim_matrix, t_grid)

    # -----------------------------------------------------------------------
    # Figure 1: discretizations vs raw trace
    # -----------------------------------------------------------------------
    fig1, (ax1t, ax1b) = plt.subplots(2, 1, figsize=(13, 7), sharex=True,
                                       gridspec_kw={'height_ratios': [3, 1]})
    ax1t.plot(t_trace / 3600, v_trace, color='steelblue', linewidth=0.9,
              alpha=0.85, label='Raw trace')
    for i, (label, (t_d, v_d)) in enumerate(disc_signals.items()):
        ax1t.step(t_d / 3600, v_d, where='post', color=colors[i],
                  linewidth=1.0, alpha=0.85,
                  label=f"{label}  RMSE={rmse_vs_raw[label]:.4f}  MAE={mae_vs_raw[label]:.4f} °C")
    ax1t.set_ylabel("Temperature (°C)")
    ax1t.set_title("Discretizations vs raw trace")
    ax1t.legend(fontsize=8, loc='lower right', framealpha=0.9)
    ax1t.grid(True, alpha=0.3)

    ax1b.axhline(0, color='black', linewidth=0.8, linestyle='--')
    for i, (label, (t_d, v_d)) in enumerate(disc_signals.items()):
        res = v_d - np.interp(t_d, t_trace, v_trace)
        ax1b.plot(t_d / 3600, res, color=colors[i], linewidth=0.8, label=label)
        ax1b.fill_between(t_d / 3600, res, alpha=0.15, color=colors[i])
    ax1b.set_ylabel("Residual (°C)")
    ax1b.set_xlabel("Time (hours)")
    ax1b.legend(fontsize=7, loc='lower right', framealpha=0.9)
    ax1b.grid(True, alpha=0.3)
    fig1.tight_layout()

    # -----------------------------------------------------------------------
    # Figure 2: UPPAAL mean ± std band + raw trace + discretizations
    # -----------------------------------------------------------------------
    fig2, (ax2t, ax2b) = plt.subplots(2, 1, figsize=(13, 7), sharex=True,
                                       gridspec_kw={'height_ratios': [3, 1]})
    _plot_sim_band(ax2t, t_grid, sim_matrix, n_std=n_std)
    rmse_raw = rmse_dist['Raw trace']
    mae_raw  = mae_dist['Raw trace']
    ax2t.plot(t_trace / 3600, v_trace, color='steelblue', linewidth=0.9, alpha=0.85,
              label=f"Raw trace  RMSE={rmse_raw.mean():.4f}  MAE={mae_raw.mean():.4f} °C")
    for i, (label, (t_d, v_d)) in enumerate(disc_signals.items()):
        rd = rmse_dist[label]
        md = mae_dist[label]
        ax2t.step(t_d / 3600, v_d, where='post', color=colors[i],
                  linewidth=1.0, alpha=0.85,
                  label=f"{label}  RMSE={rd.mean():.4f}  MAE={md.mean():.4f} °C")
    ax2t.set_ylabel("Temperature (°C)")
    ax2t.set_title("UPPAAL band vs raw trace and discretizations")
    ax2t.legend(fontsize=8, loc='lower right', framealpha=0.9)
    ax2t.grid(True, alpha=0.3)

    # Residuals vs UPPAAL mean
    ax2b.axhline(0, color='black', linewidth=0.8, linestyle='--')
    res_raw = np.interp(t_grid, t_trace, v_trace) - sim_mean
    ax2b.plot(t_grid / 3600, res_raw, color='steelblue', linewidth=0.8, label='Raw trace')
    ax2b.fill_between(t_grid / 3600, res_raw, alpha=0.15, color='steelblue')
    for i, (label, (t_d, v_d)) in enumerate(disc_signals.items()):
        res = np.interp(t_grid, t_d, v_d) - sim_mean
        ax2b.plot(t_grid / 3600, res, color=colors[i], linewidth=0.8, label=label)
        ax2b.fill_between(t_grid / 3600, res, alpha=0.15, color=colors[i])
    ax2b.set_ylabel("Residual vs mean (°C)")
    ax2b.set_xlabel("Time (hours)")
    ax2b.legend(fontsize=7, loc='lower right', framealpha=0.9)
    ax2b.grid(True, alpha=0.3)
    fig2.tight_layout()

    # -----------------------------------------------------------------------
    # Figure 3: RMSE and MAE distributions — side-by-side violin plots
    # -----------------------------------------------------------------------
    labels_order = list(rmse_dist.keys())
    positions    = range(len(labels_order))
    signal_colors = [
        'steelblue' if l == 'Raw trace'
        else colors[list(disc_signals.keys()).index(l)]
        for l in labels_order
    ]

    fig3, (ax3l, ax3r) = plt.subplots(1, 2, figsize=(14, 5), sharey=False)

    for ax, dist, metric_label in [
        (ax3l, rmse_dist, 'RMSE'),
        (ax3r, mae_dist,  'MAE'),
    ]:
        data = [dist[l] for l in labels_order]
        vp   = ax.violinplot(data, positions=list(positions),
                             showmedians=True, showextrema=True)
        for body, c in zip(vp['bodies'], signal_colors):
            body.set_facecolor(c)
            body.set_alpha(0.6)
        ax.set_xticks(list(positions))
        ax.set_xticklabels(labels_order, fontsize=9)
        ax.set_ylabel(f"{metric_label} vs UPPAAL simulations (°C)")
        ax.set_title(f"{metric_label} distribution across UPPAAL simulations")
        ax.grid(True, alpha=0.3, axis='y')

    fig3.tight_layout()

    # -----------------------------------------------------------------------
    # Figure 4: UPPAAL simulations only
    # -----------------------------------------------------------------------
    fig4, ax4 = plt.subplots(figsize=(13, 4))
    for t_s, v_s in simulations:
        ax4.step(t_s / 3600, v_s, where='post', color='grey',
                 linewidth=0.4, alpha=0.3)
    _plot_sim_band(ax4, t_grid, sim_matrix, n_std=n_std)
    ax4.set_ylabel("Temperature (°C)")
    ax4.set_xlabel("Time (hours)")
    ax4.set_title(f"UPPAAL reference — {len(simulations)} simulations")
    ax4.legend(fontsize=8, loc='lower right', framealpha=0.9)
    ax4.grid(True, alpha=0.3)
    fig4.tight_layout()

    # -----------------------------------------------------------------------
    # Save / show
    # -----------------------------------------------------------------------
    if output_path:
        base, ext = output_path.rsplit('.', 1)
        paths = {
            'discretizations': fig1,
            'uppaal_band':     fig2,
            'rmse_dist':       fig3,
            'reference':       fig4,
        }
        for suffix, fig in paths.items():
            p = f"{base}_{suffix}.{ext}"
            fig.savefig(p, dpi=150)
            print(f"Saved {p}")
            plt.close(fig)
    else:
        plt.show()

    print("\nRMSE / MAE vs raw trace:")
    for label in rmse_vs_raw:
        print(f"  {label:30s}  RMSE={rmse_vs_raw[label]:.4f}  MAE={mae_vs_raw[label]:.4f} °C")
    print("\nRMSE vs UPPAAL simulations (mean / std / min):")
    for label, arr in rmse_dist.items():
        print(f"  {label:30s}  mean={arr.mean():.4f}  std={arr.std():.4f}  min={arr.min():.4f} °C")
    print("\nMAE vs UPPAAL simulations (mean / std / min):")
    for label, arr in mae_dist.items():
        print(f"  {label:30s}  mean={arr.mean():.4f}  std={arr.std():.4f}  min={arr.min():.4f} °C")

    return rmse_vs_raw, mae_vs_raw, rmse_dist, mae_dist


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

    naive_traces, naive_bins     = equal_width_discretization(data_lists, k=5)
    sax_traces, sax_breakpoints  = sax_discretization_multi(data_lists, w=20, k=5)
    ts           = flatten_traces_to_ts(data_lists)
    persist_obj  = Persist(ts, break_min=3, break_max=15)
    persist_bins = get_best_bins(persist_obj, ts)
    persist_traces = discretize_traces_with_bins(data_lists, persist_bins)

    compare_trace_to_reference(
        trace_csv     = trace_path,
        reference_csv = "../../Data/7-ExtractedUppaalGraphData/test2-multiple.csv",
        output_path   = "comparison.png",
        temp_scale    = 0.01,
        n_std         = 2,   # 1 = ~68% of sims, 2 = ~95%, 3 = ~99.7%
        discretized_traces = {
            'naive  (k=5)':      (naive_traces[0],   naive_bins_celsius(naive_bins)),
            'sax    (w=20,k=5)': (sax_traces[0],     sax_bins_celsius(sax_breakpoints, raw_values)),
            'persist (k=5)':     (persist_traces[0], naive_bins_celsius(persist_bins)),
        }
    )