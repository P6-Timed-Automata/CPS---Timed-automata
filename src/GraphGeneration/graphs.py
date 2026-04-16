import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import os


def plot_traces(trace_folder, output_folder):
    """
    Generates individual plots per trace and one overlay plot of all traces.

    Args:
        trace_folder : folder containing trace CSV files from extract_time_intervals
        output_folder: folder to save plots into (created if not exists)
    """
    os.makedirs(output_folder, exist_ok=True)

    # Load all trace files, sorted by trace number
    trace_files = sorted(
        [f for f in os.listdir(trace_folder) if f.endswith('.csv')],
        key=lambda f: int(f.split('_trace')[1].split('.')[0])
    )

    if not trace_files:
        print(f"No trace CSV files found in {trace_folder}")
        return

    # Use a colormap so each trace gets a distinct color on the overlay
    colors = cm.viridis(np.linspace(0, 1, len(trace_files)))

    fig_overlay, ax_overlay = plt.subplots(figsize=(12, 5))

    for i, filename in enumerate(trace_files):
        trace_name = filename.replace('.csv', '')
        filepath   = os.path.join(trace_folder, filename)
        data       = np.genfromtxt(filepath, delimiter=';', skip_header=1)
        times      = data[:, 0]
        temps      = data[:, 1]

        # --- Individual plot ---
        fig, ax = plt.subplots(figsize=(12, 5))
        ax.plot(times/(60*60), temps, color=colors[i], linewidth=0.8)
        ax.set_title(trace_name)
        ax.set_xlabel("Time (hours)")
        ax.set_ylabel("Temperature")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(os.path.join(output_folder, f"{trace_name}.png"), dpi=150)
        plt.close(fig)
        print(f"Saved individual plot for {trace_name}")

        # --- Add to overlay ---
        ax_overlay.plot(times/(60*60), temps, color=colors[i], linewidth=0.8, label=trace_name)

    # --- Save overlay plot ---
    ax_overlay.set_title("All Traces")
    ax_overlay.set_xlabel("Time (hours")
    ax_overlay.set_ylabel("Temperature")
    ax_overlay.grid(True, alpha=0.3)
    ax_overlay.legend(fontsize=7, ncol=2)
    fig_overlay.tight_layout()
    overlay_path = os.path.join(output_folder, "overlay.png")
    fig_overlay.savefig(overlay_path, dpi=150)
    plt.close(fig_overlay)
    print(f"Saved overlay plot to {overlay_path}")


# --- Usage ---
plot_traces(
    trace_folder  = ".../../Data/3-ExtractInterval/A/30day",
    output_folder = "experiment_30_day_trace_plots"
)