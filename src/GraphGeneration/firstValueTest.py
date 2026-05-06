import numpy as np
import matplotlib.pyplot as plt
import glob

def load_trace(path):
    data = np.genfromtxt(path, delimiter=';', skip_header=1)
    return data[:, 0], data[:, 1]

trace_paths = sorted(glob.glob("../../data/3-ExtractInterval/1day-experiment/roomA/*.csv"))

first_vals, last_vals, mean_vals = [], [], []
for p in trace_paths:
    t, v = load_trace(p)
    first_vals.append(v[0])
    last_vals.append(v[-1])
    mean_vals.append(v.mean())

def plot_distribution(values, title, output_path):
    values = np.array(values)
    bins   = np.arange(np.floor(values.min()), np.ceil(values.max()) + 1, 1)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(values, bins=bins, color='steelblue', alpha=0.85, edgecolor='white', linewidth=0.5)
    ax.axvline(values.mean(), color='crimson', linestyle='--',
               linewidth=1.4, label=f"Mean = {values.mean():.2f} °C")

    ax.set_xlabel("Temperature (°C)")
    ax.set_ylabel("Number of traces")
    ax.set_title(f"{title} — n={len(values)} traces")
    ax.set_xticks(bins)
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.4, axis='y')
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"{title}: min={values.min():.2f}  max={values.max():.2f}  "
          f"mean={values.mean():.2f}  std={values.std():.2f} °C  → {output_path}")

plot_distribution(first_vals, "Distribution of first temperature value", "dist_first.png")
plot_distribution(last_vals,  "Distribution of last temperature value",  "dist_last.png")
plot_distribution(mean_vals,  "Distribution of mean temperature",        "dist_mean.png")