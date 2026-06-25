import re
from scipy.signal import welch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# ======================================================
# PATHS
# ======================================================

BASE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = BASE_DIR.parent
DATA_DIR = PROJECT_DIR / "Data"

RESULTS_DIR = PROJECT_DIR / "Results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

GRAPH_DIR = DATA_DIR / "Graphs" / "statestik"
GRAPH_DIR.mkdir(parents=True, exist_ok=True)

REAL_TRAIN_DIR = (
    DATA_DIR /
    "3-ExtractInterval" /
    "1day-experiment" /
    "A-train"
)

REAL_TEST_DIR = (
    DATA_DIR /
    "3-ExtractInterval" /
    "1day-experiment" /
    "A-test" /
    "positive"
)

method = "naiv"  # "naiv", "persist", "sax"

if method == "sax":

    SIMULATION_FOLDER = (
        DATA_DIR /
        "7-ExtractedUppaalGraphData" /
        method
    )

else:

    SIMULATION_FOLDER = (
        DATA_DIR /
        "7-ExtractedUppaalGraphData" /
        method /
        "temp"
    )

simulation_files = list(
    SIMULATION_FOLDER.glob("*.csv")
)

print(f"Found {len(simulation_files)} simulation files")

# ======================================================
# REAL DATA
# ======================================================


def load_real_data(folder):

    data = []

    for file in folder.glob("roomA-1day-*.csv"):

        df = pd.read_csv(file, sep=";")

        if "temperature" not in df.columns:
            raise ValueError(f"Missing temperature column in {file}")

        data.append(df["temperature"].values)

    if not data:
        raise ValueError(f"No real data files found in {folder}")

    return np.vstack(data)


real_train = load_real_data(REAL_TRAIN_DIR)
real_test = load_real_data(REAL_TEST_DIR)

# ======================================================
# RESAMPLE EVENT TRACE
# ======================================================


def resample_trace(
    events,
    step=300,
    end_time=86400
):

    times = np.arange(
        0,
        end_time,
        step
    )

    values = np.zeros(len(times))

    if not events:
        return values

    current_temp = events[0][1]

    event_index = 0

    for i, t in enumerate(times):

        while (
            event_index + 1 < len(events) and
            events[event_index + 1][0] <= t
        ):

            event_index += 1

            current_temp = events[event_index][1]

        values[i] = current_temp / 100.0

    return values

# ======================================================
# METRICS
# ======================================================


def autocorr(x, lag=1):

    if len(x) <= lag:
        return np.nan

    x1 = x[:-lag]
    x2 = x[lag:]

    # ------------------------------------------
    # CONSTANT SIGNAL CHECK
    # ------------------------------------------

    if np.std(x1) == 0 or np.std(x2) == 0:
        return 1.0

    return np.corrcoef(x1, x2)[0, 1]


def align_traces(real, sim):
    n = min(len(real), len(sim))
    return real[:n], sim[:n]


def trend_error(real, sim):
    real_d = np.diff(real)
    sim_d = np.diff(sim)

    n = min(len(real_d), len(sim_d))
    if n == 0:
        return np.nan

    return np.mean((real_d[:n] - sim_d[:n]) ** 2)


def acf_series(x, nlags=100):

    if len(x) <= 1:
        return np.full(nlags, np.nan)

    return np.array([
        autocorr(x, lag=i)
        for i in range(1, nlags + 1)
    ])


def evaluate(
    real_data,
    sim_matrix,
    name
):

    real_mean_series = np.mean(real_data, axis=0)
    sim_mean_series = np.mean(sim_matrix, axis=0)

    real_mean = np.mean(real_mean_series)
    sim_mean = np.mean(sim_mean_series)

    real_std = np.std(real_mean_series, ddof=1)
    sim_std = np.std(sim_mean_series, ddof=1)

    # ==================================================
    # RMSE on the aligned mean series
    # ==================================================

    errors = real_mean_series - sim_mean_series
    rmse = np.sqrt(np.mean(errors ** 2))
    mean_error = np.mean(np.abs(errors))

    # ==================================================
    # Pointwise coverage across the sim ensemble
    # ==================================================

    lower = np.percentile(sim_matrix, 2.5, axis=0)
    upper = np.percentile(sim_matrix, 97.5, axis=0)

    coverage = np.mean(
        (real_data >= lower) &
        (real_data <= upper)
    ) * 100

    # ==================================================
    # Variance error on the dynamic spread
    # ==================================================

    real_std_series = np.std(real_data, axis=0, ddof=1)
    sim_std_series = np.std(sim_matrix, axis=0, ddof=1)
    variance_error = np.mean((real_std_series - sim_std_series) ** 2)

    # ==================================================
    # ACF comparison on the mean series
    # ==================================================

    NLAGS = max(1, len(real_mean_series) - 1)
    real_acf_series = acf_series(real_mean_series, nlags=NLAGS)
    sim_acf_series = acf_series(sim_mean_series, nlags=NLAGS)
    acf_rmse = np.sqrt(np.nanmean((real_acf_series - sim_acf_series) ** 2))

    # ==================================================
    # PSD comparison on the mean series
    # ==================================================

    fs = 1.0 / 300.0 # Sampling frequency (1 sample every 300 seconds)
    nperseg = min(256, len(real_mean_series), len(sim_mean_series))

    try:
        _, psd_real = welch(real_mean_series, fs=fs, nperseg=nperseg)
        _, psd_sim = welch(sim_mean_series, fs=fs, nperseg=nperseg)
        psd_rmse = np.sqrt(np.mean((psd_real - psd_sim) ** 2))
    except Exception:
        psd_rmse = np.nan

    return {
        "dataset": name,
        "traces": sim_matrix.shape[0],
        "symbols": sim_matrix.shape[1],
        "real_mean": real_mean,
        "real_std": real_std,
        "sim_mean": sim_mean,
        "sim_std": sim_std,
        "mean_error": mean_error,
        "coverage": coverage,
        "variance_error": variance_error,
        "acf_rmse": acf_rmse,
        "psd_rmse": psd_rmse,
        "rmse": rmse,
    }

# ======================================================
# GROUP PARSING
# ======================================================


def extract_temp_group(name):

    match = re.match(
        r"(\d+t)",
        name
    )

    return (
        match.group(1)
        if match else "unknown"
    )


def extract_scenario(name):

    match = re.search(
        r"(s\d+)",
        name
    )

    return (
        match.group(1)
        if match else "unknown"
    )

# ======================================================
# LOOP
# ======================================================


for sim_file in simulation_files:

    SIM_NAME = sim_file.stem

    TEMP_GROUP = extract_temp_group(SIM_NAME)
    SCENARIO = extract_scenario(SIM_NAME)

    print("\n" + "=" * 60)
    print(f"{SIM_NAME} → {TEMP_GROUP} / {SCENARIO}")

    # ==================================================
    # OUTPUT STRUCTURE
    # ==================================================

    GROUP_RESULTS_DIR = (
        RESULTS_DIR / method
    )

    GROUP_GRAPH_DIR = (
        GRAPH_DIR /
        method /
        TEMP_GROUP /
        SCENARIO
    )

    GROUP_RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    GROUP_GRAPH_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    # ==================================================
    # LOAD EVENT TRACES
    # ==================================================

    simulations_up = {}

    current_sim = None

    with open(sim_file, "r") as f:

        for line in f:

            line = line.strip()

            if not line:
                continue

            if line.startswith("#"):

                current_sim = line

                simulations_up[current_sim] = []

                continue

            try:

                time, temp = line.split(",")

                simulations_up[current_sim].append(
                    (
                        float(time),
                        float(temp)
                    )
                )

            except:
                continue

    # ==================================================
    # RESAMPLE TRACES
    # ==================================================

    clean_sims = []

    for trace in simulations_up.values():

        if len(trace) < 2:
            continue

        sampled = resample_trace(trace)

        clean_sims.append(sampled)

    if not clean_sims:

        print(f"Skipping {SIM_NAME}")
        continue

    sim_matrix = np.vstack(clean_sims)

    # ==================================================
    # EVALUATION
    # ==================================================

    train_result = evaluate(
        real_train,
        sim_matrix,
        f"train-{SIM_NAME}"
    )

    test_result = evaluate(
        real_test,
        sim_matrix,
        f"test-{SIM_NAME}"
    )

    results = pd.DataFrame([
        train_result,
        test_result
    ])

    # ==================================================
    # SAVE CSV
    # ==================================================

    results.to_csv(
        GROUP_RESULTS_DIR /
        f"summary_{SIM_NAME}.csv",
        index=False
    )

    print(
        f"Saved → Results/{TEMP_GROUP}/{SCENARIO}"
    )

    # ==================================================
    # PLOTS
    # ==================================================

    real_train_flat = real_train.ravel()
    sim_flat = sim_matrix.ravel()
    real_train_mean = np.mean(real_train, axis=0)
    sim_mean = np.mean(sim_matrix, axis=0)

    lower = np.percentile(sim_matrix, 2.5, axis=0)
    upper = np.percentile(sim_matrix, 97.5, axis=0)

    # Histogram of marginals
    plt.figure(figsize=(8, 4))
    plt.hist(sim_flat, bins=30, alpha=0.5, label="Sim", density=True)
    plt.hist(real_train_flat, bins=30, alpha=0.5,
             label="Real train", density=True)
    plt.xlabel("Temperature")
    plt.ylabel("Density")
    plt.title(f"Histogram — {SIM_NAME}")
    plt.legend()
    plt.grid()
    plt.savefig(GROUP_GRAPH_DIR /
                f"hist_{SIM_NAME}.png", dpi=300, bbox_inches="tight")
    plt.close()

    # Mean daily profile + coverage band
    plt.figure(figsize=(12, 5))
    t = np.arange(sim_matrix.shape[1])
    plt.plot(t, sim_mean, label="Sim mean", color="tab:orange")
    plt.plot(t, real_train_mean, label="Real train mean", color="tab:blue")
    plt.fill_between(t, lower, upper, color="tab:orange",
                     alpha=0.2, label="Sim 95% band")
    plt.xlabel("Time index")
    plt.ylabel("Temperature")
    plt.title(f"Mean daily profile + coverage band — {SIM_NAME}")
    plt.legend()
    plt.grid()
    plt.savefig(GROUP_GRAPH_DIR /
                f"profile_{SIM_NAME}.png", dpi=300, bbox_inches="tight")
    plt.close()

    # ACF comparison on mean series
    NLAGS = 50
    real_acf_s = acf_series(real_train_mean, nlags=NLAGS)
    sim_acf_s = acf_series(sim_mean, nlags=NLAGS)

    plt.figure(figsize=(8, 4))
    lags = np.arange(1, NLAGS + 1)
    plt.plot(lags, real_acf_s, label="Real train mean")
    plt.plot(lags, sim_acf_s, label="Sim mean")
    plt.xlabel("Lag")
    plt.ylabel("Autocorrelation")
    plt.legend()
    plt.grid()
    plt.title(f"ACF comparison — {SIM_NAME}")
    plt.savefig(GROUP_GRAPH_DIR /
                f"acf_{SIM_NAME}.png", dpi=300, bbox_inches="tight")
    plt.close()

    # PSD comparison on mean series
    fs = 1.0 / 300.0
    nperseg = min(128, len(sim_mean))
    try:
        f_r, psd_r = welch(real_train_mean, fs=fs, nperseg=nperseg)
        f_s, psd_s = welch(sim_mean, fs=fs, nperseg=nperseg)
        plt.figure(figsize=(8, 4))
        plt.semilogy(f_r, psd_r, label="Real train mean")
        plt.semilogy(f_s, psd_s, label="Sim mean")
        plt.xlabel("Frequency (Hz)")
        plt.ylabel("PSD")
        plt.legend()
        plt.grid()
        plt.title(f"PSD (Welch) — {SIM_NAME}")
        plt.savefig(GROUP_GRAPH_DIR /
                    f"psd_{SIM_NAME}.png", dpi=300, bbox_inches="tight")
        plt.close()
    except Exception:
        pass

print("\nALL DONE")
