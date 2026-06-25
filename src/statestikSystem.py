import re
from scipy.stats import ks_2samp
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.stats import wasserstein_distance


# ======================================================
# PATHS
# ======================================================

BASE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = BASE_DIR.parent
DATA_DIR = PROJECT_DIR / "Data"

RESULTS_DIR = PROJECT_DIR / "Results" / "system_evaluation"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

GRAPH_DIR = DATA_DIR / "Graphs" / "statestik" / "system_evaluation"
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

method = "persist"  # "naiv", "persist", "sax"

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

        data.append(
            df["temperature"].values
        )

    return np.concatenate(data)


real_train = load_real_data(REAL_TRAIN_DIR)
real_test = load_real_data(REAL_TEST_DIR)

# ======================================================
# RESAMPLE EVENT TRACE
# ======================================================


def resample_trace(
    events,
    step=60,
    end_time=86400
):

    times = np.arange(
        0,
        end_time,
        step
    )

    values = np.zeros(len(times))

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


# ======================================================
# METRICS HELPERS
# ======================================================

def align_traces(real, sim):
    """
    Simple alignment (truncation-based).
    Keeps evaluation stable and comparable.
    """
    n = min(len(real), len(sim))
    return real[:n], sim[:n]


def trend_error(real, sim):
    """
    Measures dynamic behavior similarity using first-order differences.
    """
    real_d = np.diff(real)
    sim_d = np.diff(sim)

    n = min(len(real_d), len(sim_d))

    if n == 0:
        return np.nan

    return np.mean((real_d[:n] - sim_d[:n]) ** 2)


def autocorr(x, lag=1):

    if len(x) <= lag:
        return np.nan

    x1 = x[:-lag]
    x2 = x[lag:]

    if np.std(x1) == 0 or np.std(x2) == 0:
        return 1.0

    return np.corrcoef(x1, x2)[0, 1]


# ======================================================
# SYSTEM BEHAVIOR EVALUATION
# ======================================================

def evaluate(
    real_data,
    sim_matrix,
    name
):

    sim_flat = sim_matrix.ravel()

    # ==================================================
    # 1. DISTRIBUTION METRICS
    # ==================================================

    real_mean = np.mean(real_data)
    real_std = np.std(real_data, ddof=1)

    sim_mean = np.mean(sim_flat)
    sim_std = np.std(sim_flat, ddof=1)

    mean_error = abs(real_mean - sim_mean)

    #ks_stat, ks_pvalue = ks_2samp(real_data, sim_flat)
    wasserstein = wasserstein_distance(real_data, sim_flat)

    low = np.percentile(sim_flat, 2.5)
    high = np.percentile(sim_flat, 97.5)

    coverage = np.mean(
        (real_data >= low) &
        (real_data <= high)
    ) * 100

    # ==================================================
    # 2. TEMPORAL MEMORY (AUTOCORRELATION)
    # ==================================================

    real_acf = autocorr(real_data)

    sim_acf = np.nanmean([
        autocorr(trace)
        for trace in sim_matrix
    ])

    acf_error = abs(real_acf - sim_acf)

    # ==================================================
    # 3. SYSTEM BEHAVIOR (DYNAMICS + TRAJECTORY)
    # ==================================================

    sim_trace = sim_matrix[0]

    real_aligned, sim_aligned = align_traces(
        real_data,
        sim_trace
    )

    trajectory_mse = np.mean(
        (real_aligned - sim_aligned) ** 2
    )

    trend_err = trend_error(
        real_aligned,
        sim_aligned
    )

    # ==================================================
    # RETURN RESULTS
    # ==================================================

    return {

        "dataset": name,

        "traces": sim_matrix.shape[0],
        "symbols": sim_matrix.shape[1],

        # distribution
        "real_mean": real_mean,
        "real_std": real_std,
        "sim_mean": sim_mean,
        "sim_std": sim_std,
        "mean_error": mean_error,

        #"ks_stat": ks_stat,
        "wasserstein_dist": wasserstein,
        #"ks_pvalue": ks_pvalue,
        "coverage": coverage,

        # temporal structure
        "real_acf": real_acf,
        "sim_acf": sim_acf,
        "acf_error": acf_error,

        # system behavior (NEW)
        "trajectory_mse": trajectory_mse,
        "trend_error": trend_err
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

    sim_flat = sim_matrix.ravel()

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

    plt.figure(figsize=(10, 5))

    plt.hist(
        sim_flat,
        bins=30,
        alpha=0.5,
        label="Sim"
    )

    plt.hist(
        real_train,
        bins=30,
        alpha=0.5,
        label="Train"
    )

    plt.hist(
        real_test,
        bins=30,
        alpha=0.5,
        label="Test"
    )

    plt.legend()

    plt.grid()

    plt.savefig(
        GROUP_GRAPH_DIR /
        f"hist_{SIM_NAME}.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

print("\nALL DONE")