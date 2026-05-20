""" import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import ks_2samp
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

REAL_TRAIN_DIR = DATA_DIR / "3-ExtractInterval" / "1day-experiment" / "A-train"
REAL_TEST_DIR = DATA_DIR / "3-ExtractInterval" / "1day-experiment" / "A-test" / "positive"

SIMULATION_FILE = DATA_DIR / "7-ExtractedUppaalGraphData" / "SAX" /"1kSimulationsTemp.csv"

SIM_NAME = "1k-sims"

#outpufiler
SIM_RESULTS_DIR = RESULTS_DIR / SIM_NAME
SIM_RESULTS_DIR.mkdir(parents=True, exist_ok=True)

SIM_GRAPH_DIR = GRAPH_DIR / SIM_NAME
SIM_GRAPH_DIR.mkdir(parents=True, exist_ok=True)


# ======================================================
# LOAD REAL DATA
# ======================================================

def load_real_data(folder):
    data = []
    for file in folder.glob("roomA-1day-*.csv"):
        df = pd.read_csv(file, sep=";")
        data.append(df["temperature"].values)
    return np.concatenate(data)


real_train = load_real_data(REAL_TRAIN_DIR)
real_test = load_real_data(REAL_TEST_DIR)

print("Loaded train samples:", len(real_train))
print("Loaded test samples:", len(real_test))


# ======================================================
# LOAD UPPAAL SIMULATIONS (FIXED)
# ======================================================

print("Loading UPPAAL simulations...")

simulations_up = {}
current_sim = None

with open(SIMULATION_FILE, "r") as f:
    for line in f:
        line = line.strip()

        if not line:
            continue

        if line.startswith("#"):
            current_sim = line
            simulations_up[current_sim] = []
            continue

        if "temp" in line.lower() and "time" in line.lower():
            continue

        try:
            _, temp = line.split(",")
            simulations_up[current_sim].append(float(temp) / 100.0)
        except:
            continue

print("Loaded simulation traces:", len(simulations_up))


# ======================================================
# FIX: MAKE ALL SIMULATIONS SAME LENGTH (IMPORTANT)
# ======================================================

sim_list = list(simulations_up.values())

lengths = [len(s) for s in sim_list if len(s) > 0]

if len(lengths) == 0:
    raise ValueError("No valid simulation traces found")

EXPECTED_LEN = int(np.median(lengths))

clean_sims = []

for sim in sim_list:

    sim = np.array(sim)

    # SKIP EMPTY SIMS (IMPORTANT FIX)
    if len(sim) == 0:
        continue

    # TRIM LONG SIMS
    if len(sim) > EXPECTED_LEN:
        sim = sim[:EXPECTED_LEN]

    # PAD SHORT SIMS
    elif len(sim) < EXPECTED_LEN:
        sim = np.pad(sim, (0, EXPECTED_LEN - len(sim)), mode="edge")

    clean_sims.append(sim)

if len(clean_sims) == 0:
    raise ValueError("All simulation traces were empty")

sim_matrix = np.vstack(clean_sims)
sim_flat = sim_matrix.flatten()


# ======================================================
# METRICS FUNCTION
# ======================================================

def evaluate(real_data, sim_matrix, sim_flat, name):

    real_mean = np.mean(real_data)
    real_std = np.std(real_data)

    sim_means = np.mean(sim_matrix, axis=1)
    sim_mean = np.mean(sim_means)

    ci_low = np.percentile(sim_means, 2.5)
    ci_high = np.percentile(sim_means, 97.5)

    ks_stat, p_value = ks_2samp(real_data, sim_flat)

    sim_mean_trace = np.mean(sim_matrix, axis=0)
    sim_std_trace = np.std(sim_matrix, axis=0)

    min_len = min(len(real_data), len(sim_mean_trace))

    real_cut = real_data[:min_len]
    mean_cut = sim_mean_trace[:min_len]
    std_cut = sim_std_trace[:min_len]

    lower = mean_cut - 2 * std_cut
    upper = mean_cut + 2 * std_cut

    coverage = np.mean((real_cut >= lower) & (real_cut <= upper)) * 100

    return {
        "dataset": name,
        "real_mean": real_mean,
        "real_std": real_std,
        "sim_mean": sim_mean,
        "ci_low": ci_low,
        "ci_high": ci_high,
        "mean_in_ci": int(ci_low <= real_mean <= ci_high),
        "ks_stat": ks_stat,
        "ks_pvalue": p_value,
        "coverage": coverage
    }


# ======================================================
# RUN EVALUATION
# ======================================================

train_result = evaluate(real_train, sim_matrix, sim_flat, "train-1k-sims")
test_result = evaluate(real_test, sim_matrix, sim_flat, "test-1k-sims")

results = pd.DataFrame([train_result, test_result])


# ======================================================
# SAVE RESULTS
# ======================================================

results.to_csv(
    SIM_RESULTS_DIR / f"validation_summary_{SIM_NAME}.csv",
    index=False
)

print(f"\nSaved: validation_summary_{SIM_NAME}.csv")


# ======================================================
# PRINT SUMMARY
# ======================================================

print("\nFINAL SUMMARY")
print(results)


# ======================================================
# PLOTS (SAFE)
# ======================================================

sim_mean_trace = np.mean(sim_matrix, axis=0)
sim_std_trace = np.std(sim_matrix, axis=0)


# ---------------- HISTOGRAM ----------------

plt.figure(figsize=(10, 5))
plt.hist(sim_flat, bins=30, alpha=0.5, label="UPPAAL Simulation")
plt.hist(real_train, bins=30, alpha=0.5, label="Train Data")
plt.hist(real_test, bins=30, alpha=0.5, label="Test Data")

plt.xlabel("Temperature (°C)")
plt.ylabel("Frequency")
plt.title("Temperature Distribution: Train vs Test vs Simulation")
plt.legend()
plt.grid()

plt.savefig(
    SIM_GRAPH_DIR / f"histogram_train_test_sim_{SIM_NAME}.png",
    dpi=300,
    bbox_inches="tight"
)
plt.close()


# ---------------- TRAIN VS SIM ----------------




min_len_train = min(len(real_train), len(sim_mean_trace))
time_hours = np.linspace(0, 24, min_len_train)

plt.figure(figsize=(12, 5))

plt.plot(time_hours, real_train[:min_len_train], label="Train Data")
plt.plot(time_hours, sim_mean_trace[:min_len_train], label="Simulation Mean")

plt.fill_between(
    time_hours,
    sim_mean_trace[:min_len_train] - 2 * sim_std_trace[:min_len_train],
    sim_mean_trace[:min_len_train] + 2 * sim_std_trace[:min_len_train],
    alpha=0.3,
    label="±Simulation uncertainty (±2σ)"
)

plt.xlabel("Time (hours)")
plt.ylabel("Temperature (°C)")

plt.xticks(np.arange(0, 25, 2))

plt.title("Time Series: Train vs Simulation")
plt.legend()
plt.grid()

plt.savefig(
    SIM_GRAPH_DIR / f"timeseries_train_vs_sim_{SIM_NAME}.png",
    dpi=300,
    bbox_inches="tight"
)
plt.close()


# ---------------- TEST VS SIM ----------------

min_len_test = min(len(real_test), len(sim_mean_trace))
time_hours = np.linspace(0, 24, min_len_test)

plt.figure(figsize=(12, 5))

plt.plot(time_hours, real_test[:min_len_test], label="Test Data")
plt.plot(time_hours, sim_mean_trace[:min_len_test], label="Simulation Mean")

plt.fill_between(
    time_hours,
    sim_mean_trace[:min_len_test] - 2 * sim_std_trace[:min_len_test],
    sim_mean_trace[:min_len_test] + 2 * sim_std_trace[:min_len_test],
    alpha=0.3,
    label="Simulation uncertainty (±2σ)"
)

plt.title("Time Series: Test vs Simulation")
plt.xlabel("Time (hours)")
plt.ylabel("Temperature (°C)")

plt.xticks(np.arange(0, 25, 2))

plt.legend()
plt.grid()

plt.savefig(
    SIM_GRAPH_DIR / f"timeseries_test_vs_sim_{SIM_NAME}.png",
    dpi=300,
    bbox_inches="tight"
)
plt.close() """


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import ks_2samp
from pathlib import Path
import re

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

REAL_TRAIN_DIR = DATA_DIR / "3-ExtractInterval" / "1day-experiment" / "A-train"
REAL_TEST_DIR = DATA_DIR / "3-ExtractInterval" / \
    "1day-experiment" / "A-test" / "positive"

method = "sax"

SIMULATION_FOLDER = DATA_DIR / "7-ExtractedUppaalGraphData"/ method
simulation_files = list(SIMULATION_FOLDER.glob("*.csv"))

print(f"Found {len(simulation_files)} simulation files")

# ======================================================
# REAL DATA
# ======================================================


def load_real_data(folder):
    data = []
    for file in folder.glob("roomA-1day-*.csv"):
        df = pd.read_csv(file, sep=";")
        data.append(df["temperature"].values)
    return np.concatenate(data)


real_train = load_real_data(REAL_TRAIN_DIR)
real_test = load_real_data(REAL_TEST_DIR)

# ======================================================
# METRICS
# ======================================================


def evaluate(real_data, sim_matrix, sim_flat, name):

    real_mean = np.mean(real_data)
    real_std = np.std(real_data)

    sim_means = np.mean(sim_matrix, axis=1)
    sim_mean = np.mean(sim_means)

    ci_low = np.percentile(sim_means, 2.5)
    ci_high = np.percentile(sim_means, 97.5)

    ks_stat, p_value = ks_2samp(real_data, sim_flat)

    sim_mean_trace = np.mean(sim_matrix, axis=0)
    sim_std_trace = np.std(sim_matrix, axis=0)

    min_len = min(len(real_data), len(sim_mean_trace))

    real_cut = real_data[:min_len]
    mean_cut = sim_mean_trace[:min_len]
    std_cut = sim_std_trace[:min_len]

    coverage = np.mean(
        (real_cut >= mean_cut - 2 * std_cut) &
        (real_cut <= mean_cut + 2 * std_cut)
    ) * 100

    return {
        "dataset": name,
        "real_mean": real_mean,
        "real_std": real_std,
        "sim_mean": sim_mean,
        "ci_low": ci_low,
        "ci_high": ci_high,
        "ks_stat": ks_stat,
        "ks_pvalue": p_value,
        "coverage": coverage
    }

# ======================================================
# GROUP PARSING
# ======================================================


def extract_temp_group(name):
    match = re.match(r"(\d+t)", name)
    return match.group(1) if match else "unknown"


def extract_scenario(name):
    # extracts s10 from: 50t-sax-s10-w48
    match = re.search(r"(s\d+)", name)
    return match.group(1) if match else "unknown"

# ======================================================
# LOOP
# ======================================================


for sim_file in simulation_files:

    SIM_NAME = sim_file.stem

    TEMP_GROUP = extract_temp_group(SIM_NAME)   # 50t
    SCENARIO = extract_scenario(SIM_NAME)       # s10

    print("\n" + "=" * 60)
    print(f"{SIM_NAME} → {TEMP_GROUP} / {SCENARIO}")

    # ======================================================
    # OUTPUT STRUCTURE
    # ======================================================

    GROUP_RESULTS_DIR = RESULTS_DIR / method / TEMP_GROUP / SCENARIO
    GROUP_GRAPH_DIR = GRAPH_DIR / method / TEMP_GROUP / SCENARIO

    GROUP_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    GROUP_GRAPH_DIR.mkdir(parents=True, exist_ok=True)

    # ======================================================
    # LOAD SIMS
    # ======================================================

    simulations_up = {}
    current_sim = None

    with open(sim_file, "r") as f:

        for line in f:

            line = line.strip()
            if not line:
                continue

            if line.startswith("#"):
                current_sim = line
                simulations_up.setdefault(current_sim, [])
                continue

            try:
                t, temp = line.split(",")
                simulations_up[current_sim].append(float(temp) / 100.0)
            except:
                continue

    sim_list = [s for s in simulations_up.values() if len(s) > 5]

    if not sim_list:
        print(f"Skipping {SIM_NAME}")
        continue

    lengths = [len(s) for s in sim_list]
    EXPECTED_LEN = int(np.median(lengths))

    clean_sims = []

    for sim in sim_list:

        sim = np.array(sim)

        if len(sim) > EXPECTED_LEN:
            sim = sim[:EXPECTED_LEN]
        elif len(sim) < EXPECTED_LEN:
            sim = np.pad(sim, (0, EXPECTED_LEN - len(sim)), mode="edge")

        clean_sims.append(sim)

    sim_matrix = np.vstack(clean_sims)
    sim_flat = sim_matrix.flatten()




    # ======================================================
    # EVALUATION
    # ======================================================

    train_result = evaluate(real_train, sim_matrix,
                            sim_flat, f"train-{SIM_NAME}")
    test_result = evaluate(real_test, sim_matrix, sim_flat, f"test-{SIM_NAME}")

    results = pd.DataFrame([train_result, test_result])

    # ======================================================
    # SAVE
    # ======================================================

    results.to_csv(
        GROUP_RESULTS_DIR / f"summary_{SIM_NAME}.csv",
        index=False
    )

    print(f"Saved → Results/{TEMP_GROUP}/{SCENARIO}")

    # ======================================================
    # PLOT
    # ======================================================

    sim_mean = np.mean(sim_matrix, axis=0)
    sim_std = np.std(sim_matrix, axis=0)

    plt.figure(figsize=(10, 5))
    plt.hist(sim_flat, bins=30, alpha=0.5, label="Sim")
    plt.hist(real_train, bins=30, alpha=0.5, label="Train")
    plt.hist(real_test, bins=30, alpha=0.5, label="Test")

    plt.legend()
    plt.grid()

    plt.savefig(
        GROUP_GRAPH_DIR / f"hist_{SIM_NAME}.png",
        dpi=300,
        bbox_inches="tight"
    )
    plt.close()

print("\nALL DONE")
