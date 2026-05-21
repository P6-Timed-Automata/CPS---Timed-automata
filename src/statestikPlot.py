import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = BASE_DIR.parent

RESULTS_DIR = PROJECT_DIR / "Results"
METHOD = "sax"

INPUT_DIR = RESULTS_DIR / METHOD

OUTPUT_DIR = RESULTS_DIR / "PNG" / METHOD
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================
# OUTPUT FILES
# ============================================================

train_24_output = OUTPUT_DIR / "train_window24.png"
train_48_output = OUTPUT_DIR / "train_window48.png"
test_24_output = OUTPUT_DIR / "test_window24.png"
test_48_output = OUTPUT_DIR / "test_window48.png"

# ============================================================
# LOAD ALL CSV FILES
# ============================================================

csv_files = list(INPUT_DIR.glob("*.csv"))

if not csv_files:
    raise FileNotFoundError(f"No CSV files found in {INPUT_DIR}")

df = pd.concat([pd.read_csv(f) for f in csv_files], ignore_index=True)

print(f"Loaded {len(csv_files)} files")
print(f"Total rows: {len(df)}")

# ============================================================
# FEATURE ENGINEERING
# ============================================================

df["mean_error"] = (df["real_mean"] - df["sim_mean"]).abs()

# ============================================================
# PARSE DATASET NAME
# Example: train-10t-sax-s5-w24
# ============================================================

parts = df["dataset"].str.extract(r"(train|test)-(\d+)t-sax-s(\d+)-w(\d+)")

df["Type"] = parts[0]
df["Time"] = parts[1].astype(int)
df["Symbols"] = parts[2].astype(int)
df["Window"] = parts[3].astype(int)

# Clean label for display
df["Dataset"] = df["Time"].astype(str) + "t"

# ============================================================
# SORT DATA (IMPORTANT)
# ============================================================

df = df.sort_values(
    by=["Type", "Window", "Time", "Symbols"]
).reset_index(drop=True)

# ============================================================
# FORMAT TABLE
# ============================================================

display_df = df.copy()

round_cols = [
    "real_mean",
    "real_std",
    "sim_mean",
    "mean_error",
    "ks_stat",
    "ks_pvalue",
    "coverage",
]

display_df[round_cols] = display_df[round_cols].round(3)

display_df = display_df.rename(columns={
    "real_mean": "μ Real",
    "real_std": "σ Real",
    "sim_mean": "μ Sim",
    "mean_error": "Mean Error",
    "ks_stat": "KS Stat",
    "ks_pvalue": "KS p-value",
    "coverage": "Coverage",
})

display_df = display_df[
    [
        "Type",
        "Window",
        "Dataset",
        "Symbols",
        "μ Real",
        "σ Real",
        "μ Sim",
        "Mean Error",
        "KS Stat",
        "KS p-value",
        "Coverage",
    ]
]

# ============================================================
# SPLIT INTO 4 GROUPS
# ============================================================

train_24 = display_df[
    (display_df["Type"] == "train") & (display_df["Window"] == 24)
].drop(columns=["Type", "Window"])

train_48 = display_df[
    (display_df["Type"] == "train") & (display_df["Window"] == 48)
].drop(columns=["Type", "Window"])

test_24 = display_df[
    (display_df["Type"] == "test") & (display_df["Window"] == 24)
].drop(columns=["Type", "Window"])

test_48 = display_df[
    (display_df["Type"] == "test") & (display_df["Window"] == 48)
].drop(columns=["Type", "Window"])

# ============================================================
# SAFE TABLE FUNCTION
# ============================================================

def save_table(df, title, output_path):

    if df.empty:
        print(f"Skipping empty table: {title}")
        return

    n_rows = len(df)
    fig_height = max(2, n_rows * 0.45)

    fig, ax = plt.subplots(figsize=(14, fig_height))
    ax.axis("off")

    table = ax.table(
        cellText=df.values,
        colLabels=df.columns,
        loc="center",
        cellLoc="center",
    )

    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.3)

    # Column index map (CRITICAL FIX)
    col_index = {name: i for i, name in enumerate(df.columns)}

    # Header styling
    for col in range(len(df.columns)):
        cell = table[(0, col)]
        cell.set_text_props(weight='bold')
        cell.set_facecolor("#d9d9d9")

    # ========================================================
    # COLORING
    # ========================================================
    for i in range(1, n_rows + 1):

        # KS STAT
        ks_value = df.iloc[i - 1]["KS Stat"]
        ks_cell = table[(i, col_index["KS Stat"])]

        if ks_value < 0.2:
            ks_cell.set_facecolor("#c7f5cc")
        elif ks_value < 0.5:
            ks_cell.set_facecolor("#fff3bf")
        else:
            ks_cell.set_facecolor("#ffc9c9")

        # COVERAGE
        cov_value = df.iloc[i - 1]["Coverage"]
        cov_cell = table[(i, col_index["Coverage"])]

        if cov_value > 0.8:
            cov_cell.set_facecolor("#c7f5cc")
        elif cov_value > 0.5:
            cov_cell.set_facecolor("#fff3bf")
        else:
            cov_cell.set_facecolor("#ffc9c9")

    plt.title(title, fontsize=14, weight="bold", pad=20)

    plt.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)

    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Saved: {output_path}")

# ============================================================
# SAVE ALL TABLES
# ============================================================

save_table(train_24, "SAX Benchmark (TRAIN, Window=24)", train_24_output)
save_table(train_48, "SAX Benchmark (TRAIN, Window=48)", train_48_output)

save_table(test_24, "SAX Benchmark (TEST, Window=24)", test_24_output)
save_table(test_48, "SAX Benchmark (TEST, Window=48)", test_48_output)