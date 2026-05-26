import argparse
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = BASE_DIR.parent

RESULTS_DIR = PROJECT_DIR / "Results"
DEFAULT_METHOD = "naiv"

parser = argparse.ArgumentParser(
    description="Plot benchmark statistics for a given method folder in Results"
)
parser.add_argument(
    "--method",
    default=DEFAULT_METHOD,
    help="Method subdirectory under Results (e.g. sax, persist, naiv)",
)
args = parser.parse_args()

METHOD = args.method
INPUT_DIR = RESULTS_DIR / METHOD

OUTPUT_DIR = RESULTS_DIR / "SVG" / METHOD
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

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
# Examples:
#   train-10t-sax-s5-w24
#   train-10t-persist-s10
#   test-20t-naiv-s15
# ============================================================

pattern = r"^(train|test)-(\d+)t-([^\s-]+)-s(\d+)(?:-w(\d+))?$"
parts = df["dataset"].str.extract(pattern)
parts.columns = ["Type", "Time", "Method", "Symbols", "Window"]

invalid = df[parts["Type"].isna() | parts["Time"].isna(
) | parts["Method"].isna() | parts["Symbols"].isna()]
if not invalid.empty:
    raise ValueError(
        "Unable to parse the following dataset names: "
        + ", ".join(invalid["dataset"].astype(str).unique()[:10])
    )

# Cast columns to the correct type
parts["Time"] = parts["Time"].astype(int)
parts["Symbols"] = parts["Symbols"].astype(int)
parts["Window"] = parts["Window"].astype("Int64")

for col in ["Type", "Time", "Symbols", "Method", "Window"]:
    df[col] = parts[col]

# Clean label for display
df["Dataset"] = df["Time"].astype(str) + "t"

# ============================================================
# SORT DATA (IMPORTANT)
# ============================================================

sort_columns = ["Type", "Method", "Window", "Time", "Symbols"]
df = df.sort_values(by=sort_columns, na_position="last").reset_index(drop=True)

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

selected_cols = ["Type"]
if "Window" in display_df.columns and display_df["Window"].notna().any():
    selected_cols.append("Window")

selected_cols += [
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

display_df = display_df[selected_cols]

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
        colLoc="center",
        edges="closed",
    )

    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.4)

    # Column index map (CRITICAL FIX)
    col_index = {name: i for i, name in enumerate(df.columns)}

    # Header styling
    header_color = "#f9d34b"
    for col in range(len(df.columns)):
        cell = table[(0, col)]
        cell.set_text_props(weight='bold')
        cell.set_facecolor(header_color)
        cell.set_edgecolor("black")
        cell.set_linewidth(1.0)

    # Body styling
    for row in range(1, n_rows + 1):
        for col in range(len(df.columns)):
            cell = table[(row, col)]
            cell.set_edgecolor("black")
            cell.set_linewidth(0.8)
            cell.set_text_props(ha="center", va="center")
            cell.set_facecolor("white")

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
# QUALITY SCORE
# lower KS + lower error + higher coverage
# ============================================================


df["quality_score"] = (
    (1 - df["ks_stat"]) * 0.5 +
    (1 / (1 + df["mean_error"])) * 0.3 +
    (df["coverage"]) * 0.2
)

best_configs = df.sort_values(by="quality_score", ascending=False).head(10)
print("Top 10 quality configs:")
print(best_configs[["dataset", "quality_score",
      "ks_stat", "mean_error", "coverage"]])

# ============================================================
# PLOT KS STATISTIC
# ============================================================

has_window = "Window" in df.columns and df["Window"].notna().any()

if has_window:
    windows = sorted(df["Window"].dropna().unique())
else:
    windows = [None]

for window in windows:
    if window is None:
        subset = df[df["Type"] == "test"]
        output_name = "ks.svg"
        title = "KS Statistic vs Time Horizon"
    else:
        subset = df[(df["Type"] == "test") & (df["Window"] == window)]
        output_name = f"ks_window_{window}.svg"
        title = f"KS Statistic vs Time Horizon (Window={window})"

    if subset.empty:
        print(f"Skipping KS plot for {title}: no rows")
        continue

    plt.figure(figsize=(8, 5))

    for s in sorted(subset["Symbols"].unique()):
        temp = subset[subset["Symbols"] == s]
        plt.plot(
            temp["Time"],
            temp["ks_stat"],
            marker="o",
            label=f"s={s}"
        )

    plt.xlabel("Time Horizon")
    plt.ylabel("KS Statistic")
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.savefig(OUTPUT_DIR / output_name, dpi=300, bbox_inches="tight")
    plt.close()

# ============================================================
# SAVE ALL TABLES
# ============================================================


def build_groups(display_df):
    groups = []
    has_window = "Window" in display_df.columns and display_df["Window"].notna(
    ).any()

    for dataset_type in ["train", "test"]:
        if has_window:
            for window in sorted(display_df["Window"].dropna().unique()):
                subset = display_df[
                    (display_df["Type"] == dataset_type) &
                    (display_df["Window"] == window)
                ].drop(columns=["Type", "Window"])
                groups.append((dataset_type, window, subset))
        else:
            subset = display_df[display_df["Type"] ==
                                dataset_type].drop(columns=["Type"])
            groups.append((dataset_type, None, subset))

    return groups


for dataset_type, window, table in build_groups(display_df):
    if window is None:
        title = f"{METHOD.upper()} Benchmark ({dataset_type.upper()})"
        output_file = OUTPUT_DIR / f"{dataset_type}.svg"
    else:
        title = f"{METHOD.upper()} Benchmark ({dataset_type.upper()}, Window={window})"
        output_file = OUTPUT_DIR / f"{dataset_type}_window{window}.svg"

    save_table(table, title, output_file)
