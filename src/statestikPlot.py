import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np

# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = BASE_DIR.parent

RESULTS_ROOT = PROJECT_DIR / "Results"

# Set this manually to sax, persist, or naiv
METHOD = "naiv"  # Change this to "sax" or "naiv" or "persist"

INPUT_DIR = RESULTS_ROOT / METHOD

# if not INPUT_DIR.exists():
#    INPUT_DIR = RESULTS_ROOT / "system_evaluation" / METHOD

OUTPUT_DIR = RESULTS_ROOT / "SVG" / METHOD
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================
# LOAD DATA
# ============================================================

csv_files = list(INPUT_DIR.glob("*.csv"))

if not csv_files:
    raise FileNotFoundError(f"No CSV files found in {INPUT_DIR}")

df = pd.concat(
    [pd.read_csv(f) for f in csv_files],
    ignore_index=True
)

print(f"Loaded {len(csv_files)} files")
print(f"Total rows: {len(df)}")

# ============================================================
# FEATURE ENGINEERING
# ============================================================

if "rmse" not in df.columns and "real_mean" in df.columns and "sim_mean" in df.columns:
    df["rmse"] = np.sqrt((df["real_mean"] - df["sim_mean"]).pow(2))

# ============================================================
# PARSE DATASET NAME
# ============================================================


def parse_dataset_name(name):

    parts = str(name).split("-")

    result = {
        "Type": None,
        "Time": None,
        "Method": None,
        "Symbols": None,
        "Window": None
    }

    if len(parts) > 0:
        result["Type"] = parts[0]

    for p in parts:

        if p.endswith("t") and p[:-1].isdigit():
            result["Time"] = int(p[:-1])

        elif p.startswith("s") and p[1:].isdigit():
            result["Symbols"] = int(p[1:])

        elif p.startswith("w") and p[1:].isdigit():
            result["Window"] = int(p[1:])

    methods = ["sax", "persist", "naiv"]

    for p in parts:
        if p in methods:
            result["Method"] = p
            break

    return pd.Series(result)


parsed = df["dataset"].apply(parse_dataset_name)

for col in parsed.columns:
    df[col] = parsed[col]

# ============================================================
# CLEAN TYPES
# ============================================================

for col in ["Time", "Symbols", "Window"]:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

df["Traces"] = df["Time"].astype("Int64").astype(str)

# ============================================================
# SORT DATA
# ============================================================

sort_columns = ["Type", "Method", "Window", "Time", "Symbols"]

df = df.sort_values(
    by=[c for c in sort_columns if c in df.columns],
    na_position="last"
).reset_index(drop=True)

# ============================================================
# FORMAT TABLE
# ============================================================

display_df = df.copy()

round_cols = [
    "real_mean", "real_std",
    "sim_mean", "sim_std",
    "mean_error",
    "coverage",
    "rmse", "acf_rmse",
]

display_df[
    [c for c in round_cols if c in display_df.columns]
] = display_df[
    [c for c in round_cols if c in display_df.columns]
].round(3)

display_df = display_df.rename(columns={

    "real_mean": "μ Real",
    "real_std": "σ Real",
    "sim_mean": "μ Sim",
    "sim_std": "σ Sim",
    "mean_error": "Mean Error",
    "rmse": "RMSE",
    "coverage": "Coverage",
    "acf_rmse": "ACF RMSE",
})

selected_cols = ["Type"]

if "Window" in display_df.columns and display_df["Window"].notna().any():
    selected_cols.append("Window")

selected_cols += [
    "Traces",
    "Symbols",

    "μ Real",
    "σ Real",
    "μ Sim",
    "σ Sim",

    "Mean Error",
    "RMSE",
    "Coverage",
    "ACF RMSE",
]

selected_cols = [c for c in selected_cols if c in display_df.columns]

display_df = display_df[selected_cols]

# ============================================================
# TABLE FUNCTION
# ============================================================


def save_table(df, title, output_path):

    if df.empty:
        return

    fig, ax = plt.subplots(figsize=(16, max(2, len(df) * 0.45)))

    ax.axis("off")

    table = ax.table(
        cellText=df.values,
        colLabels=df.columns,
        loc="center",
        cellLoc="center"
    )

    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.4)

    col_index = {name: i for i, name in enumerate(df.columns)}

    # HEADER
    for col in range(len(df.columns)):
        table[(0, col)].set_facecolor("#f9d34b")

    # BODY COLORING
    for i in range(1, len(df) + 1):

        # KS
       # if "KS Stat" in df.columns:
        #    v = df.iloc[i-1]["KS Stat"]
        #   c = table[(i, col_index["KS Stat"])]

        #  c.set_facecolor(
        #     "#c7f5cc" if v < 0.2
        #    else "#fff3bf" if v < 0.5
        #   else "#ffc9c9"
        # )

        # RMSE
        if "RMSE" in df.columns:
            v = df.iloc[i-1]["RMSE"]
            c = table[(i, col_index["RMSE"])]

            c.set_facecolor(
                "#c7f5cc" if v < 0.5
                else "#fff3bf" if v < 1.5
                else "#ffc9c9"
            )

        # MAE
        if "MAE" in df.columns:
            v = df.iloc[i-1]["MAE"]
            c = table[(i, col_index["MAE"])]

            c.set_facecolor(
                "#c7f5cc" if v < 0.5
                else "#fff3bf" if v < 1.0
                else "#ffc9c9"
            )

        # Correlation
        if "Correlation" in df.columns:
            v = df.iloc[i-1]["Correlation"]
            c = table[(i, col_index["Correlation"])]

            c.set_facecolor(
                "#c7f5cc" if v >= 0.9
                else "#fff3bf" if v >= 0.7
                else "#ffc9c9"
            )

        # Coverage
        if "Coverage" in df.columns:
            v = df.iloc[i-1]["Coverage"]
            c = table[(i, col_index["Coverage"])]

            c.set_facecolor(
                "#c7f5cc" if v >= 95
                else "#fff3bf" if v >= 50
                else "#ffc9c9"
            )

        # ACF Real
        if "ACF Real" in df.columns:
            c = table[(i, col_index["ACF Real"])]
            c.set_facecolor("#ffffff")

        # ACF Sim
        if "ACF Sim" in df.columns:
            c = table[(i, col_index["ACF Sim"])]
            c.set_facecolor("#ffffff")

        # ACF Error
        if "ACF Error" in df.columns:
            v = df.iloc[i-1]["ACF Error"]
            c = table[(i, col_index["ACF Error"])]

            c.set_facecolor(
                "#c7f5cc" if v < 0.05
                else "#fff3bf" if v < 0.15
                else "#ffc9c9"
            )

        # Trajectory MSE
        if "Trajectory MSE" in df.columns:
            v = df.iloc[i-1]["Trajectory MSE"]
            c = table[(i, col_index["Trajectory MSE"])]

            c.set_facecolor(
                "#c7f5cc" if v < 0.5
                else "#fff3bf" if v < 2
                else "#ffc9c9"
            )

        # Trend Error
        if "Trend Error" in df.columns:
            v = df.iloc[i-1]["Trend Error"]
            c = table[(i, col_index["Trend Error"])]

            c.set_facecolor(
                "#c7f5cc" if v < 0.05
                else "#fff3bf" if v < 0.15
                else "#ffc9c9"
            )

        # ACF RMSE
        if "ACF RMSE" in df.columns:
            v = df.iloc[i-1]["ACF RMSE"]
            c = table[(i, col_index["ACF RMSE"])]

            c.set_facecolor(
                "#c7f5cc" if v < 0.05
                else "#fff3bf" if v < 0.15
                else "#ffc9c9"
            )

    plt.title(title, fontsize=14, weight="bold")
    plt.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)

    plt.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

# ============================================================
# QUALITY SCORE
# ============================================================


quality = np.zeros(len(df), dtype=float)

if "mean_error" in df.columns:
    quality += (1 / (1 + df["mean_error"])) * 0.30

if "rmse" in df.columns:
    quality += (1 / (1 + df["rmse"])) * 0.18

if "coverage" in df.columns:
    quality += (df["coverage"] / 100) * 0.18

if "acf_rmse" in df.columns:
    quality += (1 / (1 + df["acf_rmse"])) * 0.08

if "quality_score" in df.columns:
    df["quality_score_old"] = df["quality_score"]
df["quality_score"] = quality

best_cols = ["dataset", "quality_score"]
for col in ["mean_error", "rmse", "coverage", "acf_rmse"]:
    if col in df.columns:
        best_cols.append(col)

best_configs = df.sort_values(
    "quality_score",
    ascending=False
).head(10)

print("\nTop 10 configs:\n")
print(best_configs[best_cols])

# ============================================================
# SAVE TABLES
# ============================================================


def build_groups(display_df):

    groups = []

    has_window = (
        "Window" in display_df.columns and
        display_df["Window"].notna().any()
    )

    for dataset_type in ["train", "test"]:

        if has_window:

            for w in sorted(
                display_df["Window"].dropna().unique()
            ):

                subset = display_df[
                    (display_df["Type"] == dataset_type) &
                    (display_df["Window"] == w)
                ].drop(columns=["Type", "Window"])

                groups.append((dataset_type, w, subset))

        else:

            subset = display_df[
                display_df["Type"] == dataset_type
            ].drop(columns=["Type"])

            groups.append((dataset_type, None, subset))

    return groups


for dataset_type, window, table in build_groups(display_df):

    title = (
        f"{METHOD.upper()} Benchmark ({dataset_type.upper()})"
        if window is None else
        f"{METHOD.upper()} Benchmark ({dataset_type.upper()}, Window={window})"
    )

    out = (
        OUTPUT_DIR / f"{dataset_type}.svg"
        if window is None else
        OUTPUT_DIR / f"{dataset_type}_w{window}.svg"
    )

    save_table(table, title, out)

print(f"\nSaved tables to {OUTPUT_DIR}")
print("\nALL DONE")
