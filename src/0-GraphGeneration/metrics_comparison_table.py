import pandas as pd
import numpy as np
import re
from pathlib import Path
import matplotlib.pyplot as plt

BASE_DIR = Path(__file__).resolve().parent.parent.parent

TOP_N = 5
SAVE_TABLE = True
#Title config
DISCRETIZATION = "SAX"
DATA_SET = "ECG"

# File config
W_FILTER = 48
discretization_method = "sax"
data_set = "ecg"

OUTPUT_PATH = (
        BASE_DIR
        / "Data"
        / "Graphs"
        / "Metrics_temp"
        / "Observed"
        / discretization_method
        / data_set
        /"table"
        / f"top-PAR-bins-{discretization_method}-{data_set}-table.svg"
)

metrics = BASE_DIR / "Data" / "8-LoggedData" / "metrics" / f"{discretization_method}-{data_set}-log-extra.csv"



# ----------------------------
# Load CSV
# ----------------------------
df = pd.read_csv(
    metrics,
    engine="python",
    usecols=lambda c: c != "FP_indices"
)

# -------------------------------------------------
# PARSE RUN ID
# Supports:
#   s5
#   w24   (SAX)
#   300trace
# -------------------------------------------------
def parse_run_id(rid):

    rid = str(rid)

    trace_match = re.search(r"(\d+)trace", rid)
    s_match = re.search(r"s(\d+)", rid)
    w_match = re.search(r"w(\d+)", rid)

    trace = int(trace_match.group(1)) if trace_match else None
    s = int(s_match.group(1)) if s_match else None
    w = int(w_match.group(1)) if w_match else None

    return trace, s, w


df["trace"], df["s"], df["w"] = zip(
    *df["run_id"].apply(parse_run_id)
)

# -------------------------------------------------
# NUMERIC
# -------------------------------------------------
numeric_cols = [
    "PAR",
    "NAR",
]

for col in numeric_cols:
    df[col] = pd.to_numeric(df[col], errors="coerce")

# Filter
df = df[df["trace"] != 1000]

# valid_w = [24, 48]
#
# if discretization_method.lower() == "sax" and "w" in df.columns:
#     df = df[df["w"].isin(valid_w)]
#

if discretization_method.lower() == "sax" and "w" in df.columns:
    if W_FILTER is not None:
        df = df[df["w"] == W_FILTER]


# -------------------------------------------------
# SORT BY PAR (highest first)
# -------------------------------------------------
top_df = (
    df.sort_values("PAR", ascending=False)
    .groupby("s", group_keys=False)
    .head(TOP_N)
    .reset_index(drop=True)
)

# -------------------------------------------------
# RANK COLUMN
# -------------------------------------------------
top_df.insert(0, "Rank", range(1, len(top_df) + 1))


# -------------------------------------------------
# CLEAN DISPLAY TABLE
# -------------------------------------------------

display_dict = {
    "Rank": top_df["Rank"].astype(int),
    "Traces": top_df["trace"].astype(int),
    "Bins": top_df["s"].astype("Int64"),
}




# Only SAX has
if discretization_method.lower() == "sax"  and "w" in top_df.columns:
    display_dict["Wordsize"] = top_df["w"].astype("Int64")

display_dict.update({
    "Positive Acceptance Rate": top_df["PAR"].map(lambda x: f"{x:.2f}%"),
    "Negative Acceptance Rate": top_df["NAR"].map(lambda x: f"{x:.2f}%"),
})

display_df = pd.DataFrame(display_dict)


# -------------------------------------------------
# PRINT TABLE
# -------------------------------------------------
print(f"\nTop {TOP_N} Positive Acceptance Rates (bins: 5, 10, 15) — {discretization_method}, {data_set}\n")
print(display_df.to_string(index=False))

# -------------------------------------------------
# CREATE FIGURE TABLE
# -------------------------------------------------
fig_h = 0.5 * len(display_df) + 1.5
#fig_h = max(8, 0.6 * len(display_df) + 2.5)

fig, ax = plt.subplots(figsize=(11, fig_h))
ax.axis("off")

fig.subplots_adjust(left=0, right=1, top=1, bottom=0)

fig.subplots_adjust(top=0.40)

table = ax.table(
    cellText=display_df.values,
    colLabels=display_df.columns,
    cellLoc="center",
    loc="center"
)

table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1.2, 1.5)
#table.scale(1.2, 3.2)


# ----------------------------
# COLOR BY BIN (s)
# ----------------------------
bin_colors = {
    5:  "#f5f8ff",  # near-white cool
    10: "#dbe7ff",  # soft pale blue
    15: "#b6ccff",  # medium muted blue (NOT saturated)
}

# IMPORTANT: use display_df OR top_df consistently
for row_idx in range(len(display_df)):
    bin_value = top_df.iloc[row_idx]["s"]

    color = bin_colors.get(bin_value, "white")

    # +1 because row 0 is header in matplotlib table
    for col_idx in range(len(display_df.columns)):
        table[(row_idx + 1, col_idx)].set_facecolor(color)


# -------------------------------------------------
# HEADER COLOR
# -------------------------------------------------
for j in range(len(display_df.columns)):
    table[(0, j)].set_facecolor("#FFD700")


# ----------------------------
# COLUMN WIDTH CONTROL (SAX SAFE)
# ----------------------------
col_widths = {}

for i, col in enumerate(display_df.columns):

    if col == "Traces":
        col_widths[i] = 0.08

    elif col == "Rank":
        col_widths[i] = 0.08

    elif col == "Bins":
        col_widths[i] = 0.06

    elif col == "Wordsize":
        col_widths[i] = 0.08  # slightly wider, fits "Wordsize"
    else:
        col_widths[i] = 0.25

for (row, col), cell in table.get_celld().items():
    if col in col_widths:
        cell.set_width(col_widths[col])

# -------------------------------------------------
# TITLE
# -------------------------------------------------

if discretization_method.lower() == "sax"  and "w" in top_df.columns:
    plt.title(
        f"\nTop {TOP_N} Positive Acceptance Rates — SAX for wordsize {W_FILTER},{DATA_SET} Dataset\n",
        fontsize=13,
        fontweight="bold",
        pad=20
    )
else:
    plt.title(
        f"\nTop {TOP_N} Positive Acceptance Rates — {DISCRETIZATION}, {DATA_SET} Dataset\n",
        fontsize=13,
        fontweight="bold",
        pad=20
    )


# -------------------------------------------------
# SAVE
# -------------------------------------------------
if SAVE_TABLE:

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    plt.savefig(
        OUTPUT_PATH,
        dpi=300,
        bbox_inches="tight"
    )

    print(f"\nSaved table to:\n{OUTPUT_PATH}")

plt.show()