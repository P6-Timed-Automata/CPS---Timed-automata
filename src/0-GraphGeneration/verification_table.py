import re
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ----------------------------
# PATHS
# ----------------------------
BASE_DIR = Path(__file__).resolve().parent.parent.parent
PROJECT_DIR = BASE_DIR.parent

# Config title
DISCRETIZATION_METHOD = "Sax"
DATASET = "Temperature"


# Config file

discretization_method = "sax"
data_set = "temp"

OUTPUT_DIR = (
        BASE_DIR
        / "Data"
        / "Graphs"
        / "Metrics_temp"
        / "Observed"
        / discretization_method
        / data_set
        / "table"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

INPUT_DIR = BASE_DIR / "Data" / "8-LoggedData" / "queries" / discretization_method
input_file = INPUT_DIR / f"{discretization_method}-{data_set}-queries.csv"

# ----------------------------
# LOAD
# ----------------------------
with open(input_file, newline='', encoding='utf-8') as f:
    sample = f.read(4096)
    f.seek(0)
    delimiter = __import__('csv').Sniffer().sniff(sample, delimiters=';,').delimiter

df = pd.read_csv(input_file, sep=delimiter)
df["Config"] = df["Config"].astype(str)

# ----------------------------
# EXTRACT TRACE / BINS / WORDSIZE
# ----------------------------
def parse_config(cfg):
    trace = re.search(r"(\d+)trace", cfg)
    s = re.search(r"s(\d+)", cfg)
    w = re.search(r"w(\d+)", cfg)

    return (
        int(trace.group(1)) if trace else None,
        int(s.group(1)) if s else None,
        int(w.group(1)) if w else None,
    )

df[["Trace", "Bins", "Wordsize"]] = df["Config"].apply(
    lambda x: pd.Series(parse_config(x))
)

# ----------------------------
# COLUMNS
# ----------------------------

# For ECG
if data_set == "ecg":
    selected_cols = {
        "E2": "E<> SPIKE",
        "A[]1": "A[] not deadlock",
        "P5": "Pr<> (FLAT && SPIKE_seen && Flat duration)",
        "P6": "Pr<> (FLAT && !SPIKE_seen && Flat duration)",
        "P7": "Pr<> (SPIKE && Spike duration)",
    }
else:
    # For temp
    selected_cols = {
        "E2": "E<> SPIKE",
        "A[]1": "A<> Temp range",
        "A[]2": "A[] not deadlock",
        "P7": "Pr<> (STABILIZED && temp range)",
        "P8": "Pr<> (SPIKE_seen && STABILIZED)",
    }

# ----------------------------
# CLEAN FUNCTIONS
# ----------------------------
def clean_probability(x):
    if pd.isna(x):
        return ""

    text = str(x).strip()
    text = text.replace(">=", "").replace("<=", "").replace(">", "").replace("<", "")
    text = text.split("+/-")[0].strip()

    try:
        val = float(text)
    except:
        return ""

    if val <= 1:
        val *= 100

    return f"{val:.2f}%"


def parse_boolean_symbol(value):
    if pd.isna(value):
        return ""

    text = str(value).strip()



    if text in {"1", "1.0", "True", "true", "TRUE"}:
        return "✓"
    if text in {"0", "0.0", "False", "false", "FALSE"}:
        return "✕"

    try:
        return "✓" if float(text) >= 0.5 else "✕"
    except:
        return ""

# ----------------------------
# MAIN TABLE FUNCTION
# ----------------------------
def save_table(subset: pd.DataFrame, window_value):
    if subset.empty:
        print("empty subset")
        return

    df_local = subset.copy()

    # ----------------------------
    # BUILD TABLE
    # ----------------------------
    table_df = pd.DataFrame({
        "Traces": df_local["Trace"].astype("Int64"),
        "Bins": df_local["Bins"].astype("Int64"),
    })

    # ONLY show Wordsize for SAX
    if discretization_method.lower() == "sax" and "Wordsize" in df_local.columns:
        table_df["Wordsize"] = df_local["Wordsize"].astype("Int64")


    if data_set == "ecg":
        # bools for ecg
        for c in ["E2", "A[]1"]:
            table_df[selected_cols[c]] = df_local[c].apply(parse_boolean_symbol)

        # probabiklite for (ecg)
        for c in ["P5", "P6", "P7"]:
            table_df[selected_cols[c]] = df_local[c].apply(clean_probability)
    else:
        # booleans for temp
        for c in ["E2", "A[]1", "A[]2"]:
            table_df[selected_cols[c]] = df_local[c].apply(parse_boolean_symbol)

        # probabilities for (temp)
        for c in ["P7", "P8"]:
            table_df[selected_cols[c]] = df_local[c].apply(clean_probability)


    col_labels = table_df.columns.tolist()
    table_data = table_df.values

    # ----------------------------
    # FIGURE
    # ----------------------------
    fig, ax = plt.subplots(figsize=(12, max(3, len(table_df) * 0.35)))
    ax.axis("off")

    # ADD THIS (important)
    if discretization_method.lower() == "sax":
        fig.suptitle(
            f"Verification of Selected Queries for {DISCRETIZATION_METHOD} with wordsize {window_value}, {DATASET} Dataset",
            fontsize=14,
            fontweight="bold",
            y=0.98
        )
    else:
        fig.suptitle(
            f"Verification of Selected Queries for {DISCRETIZATION_METHOD}, {DATASET} Dataset",
            fontsize=14,
            fontweight="bold",
            y=0.98
        )


    # leave space for title
    fig.subplots_adjust(top=0.99)

    table = ax.table(
        cellText=table_data,
        colLabels=col_labels,
        cellLoc="center",
        loc="center"
    )

    # ----------------------------
    # DYNAMIC COLUMN WIDTHS (FIXED)
    # ----------------------------

    if data_set == "ecg":
        # for ecg
        base_widths = {
            "Traces": 0.05,
            "Bins": 0.05,
            "Wordsize": 0.06,

            "E<> SPIKE": 0.10,

            "A[] not deadlock": 0.12,

            "Pr<> (FLAT && SPIKE_seen && Flat duration)": 0.32,
            "Pr<> (FLAT && !SPIKE_seen && Flat duration)": 0.32,
            "Pr<> (SPIKE && Spike duration)": 0.22,
        }
    else:
        #for temp
        base_widths = {
            "Traces": 0.05,
            "Bins": 0.05,
            "Wordsize": 0.06,

            "E<> SPIKE": 0.10,
            "A<> Temp range": 0.12,
            "A[] not deadlock": 0.12,

            "Pr<> (STABILIZED && temp range)": 0.25,
            "Pr<> (SPIKE_seen && STABILIZED)": 0.25,
        }



    for col_idx, col_name in enumerate(col_labels):
        if col_name in base_widths:
            width = base_widths[col_name]
        else:
            width = 0.1  # fallback

        for (row, col), cell in table.get_celld().items():
            if col == col_idx:
                cell.set_width(width)


    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.2, 1.5)

    # ----------------------------
    # HEADER
    # ----------------------------
    for j in range(len(col_labels)):
        cell = table[(0, j)]
        cell.set_facecolor("#1f3b5b")
        cell.set_text_props(color="white", weight="bold")

    # ----------------------------
    # BODY
    # ----------------------------
    for i in range(1, len(table_df) + 1):
        for j in range(len(col_labels)):
            cell = table[(i, j)]

            if j < 2:
                cell.set_facecolor("#f5f7fa")
                continue

            val = table_df.iloc[i - 1, j]

            # IMPORTANT: handle pandas NA safely
            if pd.isna(val):
                cell.set_facecolor("#ffffff")
                continue

            val = str(val)

            if val == "✓":
                cell.set_facecolor("#4caf50")
                cell.set_text_props(color="white", weight="bold")

            elif val == "✕":
                cell.set_facecolor("#f44336")
                cell.set_text_props(color="white", weight="bold")

            else:
                cell.set_facecolor("#e8f0ff")

    # ----------------------------
    # SAVE
    # ----------------------------
    suffix = ""
    if discretization_method.lower() == "sax":
        suffix = f"-w{window_value}"

    out = OUTPUT_DIR / f"{discretization_method}-{data_set}{suffix}-verification-table.svg"
    plt.savefig(out, bbox_inches="tight", dpi=300)
    plt.close()

    print("Saved:", out)

# ----------------------------
# RUN
# ----------------------------
window_matches = df["Config"].str.extract(r"-w(\d+)", expand=False)
window_values = sorted(set(window_matches.dropna().astype(int)))

if window_values:
    for w in [24, 48]:
        subset = df[df["Config"].str.contains(f"w{w}\\b")].copy()
        save_table(subset, w)
else:
    save_table(df, "all")