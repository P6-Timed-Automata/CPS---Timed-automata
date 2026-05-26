import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent.parent

# ----------------------------
# CONFIG
# ----------------------------
S_FILTER = 15
W_FILTER = 24        # set to 24, 48, etc OR None to ignore
SAVE_FIG = True

discretization_method = "naiv"
data_set = "temp"
OUTPUT_PATH = (
        BASE_DIR
        / "Data"
        / "Graphs"
        / "Metrics_temp"
        / "Observed"
        / discretization_method
        / data_set
        / "graph"
        / f"s{S_FILTER}-w{W_FILTER}-{discretization_method}-{data_set}-graph.svg"
)

metrics = BASE_DIR / "Data" / "8-LoggedData" / "metrics" / f"{discretization_method}-{data_set}-log-done.csv"


# ----------------------------
# Load CSV safely
# ----------------------------
df = pd.read_csv(
    metrics,
    engine="python",
    usecols=lambda c: c != "FP_indices"
)


# ----------------------------
# Flexible parser (s, w, k, trace)
# ----------------------------
def parse_run_id(rid):
    rid = str(rid)

    s = re.search(r"s(\d+)", rid)
    w = re.search(r"w(\d+)", rid)
    k = re.search(r"k(\d+)", rid)
    t = re.search(r"-(\d+)trace", rid)

    return (
        int(s.group(1)) if s else np.nan,
        int(w.group(1)) if w else np.nan,
        int(k.group(1)) if k else np.nan,
        int(t.group(1)) if t else np.nan
    )


df["s"], df["w"], df["k"], df["trace"] = zip(*df["run_id"].apply(parse_run_id))


# ----------------------------
# Filtering
# ----------------------------
df = df[df["s"] == S_FILTER]

df = df[df["trace"] != 1000]

if W_FILTER is not None:
    df = df[df["w"] == W_FILTER]


# ----------------------------
# Numeric safety
# ----------------------------
for col in ["precision", "recall", "f1"]:
    df[col] = pd.to_numeric(df[col], errors="coerce")


# ----------------------------
# Aggregate per trace
# ----------------------------
df_agg = df.groupby("trace").agg(
    precision=("precision", "mean"),
    recall=("recall", "mean"),
    f1=("f1", "mean")
).reset_index()

df_agg = df_agg.sort_values("trace")


# ----------------------------
# Plot
# ----------------------------

plt.subplots_adjust(top=0.88)

plt.plot(df_agg["trace"], df_agg["precision"], marker="o", label="Precision")
plt.plot(df_agg["trace"], df_agg["recall"], marker="o", label="Recall")
plt.plot(df_agg["trace"], df_agg["f1"], marker="o", label="F1")


# ----------------------------
# Best F1 point (simple + valid)
# ----------------------------
best = df_agg.loc[df_agg["f1"].idxmax()]

plt.scatter(
    best["trace"],
    best["f1"],
    color="gold",
    s=160,
    label="Best F1",
    zorder=5
)

plt.annotate(
    f"Best F1={best['f1']:.3f}\nP={best['precision']:.2f}\nR={best['recall']:.2f}",
    (best["trace"], best["f1"]),
    xytext=(10, 10),
    textcoords="offset points"
)


# ----------------------------
# Labels
# ----------------------------
plt.xlabel("Number of Traces")
plt.ylabel("Score")
plt.title(f"Metric evolution (bins={S_FILTER}" + (f", w={W_FILTER})" if W_FILTER else ")"))
plt.grid(True)
plt.legend()


# ----------------------------
# Save safely
# ----------------------------
if SAVE_FIG:
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(OUTPUT_PATH, dpi=300, bbox_inches="tight")
    print(f"Saved figure to: {OUTPUT_PATH}")


plt.show()