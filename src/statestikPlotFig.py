import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parent.parent
RESULTS_DIR = ROOT_DIR / "Results"
METHOD_DIRS = {
    "naiv": RESULTS_DIR / "naiv",
    "persist": RESULTS_DIR / "persist",
    "sax": RESULTS_DIR / "sax",
}
OUTPUT_DIR = RESULTS_DIR / "figures" / "benchmark"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def parse_dataset_name(dataset_name: str) -> pd.Series:
    parts = str(dataset_name).split("-")
    result = {
        "Type": None,
        "Time": None,
        "Method": None,
        "Symbols": None,
        "Window": None,
    }

    for part in parts:
        if part in {"train", "test"}:
            result["Type"] = part
            continue

        if part.endswith("t") and part[:-1].isdigit():
            result["Time"] = int(part[:-1])
            continue

        if part.startswith("s") and part[1:].isdigit():
            result["Symbols"] = int(part[1:])
            continue

        if part.startswith("w") and part[1:].isdigit():
            result["Window"] = int(part[1:])
            continue

        if part in {"sax", "persist", "naiv"}:
            result["Method"] = part
            continue

    return pd.Series(result)


def method_display_label(method: str, window: float | int | None) -> str:
    if method == "naiv":
        return "Naive"
    if method == "persist":
        return "Persist"
    if method == "sax":
        if pd.notna(window):
            return f"SAX w{int(window)}"
        return "SAX"
    return str(method)


def load_all_results() -> pd.DataFrame:
    frames = []
    for method_name, directory in METHOD_DIRS.items():
        if not directory.exists():
            continue

        for csv_file in sorted(directory.glob("*.csv")):
            df = pd.read_csv(csv_file)
            df["source_method"] = method_name
            frames.append(df)

    if not frames:
        raise FileNotFoundError(
            "No CSV files found in Results/naiv, Results/persist, or Results/sax"
        )

    df = pd.concat(frames, ignore_index=True)
    df = df.drop_duplicates().reset_index(drop=True)
    return df


def prepare_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    if "rmse" not in df.columns and "real_mean" in df.columns and "sim_mean" in df.columns:
        df["rmse"] = np.sqrt((df["real_mean"] - df["sim_mean"]).pow(2))

    parsed = df["dataset"].apply(parse_dataset_name)
    for col in parsed.columns:
        df[col] = parsed[col]

    df["MethodLabel"] = df.apply(
        lambda row: method_display_label(row.get("Method"), row.get("Window")), axis=1
    )

    df["Time"] = pd.to_numeric(df["Time"], errors="coerce")
    df["Symbols"] = pd.to_numeric(df["Symbols"], errors="coerce")
    df["Window"] = pd.to_numeric(df["Window"], errors="coerce")

    df["GroupLabel"] = df.apply(
        lambda row: f"{row.Type}-{int(row.Time)}t-s{int(row.Symbols)}"
        if pd.notna(row.Time) and pd.notna(row.Symbols)
        else str(row.dataset),
        axis=1,
    )

    return df


def order_methods(labels: list[str]) -> list[str]:
    order = ["Naive", "Persist"] + sorted(
        [label for label in labels if label.startswith("SAX")],
        key=lambda label: int(label.split("w")[-1]) if "w" in label else 999,
    )
    return [label for label in order if label in labels]


def _order_types(types: list[str]) -> list[str]:
    order = ["train", "test"]
    return [t for t in order if t in types] + [t for t in types if t not in order]


def build_grouped_results(df: pd.DataFrame) -> pd.DataFrame:
    numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
    group_cols = ["Time", "Symbols", "Window", "MethodLabel", "Type"]
    return df.groupby(group_cols, dropna=False, as_index=False)[numeric_columns].mean()


def plot_metrics_benchmark(group_df: pd.DataFrame, output_prefix: Path) -> None:
    """Generate bar chart figures for key metrics."""
    key_metrics = {
        "coverage": "Coverage (%)",
        "rmse": "RMSE",
        "mean_error": "Mean Error",
        "acf_rmse": "ACF RMSE",
    }

    methods = order_methods(group_df["MethodLabel"].unique().tolist())
    types = _order_types(group_df["Type"].dropna().unique().tolist())

    for metric_col, metric_label in key_metrics.items():
        if metric_col not in group_df.columns:
            continue

        fig, ax = plt.subplots(figsize=(10, 6))
        x = np.arange(len(methods))
        width = 0.8 / len(types)

        for i, typ in enumerate(types):
            values = []
            for method in methods:
                subset = group_df[(group_df["MethodLabel"] == method) & (
                    group_df["Type"] == typ)]
                if not subset.empty:
                    values.append(float(subset[metric_col].mean()))
                else:
                    values.append(np.nan)

            offset = (i - (len(types) - 1) / 2) * width
            bars = ax.bar(x + offset, values, width,
                          label=typ.title(), alpha=0.8)

            for bar in bars:
                height = bar.get_height()
                if not np.isnan(height):
                    ax.text(bar.get_x() + bar.get_width() / 2, height,
                            f'{height:.2f}', ha='center', va='bottom', fontsize=8)

        ax.set_ylabel(metric_label, fontsize=12, weight="bold")
        ax.set_xlabel("Method", fontsize=12, weight="bold")
        ax.set_title(f"{metric_label} Comparison", fontsize=14, weight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(methods, fontsize=11)
        ax.legend(fontsize=10)
        ax.grid(axis='y', alpha=0.3)

        plt.tight_layout()
        output_prefix.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_prefix.parent /
                    f"{metric_col}_comparison.png", dpi=300, bbox_inches="tight")
        plt.close(fig)


def save_table_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def save_table_svg(df: pd.DataFrame, path: Path, title: str | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(
        figsize=(max(10, 1.2 * len(df.columns)), 1.5 + 0.4 * len(df)))
    ax.axis("off")
    table = ax.table(
        cellText=df.round(3).fillna("").values,
        colLabels=df.columns,
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.8)
    if title:
        ax.set_title(title, pad=20, fontsize=14, weight="bold")
    plt.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def build_key_benchmark_table(group_df: pd.DataFrame) -> pd.DataFrame:
    methods = order_methods(group_df["MethodLabel"].unique().tolist())
    types = _order_types(group_df["Type"].dropna().unique().tolist())

    if "coverage" not in group_df.columns or "acf_rmse" not in group_df.columns:
        return pd.DataFrame()

    rows = []
    for method in methods:
        row = {"Method": method}
        method_rows = group_df[group_df["MethodLabel"] == method]

        for typ in ["train", "test"]:
            type_rows = method_rows[method_rows["Type"] == typ]
            row[f"{typ.title()} coverage"] = float(
                type_rows["coverage"].mean()) if not type_rows.empty else np.nan
            row[f"{typ.title()} ACF RMSE"] = float(
                type_rows["acf_rmse"].mean()) if not type_rows.empty else np.nan

        if not np.isnan(row.get("Train coverage", np.nan)) and not np.isnan(row.get("Test coverage", np.nan)):
            row["Gap"] = row["Test coverage"] - row["Train coverage"]
        else:
            row["Gap"] = np.nan

        train_acf = row.get("Train ACF RMSE", np.nan)
        test_acf = row.get("Test ACF RMSE", np.nan)
        row["Verdict"] = classify_benchmark_verdict(
            row.get("Gap", np.nan), train_acf, test_acf)

        rows.append(row)

    table_df = pd.DataFrame(rows)
    return table_df


def classify_benchmark_verdict(gap: float, train_acf: float, test_acf: float) -> str:
    if np.isnan(gap):
        return "Unknown"

    if gap < -15:
        return "Overfitting"

    acf_diff = abs(test_acf - train_acf) if not np.isnan(
        train_acf) and not np.isnan(test_acf) else np.nan

    if abs(gap) <= 2 and not np.isnan(acf_diff) and acf_diff <= 0.12:
        return "Stable"

    if gap < 0:
        return "Unstable"

    return "Variable"


def plot_key_benchmark_table(group_df: pd.DataFrame, output_path: Path) -> None:
    table_df = build_key_benchmark_table(group_df)
    if table_df.empty:
        return

    save_table_csv(table_df, output_path.with_suffix(".csv"))

    fig, ax = plt.subplots(
        figsize=(12, 2.2 + 0.7 * len(table_df)))
    ax.axis("off")

    display_df = table_df.copy()
    display_df["Train coverage"] = display_df["Train coverage"].map(
        lambda x: f"{x:.1f}%" if pd.notna(x) else "")
    display_df["Test coverage"] = display_df["Test coverage"].map(
        lambda x: f"{x:.1f}%" if pd.notna(x) else "")
    display_df["Gap"] = display_df["Gap"].map(
        lambda x: f"{x:+.1f}%" if pd.notna(x) else "")
    display_df["Train ACF RMSE"] = display_df["Train ACF RMSE"].map(
        lambda x: f"{x:.2f}" if pd.notna(x) else "")
    display_df["Test ACF RMSE"] = display_df["Test ACF RMSE"].map(
        lambda x: f"{x:.2f}" if pd.notna(x) else "")

    table = ax.table(
        cellText=display_df.values,
        colLabels=display_df.columns,
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.7)

    for i in range(len(display_df.columns)):
        table[(0, i)].set_facecolor("#3b7dd8")
        table[(0, i)].set_text_props(weight="bold", color="white")

    for i in range(1, len(display_df) + 1):
        for j in range(len(display_df.columns)):
            if j == 0:
                table[(i, j)].set_facecolor("#ebf3ff")

    ax.set_title("Benchmark Table — Key Metrics",
                 pad=20, fontsize=16, weight="bold")
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    df = load_all_results()
    df = prepare_dataframe(df)
    grouped = build_grouped_results(df)

    # Filter for Time=30 and Symbols in [10, 15]
    filtered = grouped[
        (grouped["Time"] == 30) &
        (grouped["Symbols"].isin([10, 15]))
    ].copy()

    if filtered.empty:
        print("No data found for Time=30 and Symbols in [10, 15]")
        return

    # Group by Symbol value for separate benchmark figures
    for symbols in sorted(filtered["Symbols"].unique()):
        symbol_data = filtered[filtered["Symbols"] == symbols].copy()

        label = f"30t-s{int(symbols)}"
        print(f"Building benchmark figures for {label}")

        # Plot individual metric comparisons
        plot_metrics_benchmark(symbol_data, OUTPUT_DIR / label / "metrics")

        # Plot benchmark table
        plot_key_benchmark_table(
            symbol_data, OUTPUT_DIR / label / "benchmark_table.png")

    # Also create overall benchmark combining all
    print("Building overall benchmark figure")
    plot_metrics_benchmark(filtered, OUTPUT_DIR / "overall_metrics")
    plot_key_benchmark_table(filtered, OUTPUT_DIR /
                             "overall_benchmark_table.png")

    print(f"Saved benchmark figures to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
