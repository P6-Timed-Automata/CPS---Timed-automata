"""
exp_baseline.py
===============
Non-TAG threshold baseline classifier for thesis comparison.

Classifies each test trace as anomalous if its mean falls more than k·σ
from the training set's mean of means. This is the simplest reasonable
non-TAG anomaly detector — it answers "what would a freshman implement?"

Run on the same data as exp_51_52 to get directly comparable P/R/F1
numbers. Use these as a reference point in thesis discussion:
  - Where TAG-based methods beat threshold significantly, TAG adds value.
  - Where threshold matches TAG, the anomaly type is detectable by
    simple mean-shift and TAG's temporal structure is not contributing.

Sweeps k_sigma to characterize how the threshold's strictness affects
detection.

Output (timestamped folder under Graphs/Baseline_threshold/):
  config.txt
  results.json
  table_overall.csv
  table_overall.png
  table_per_mode_clean.csv
  table_per_mode_noisy.csv
  comparison_<metric>.png
"""

import csv
import json
import subprocess
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from Generate_data import load_all_data
from Generators import NEG_MODE_NAMES


# =============================================================================
# CONFIG
# =============================================================================

# Sweep k_sigma to show how threshold strictness affects P/R/F1.
# 2.0 is the textbook "outside 2 standard deviations" choice.
K_SIGMA_VARIANTS = [1.5, 2.0, 2.5, 3.0]


# =============================================================================
# HELPERS
# =============================================================================

def _git_hash():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT
        ).decode().strip()
    except Exception:
        return "unknown"


def _trace_mean(trace):
    """Mean temperature of one trace. Accepts (times, temps) or [(temp, time), ...]."""
    if isinstance(trace, tuple) and len(trace) == 2:
        times, temps = trace
        return float(np.mean(temps))
    # else assume list of (value, time) pairs
    return float(np.mean([v for v, _ in trace]))


def _classify(train_traces, test_traces, k_sigma):
    """
    Returns a list of bools: True if the trace is flagged as anomalous.

    The classifier:
      1. Computes the mean of each training trace.
      2. mu = mean(train_means), sigma = std(train_means)
      3. For each test trace, computes its mean.
      4. Flags as anomalous if abs(test_mean - mu) > k_sigma * sigma.
    """
    train_means = np.array([_trace_mean(t) for t in train_traces])
    mu = float(np.mean(train_means))
    sigma = float(np.std(train_means))
    if sigma == 0:
        sigma = 1e-9   # degenerate case

    flags = []
    for trace in test_traces:
        m = _trace_mean(trace)
        flags.append(abs(m - mu) > k_sigma * sigma)
    return flags, mu, sigma


def _metrics(pos_flags, neg_flags):
    """
    Standard P/R/F1 from binary flags.
    pos_flags : booleans for positive (clean) test traces — True = flagged anomalous (false positive)
    neg_flags : booleans for negative test traces — True = flagged anomalous (true positive)
    """
    TP = sum(neg_flags)
    FP = sum(pos_flags)
    FN = sum(1 for f in neg_flags if not f)
    TN = sum(1 for f in pos_flags if not f)

    precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        "TP": TP, "FP": FP, "FN": FN, "TN": TN,
        "precision": precision, "recall": recall, "f1": f1,
        "PAR": (TP + FN and TN / (TP + FN)) or 0.0,    # positive acceptance rate
        "NAR": (FP + TN and FP / (FP + TN)) * 100      # negative acceptance rate (%)
    }


def _per_mode_metrics(pos_flags, neg_flags, neg_modes):
    """
    Compute per-anomaly-mode rejection rates.
    Same shape as run_pipeline returns, so plots/tables stay compatible.
    """
    per_mode = {}
    mode_indices = defaultdict(list)
    for i, m in enumerate(neg_modes):
        mode_indices[m].append(i)

    for mode_idx, indices in mode_indices.items():
        mode_neg_flags = [neg_flags[i] for i in indices]
        mode_metrics = _metrics(pos_flags, mode_neg_flags)
        name = NEG_MODE_NAMES.get(mode_idx, str(mode_idx))
        per_mode[name] = {
            "rejection": 100.0 - mode_metrics["NAR"],
            "precision": mode_metrics["precision"],
            "recall":    mode_metrics["recall"],
            "f1":        mode_metrics["f1"],
        }
    return per_mode


# =============================================================================
# CONFIG FILE
# =============================================================================

def _save_config(out_dir, data):
    from collections import Counter
    lines = [
        "=" * 55,
        "Run configuration — Threshold Baseline",
        "=" * 55,
        "",
        f"Timestamp     : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"Git hash      : {_git_hash()}",
        "",
        "Classifier    : mean ± k·σ on per-trace mean",
        "",
        "--- k_sigma variants swept ---",
        f"  {K_SIGMA_VARIANTS}",
        "",
        "--- Dataset sizes ---",
        f"  clean_train : {len(data['clean_train'])} traces",
        f"  clean_test  : {len(data['clean_test'])} traces",
        f"  noisy_train : {len(data['noisy_train'])} traces",
        f"  noisy_test  : {len(data['noisy_test'])} traces",
        f"  negatives   : {len(data['neg_traces'])} traces "
        f"({len(set(data['neg_modes']))} modes)",
        "",
        "--- Negative modes ---",
        ]
    mode_counts = Counter(data["neg_modes"])
    for mode_int, count in sorted(mode_counts.items()):
        lines.append(f"  {NEG_MODE_NAMES[mode_int]:10s}: {count} traces")

    lines += ["", "--- Output folder ---", f"  {out_dir}", "", "=" * 55]
    (out_dir / "config.txt").write_text("\n".join(lines))
    print(f"  Saved: {out_dir / 'config.txt'}")


# =============================================================================
# PLOTS AND TABLES
# =============================================================================

def _save_overall_table(results, out_dir):
    csv_path = out_dir / "table_overall.csv"
    headers = ["condition", "k_sigma", "precision", "recall", "f1",
               "TP", "FP", "FN", "TN", "train_mu", "train_sigma"]
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        for r in results:
            ov = r["overall"]
            writer.writerow([
                r["condition"], r["k_sigma"],
                f"{ov['precision']:.3f}", f"{ov['recall']:.3f}",
                f"{ov['f1']:.3f}",
                ov["TP"], ov["FP"], ov["FN"], ov["TN"],
                f"{r['train_mu']:.3f}", f"{r['train_sigma']:.3f}",
            ])
    print(f"  Saved: {csv_path}")


def _save_per_mode_table(results, condition, out_dir):
    mode_names = list(NEG_MODE_NAMES.values())
    csv_path = out_dir / f"table_per_mode_{condition}.csv"
    headers = ["k_sigma"] + [m.capitalize() for m in mode_names]

    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        for r in results:
            if r["condition"] != condition:
                continue
            row = [r["k_sigma"]]
            for mode in mode_names:
                pct = r["per_mode"].get(mode, {}).get("rejection", 0.0)
                row.append(f"{pct:.1f}%")
            writer.writerow(row)
    print(f"  Saved: {csv_path}")


def _plot_metric_vs_ksigma(results, metric, ylabel, out_path):
    """Line plot showing how the chosen metric varies with k_sigma."""
    conditions = ["clean", "noisy"]
    colors = {"clean": "steelblue", "noisy": "darkorange"}

    fig, ax = plt.subplots(figsize=(8, 5))
    for cond in conditions:
        xs = []
        ys = []
        for r in results:
            if r["condition"] != cond:
                continue
            xs.append(r["k_sigma"])
            ys.append(r["overall"][metric])
        order = np.argsort(xs)
        xs = [xs[i] for i in order]
        ys = [ys[i] for i in order]
        ax.plot(xs, ys, marker="o", color=colors[cond], linewidth=2,
                markersize=6, label=cond)
        for x, y in zip(xs, ys):
            ax.text(x, y + 0.02, f"{y:.2f}", ha="center", fontsize=8)

    ax.set_xlabel("k_sigma (threshold strictness)")
    ax.set_ylabel(ylabel)
    ax.set_title(f"Threshold baseline — {ylabel} vs k_sigma")
    ax.set_ylim(0, 1.1)
    ax.set_xticks(K_SIGMA_VARIANTS)
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.4)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


# =============================================================================
# MAIN
# =============================================================================

def _run_condition(label, train_traces, test_pos, test_neg, neg_modes):
    results = []
    for k_sigma in K_SIGMA_VARIANTS:
        pos_flags, mu, sigma = _classify(train_traces, test_pos, k_sigma)
        neg_flags, _, _ = _classify(train_traces, test_neg, k_sigma)
        overall = _metrics(pos_flags, neg_flags)
        per_mode = _per_mode_metrics(pos_flags, neg_flags, neg_modes)

        print(f"  [{label}] k_sigma={k_sigma}  "
              f"P={overall['precision']:.3f} R={overall['recall']:.3f} "
              f"F1={overall['f1']:.3f}")

        results.append({
            "condition":   label,
            "k_sigma":     k_sigma,
            "train_mu":    mu,
            "train_sigma": sigma,
            "overall":     overall,
            "per_mode":    per_mode,
        })
    return results


if __name__ == "__main__":
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    out_dir = ROOT / "Data" / "Graphs" / "Baseline_threshold" / timestamp
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output folder: {out_dir}\n")

    data = load_all_data()
    _save_config(out_dir, data)

    print(f"\nSweeping k_sigma ∈ {K_SIGMA_VARIANTS}\n")

    log = {
        "timestamp": timestamp,
        "git_hash":  _git_hash(),
        "k_sigma_variants": K_SIGMA_VARIANTS,
        "n_train_clean":  len(data["clean_train"]),
        "n_train_noisy":  len(data["noisy_train"]),
        "n_test":         len(data["clean_test"]),
        "n_neg":          len(data["neg_traces"]),
        "results":        [],
    }

    print("=== Clean training ===")
    log["results"] += _run_condition(
        "clean",
        data["clean_train"], data["clean_test"],
        data["neg_traces"], data["neg_modes"],
    )
    print()

    print("=== Noisy training ===")
    log["results"] += _run_condition(
        "noisy",
        data["noisy_train"], data["noisy_test"],
        data["neg_traces"], data["neg_modes"],
    )

    with open(out_dir / "results.json", "w") as f:
        json.dump(log, f, indent=2)
    print(f"\n  Saved: {out_dir / 'results.json'}")

    print("\n=== Tables ===")
    _save_overall_table(log["results"], out_dir)
    _save_per_mode_table(log["results"], "clean", out_dir)
    _save_per_mode_table(log["results"], "noisy", out_dir)

    print("\n=== Plots ===")
    _plot_metric_vs_ksigma(log["results"], "precision", "Precision",
                           out_dir / "comparison_precision.png")
    _plot_metric_vs_ksigma(log["results"], "recall", "Recall",
                           out_dir / "comparison_recall.png")
    _plot_metric_vs_ksigma(log["results"], "f1", "F1",
                           out_dir / "comparison_f1.png")

    print(f"\nDone. Results -> {out_dir}")