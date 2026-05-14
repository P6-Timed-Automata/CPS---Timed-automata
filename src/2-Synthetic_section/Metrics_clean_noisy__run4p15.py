"""
exp_51_52_run.py
================
Runs Experiment 5.1 (clean training) and Experiment 5.2 (noisy training)
across multiple parameter variants per method.

Writes results.json to a timestamped folder. The companion plotter
(exp_51_52_plot.py) reads that file and produces all tables/figures
without re-running pipelines.

Usage:
    python exp_51_52_run.py
"""

import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from Generate_data import load_all_data
from Generators import NEG_MODE_NAMES
from Pipeline import run_pipeline


# =============================================================================
# CONFIG
# =============================================================================

TAG_K = 4

# Parameter sweep per method. Picks a few values around the benchmark's
# best for each method to characterise robustness.
METHOD_VARIANTS = {
    "naive": [
        {"bins": 5},
        {"bins": 10},
        {"bins": 15},
    ],
    "sax": [
        {"w": 24,  "bins": 5},
        {"w": 48,  "bins": 5},
        {"w": 144,  "bins": 5},
        {"w": 24,  "bins": 10},
        {"w": 48,  "bins": 10},
        {"w": 144,  "bins": 10},
        {"w": 24,  "bins": 15},
        {"w": 48,  "bins": 15},
        {"w": 144,  "bins": 15},
    ],
    "persist": [
        {"bins": 6},
        {"bins": 11},
        {"bins": 16},
    ],
}


# =============================================================================
# HELPERS
# =============================================================================

def _variant_label(method, params):
    param_str = "_".join(f"{k}={v}" for k, v in params.items())
    return f"{method}_{param_str}"


def _git_hash():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT
        ).decode().strip()
    except Exception:
        return "unknown"


def _run_condition(label, train_traces, test_pos, test_neg, neg_modes,
                   out_dir, method_variants, tag_k):
    """Run all (method, params) variants for one training condition."""
    results = []
    for method, variants in method_variants.items():
        for params in variants:
            vlabel = _variant_label(method, params)
            print(f"  [{label}] {vlabel} ...", flush=True)
            ta_folder = str(out_dir / "ta_images" / label / vlabel)

            result = run_pipeline(
                method=method, params=params,
                train_traces=train_traces,
                test_pos_traces=test_pos,
                test_neg_traces=test_neg,
                tag_k=tag_k,
                neg_modes=neg_modes,
                save_ta_path=ta_folder,
                ta_title=f"{label}_{vlabel}",
            )
            ov = result["overall"]
            print(f"    P={ov['precision']:.3f} R={ov['recall']:.3f} "
                  f"F1={ov['f1']:.3f} states={result['n_states']}")

            # Strip non-serializable fields before storing
            overall_clean = {k: v for k, v in ov.items()
                             if k not in ("save_path", "run_id")}

            results.append({
                "condition":     label,
                "method":        method,
                "params":        params,
                "variant_label": vlabel,
                "n_states":      result["n_states"],
                "n_edges":       result["n_edges"],
                "overall":       overall_clean,
                "per_mode":      result["per_mode"],
            })
    return results


def _save_config(out_dir, tag_k, method_variants, data):
    from collections import Counter

    lines = [
        "=" * 55,
        "Run configuration — Experiment 5.1 / 5.2",
        "=" * 55,
        "",
        f"Timestamp   : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"Git hash    : {_git_hash()}",
        f"TAG k-future: {tag_k}",
        "",
        "--- Method variants ---",
        ]
    total = 0
    for method, variants in method_variants.items():
        lines.append(f"  {method}:")
        for params in variants:
            param_str = ", ".join(f"{k}={v}" for k, v in params.items())
            lines.append(f"    - {param_str}")
            total += 1
    lines.append(f"  Total variants per condition: {total}")

    lines += [
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
# MAIN
# =============================================================================

if __name__ == "__main__":
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    out_dir = ROOT / "Data" / "Graphs" / "Metrics_clean_noisy" / timestamp
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output folder: {out_dir}\n")

    data = load_all_data()

    _save_config(out_dir, TAG_K, METHOD_VARIANTS, data)

    n_variants = sum(len(v) for v in METHOD_VARIANTS.values())
    print(f"\nTotal variants per condition: {n_variants}")
    print(f"Total pipeline runs: {n_variants * 2} (clean + noisy)\n")

    log = {
        "timestamp":       timestamp,
        "git_hash":        _git_hash(),
        "tag_k":           TAG_K,
        "method_variants": METHOD_VARIANTS,
        "n_train":         len(data["clean_train"]),
        "n_test":          len(data["clean_test"]),
        "n_neg":           len(data["neg_traces"]),
        "results":         [],
    }

    # --- Exp 5.1 — clean training -------------------------------------------
    print("=== Experiment 5.1 — Clean training ===")
    log["results"] += _run_condition(
        "clean",
        data["clean_train"], data["clean_test"],
        data["neg_traces"],  data["neg_modes"],
        out_dir, METHOD_VARIANTS, TAG_K,
    )

    # Save after each condition so a crash doesn't lose everything
    with open(out_dir / "results.json", "w") as f:
        json.dump(log, f, indent=2)
    print(f"  Saved partial results: {out_dir / 'results.json'}")

    # --- Exp 5.2 — noisy training -------------------------------------------
    print("\n=== Experiment 5.2 — Noisy training ===")
    log["results"] += _run_condition(
        "noisy",
        data["noisy_train"], data["noisy_test"],
        data["neg_traces"],  data["neg_modes"],
        out_dir, METHOD_VARIANTS, TAG_K,
    )

    with open(out_dir / "results.json", "w") as f:
        json.dump(log, f, indent=2)
    print(f"\n  Saved: {out_dir / 'results.json'}")

    print(f"\nRun complete. To generate plots:")
    print(f"  python exp_51_52_plot.py")
    print(f"or:")
    print(f"  python exp_51_52_plot.py --log {out_dir / 'results.json'}")