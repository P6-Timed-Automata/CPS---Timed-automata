"""
exp_51_52_run.py
================
Runs Experiment 5.1 (clean training) and Experiment 5.2 (noisy training)
across one or more parameter variants per method.

Writes results.json to a timestamped folder. The companion plotter
(exp_51_52_plot.py) reads that file and produces all tables/figures
without re-running pipelines.

Usage:
    python exp_51_52_run.py
"""

import json
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path

sys.setrecursionlimit(50000)

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

# One variant per method (the values that survived the benchmark).
METHOD_VARIANTS = {
    "naive":   [{"bins": 10}],
    "sax":     [{"w": 48, "bins": 10}],
    "persist": [{"bins": 11}],
}


# =============================================================================
# HELPERS
# =============================================================================

def _variant_label(method, params):
    param_str = "_".join(f"{k}={v}" for k, v in params.items())
    return f"{method}_{param_str}"


def _run_condition(label, train_traces, test_pos, test_neg, neg_modes,
                   out_dir, method_variants, tag_k):
    """Run every (method, params) variant once on the given training set,
    against the shared positive/negative test sets. Returns one result dict
    per variant. Failed variants get status='failed' rather than raising,
    so a single bad variant doesn't abort the sweep."""
    results = []
    for method, variants in method_variants.items():
        for params in variants:
            vlabel = _variant_label(method, params)
            print(f"  [{label}] {vlabel} ...", flush=True)
            ta_folder = str(out_dir / "ta_images" / label / vlabel)

            try:
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

                # Strip non-essential pipeline-internal fields from `overall`.
                overall_clean = {k: v for k, v in ov.items()
                                 if k not in ("save_path", "run_id")}

                results.append({
                    "condition":     label,
                    "method":        method,
                    "params":        params,
                    "variant_label": vlabel,
                    "status":        "ok",
                    "n_states":      result["n_states"],
                    "n_edges":       result["n_edges"],
                    "overall":       overall_clean,
                    "per_mode":      result["per_mode"],
                })

            except Exception as e:
                error_type = type(e).__name__
                print(f"    FAILED ({error_type}): {str(e)[:200]}", flush=True)
                results.append({
                    "condition":     label,
                    "method":        method,
                    "params":        params,
                    "variant_label": vlabel,
                    "status":        "failed",
                    "error_type":    error_type,
                    "error_msg":     str(e),
                })
    return results


def _save_config(out_dir, tag_k, method_variants, data):
    lines = [
        "=" * 55,
        "Run configuration — Experiment 5.1 / 5.2",
        "=" * 55,
        "",
        f"Timestamp   : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
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
    for mode_int, count in sorted(Counter(data["neg_modes"]).items()):
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
    print(f"Total pipeline runs:          {n_variants * 2} (clean + noisy)\n")

    log = {
        "timestamp": timestamp,
        "results":   [],
    }

    # Run clean (5.1) then noisy (5.2). Save after each condition so a
    # crash in the second sweep doesn't lose the first.
    for cond in ["clean", "noisy"]:
        exp_num = "5.1" if cond == "clean" else "5.2"
        print(f"=== Experiment {exp_num} — {cond.capitalize()} training ===")
        log["results"] += _run_condition(
            cond,
            data[f"{cond}_train"], data[f"{cond}_test"],
            data["neg_traces"], data["neg_modes"],
            out_dir, METHOD_VARIANTS, TAG_K,
        )
        with open(out_dir / "results.json", "w") as f:
            json.dump(log, f, indent=2)
        print(f"  Saved: {out_dir / 'results.json'}\n")

    print("Run complete. To generate plots:")
    print("  python exp_51_52_plot.py")
    print("or:")
    print(f"  python exp_51_52_plot.py --log {out_dir / 'results.json'}")