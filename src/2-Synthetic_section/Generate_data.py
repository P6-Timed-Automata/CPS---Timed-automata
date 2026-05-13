"""
generate_all_data.py
====================
Run this ONCE before running any experiment.

Generates and saves all synthetic datasets needed by every experiment:

  Data/synthetic_data/
    clean_train/        positive traces, low noise  (training set)
    clean_test/         positive traces, low noise  (held-out test set)
    noisy_train/        positive traces, high noise (training set)
    noisy_test/         positive traces, high noise (held-out test set)
    negative/
        spikes/         anomaly mode 0
        shifted/        anomaly mode 1
        stuck/          anomaly mode 2
        offset/         anomaly mode 3
    data_config.json    parameters used — read by experiments for verification

All experiments load from these folders instead of regenerating.
This guarantees identical data across all runs.

Usage:
    python generate_all_data.py           # generate if not already present
    python generate_all_data.py --force   # regenerate and overwrite

Script location: src/2-Synthetic_section/generate_all_data.py
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

# ============================================================
# PATHS
# ============================================================

_HERE = Path(__file__).resolve()
ROOT  = _HERE.parent.parent.parent     # CPS---Timed-automata/
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(_HERE.parent))  # for local Generators import

DATA_ROOT = ROOT / "Data" / "synthetic_data"

# ============================================================
# CENTRAL DATA PARAMETERS
# All experiments must use data generated with these settings.
# Change here, then re-run this script — never change per-experiment.
# ============================================================

CONFIG = {
    # Dataset sizes
    "n_train":          140,    # positive traces used for training
    "n_test":           60,     # positive traces held out for testing
    "n_neg":            60,     # negative traces total (15 per mode x 4 modes)

    # Random seeds — fix these so data is reproducible
    "seed_clean":       42,
    "seed_noisy":       43,     # different seed so noisy != clean + noise
    "seed_negative":    99,

    # Clean training data — tight variation, low noise
    "clean": {
        "base_temp":        22.0,
        "amplitude":        3.0,
        "base_temp_std":    0.0,    # no inter-trace mean variation
        "amplitude_std":    0.1,    # small amplitude variation
        "phase_std_h":      0.1,    # small phase variation
        "noise_std":        0.05,   # very low sensor noise
    },

    # Noisy training data — high variation, high noise
    # Designed to stress-test the pipeline under realistic conditions
    "noisy": {
        "base_temp":        22.0,
        "amplitude":        3.0,
        "base_temp_std":    2.0,    # large inter-trace mean variation
        "amplitude_std":    1.0,    # large amplitude variation
        "phase_std_h":      1.0,    # large phase variation
        "noise_std":        0.3,    # realistic sensor noise
    },
}

# ============================================================
# IMPORT GENERATORS
# ============================================================

try:
    from Generators import (
        generate_trace_set,
        generate_negative_set,
        save_traces,
        save_traces_by_mode,
        NEG_MODE_NAMES,
    )
except ImportError:
    # Fall back to shared.generators if Generators.py is not local
    from Generators import (
        generate_trace_set,
        generate_negative_set,
        save_traces,
        save_traces_by_mode,
        NEG_MODE_NAMES,
    )


# ============================================================
# HELPERS
# ============================================================

def _folder_has_data(folder: Path) -> bool:
    """True if folder exists and contains at least one CSV."""
    return folder.exists() and len(list(folder.glob("*.csv"))) > 0


def _neg_folder_has_data(folder: Path) -> bool:
    """True if all four mode subfolders exist and contain CSVs."""
    return all(
        (folder / name).exists() and
        len(list((folder / name).glob("*.csv"))) > 0
        for name in NEG_MODE_NAMES.values()
    )


def _print_folder_summary(label: str, folder: Path):
    if folder.exists():
        csvs = list(folder.glob("*.csv"))
        print(f"    {label}: {len(csvs)} traces in {folder.relative_to(ROOT)}")
    else:
        print(f"    {label}: NOT FOUND at {folder.relative_to(ROOT)}")


def _print_neg_summary(folder: Path):
    if folder.exists():
        for name in NEG_MODE_NAMES.values():
            sub  = folder / name
            csvs = list(sub.glob("*.csv")) if sub.exists() else []
            print(f"    negative/{name}: {len(csvs)} traces")
    else:
        print(f"    negative: NOT FOUND at {folder.relative_to(ROOT)}")


# ============================================================
# GENERATION
# ============================================================

def generate_all(force: bool = False):
    n_total = CONFIG["n_train"] + CONFIG["n_test"]

    # ---- Check existing data ------------------------------------------------
    folders_exist = {
        "clean_train":  _folder_has_data(DATA_ROOT / "clean_train"),
        "clean_test":   _folder_has_data(DATA_ROOT / "clean_test"),
        "noisy_train":  _folder_has_data(DATA_ROOT / "noisy_train"),
        "noisy_test":   _folder_has_data(DATA_ROOT / "noisy_test"),
        "negative":     _neg_folder_has_data(DATA_ROOT / "negative"),
    }

    if all(folders_exist.values()) and not force:
        print("All datasets already present. Use --force to regenerate.\n")
        print("Current data summary:")
        _print_folder_summary("clean_train", DATA_ROOT / "clean_train")
        _print_folder_summary("clean_test",  DATA_ROOT / "clean_test")
        _print_folder_summary("noisy_train", DATA_ROOT / "noisy_train")
        _print_folder_summary("noisy_test",  DATA_ROOT / "noisy_test")
        _print_neg_summary(DATA_ROOT / "negative")
        return

    if force:
        print("--force: regenerating all datasets.\n")
    else:
        missing = [k for k, v in folders_exist.items() if not v]
        print(f"Missing datasets: {missing}. Generating...\n")

    # ---- Generate clean traces ----------------------------------------------
    print(f"Generating {n_total} clean traces (seed={CONFIG['seed_clean']})...")
    all_clean = generate_trace_set(
        n_traces = n_total,
        seed     = CONFIG["seed_clean"],
        **CONFIG["clean"],
    )
    clean_train = all_clean[:CONFIG["n_train"]]
    clean_test  = all_clean[CONFIG["n_train"]:]

    save_traces(clean_train, DATA_ROOT / "clean_train", prefix="clean_train")
    save_traces(clean_test,  DATA_ROOT / "clean_test",  prefix="clean_test")

    # ---- Generate noisy traces ----------------------------------------------
    print(f"\nGenerating {n_total} noisy traces (seed={CONFIG['seed_noisy']})...")
    all_noisy = generate_trace_set(
        n_traces = n_total,
        seed     = CONFIG["seed_noisy"],
        **CONFIG["noisy"],
    )
    noisy_train = all_noisy[:CONFIG["n_train"]]
    noisy_test  = all_noisy[CONFIG["n_train"]:]

    save_traces(noisy_train, DATA_ROOT / "noisy_train", prefix="noisy_train")
    save_traces(noisy_test,  DATA_ROOT / "noisy_test",  prefix="noisy_test")

    # ---- Generate negative traces -------------------------------------------
    print(f"\nGenerating {CONFIG['n_neg']} negative traces "
          f"(seed={CONFIG['seed_negative']}, "
          f"{CONFIG['n_neg'] // 4} per mode)...")
    neg_traces, neg_modes = generate_negative_set(
        n_traces = CONFIG["n_neg"],
        seed     = CONFIG["seed_negative"],
    )
    save_traces_by_mode(
        neg_traces, neg_modes,
        DATA_ROOT / "negative",
        prefix="neg",
        )

    # ---- Save config --------------------------------------------------------
    config_path = DATA_ROOT / "data_config.json"
    config_record = {
        "generated_at": datetime.now().isoformat(),
        "config":       CONFIG,
        "mode_names":   NEG_MODE_NAMES,
        "folder_structure": {
            "clean_train":  str(DATA_ROOT / "clean_train"),
            "clean_test":   str(DATA_ROOT / "clean_test"),
            "noisy_train":  str(DATA_ROOT / "noisy_train"),
            "noisy_test":   str(DATA_ROOT / "noisy_test"),
            "negative":     str(DATA_ROOT / "negative"),
        },
    }
    DATA_ROOT.mkdir(parents=True, exist_ok=True)
    with open(config_path, "w") as f:
        json.dump(config_record, f, indent=2)

    # ---- Summary ------------------------------------------------------------
    print("\n" + "=" * 55)
    print("Data generation complete.")
    print("=" * 55)
    _print_folder_summary("clean_train", DATA_ROOT / "clean_train")
    _print_folder_summary("clean_test",  DATA_ROOT / "clean_test")
    _print_folder_summary("noisy_train", DATA_ROOT / "noisy_train")
    _print_folder_summary("noisy_test",  DATA_ROOT / "noisy_test")
    _print_neg_summary(DATA_ROOT / "negative")
    print(f"\n  Config saved -> {config_path.relative_to(ROOT)}")
    print("\nAll experiments can now load from Data/synthetic_data/")


# ============================================================
# LOADER HELPERS  (import these in experiments instead of
#                  calling generate_* directly)
# ============================================================

def load_all_data():
    """
    Load all datasets from disk. Call this at the top of each experiment
    instead of calling generate_trace_set / generate_negative_set.

    Returns
    -------
    dict with keys:
        clean_train, clean_test  : list of (times, temps) pairs
        noisy_train, noisy_test  : list of (times, temps) pairs
        neg_traces               : list of (times, temps) pairs
        neg_modes                : list of int  (0-3 per trace)
    """
    try:
        from Generators import load_traces, load_traces_by_mode
    except ImportError:
        from Generators import load_traces, load_traces_by_mode

    # Verify data exists before loading
    config_path = DATA_ROOT / "data_config.json"
    if not config_path.exists():
        raise FileNotFoundError(
            "data_config.json not found. "
            "Run generate_all_data.py first."
        )

    print("Loading datasets from disk...")
    data = {
        "clean_train": load_traces(DATA_ROOT / "clean_train"),
        "clean_test":  load_traces(DATA_ROOT / "clean_test"),
        "noisy_train": load_traces(DATA_ROOT / "noisy_train"),
        "noisy_test":  load_traces(DATA_ROOT / "noisy_test"),
    }
    data["neg_traces"], data["neg_modes"] = load_traces_by_mode(
        DATA_ROOT / "negative"
    )

    print(f"  clean_train : {len(data['clean_train'])} traces")
    print(f"  clean_test  : {len(data['clean_test'])} traces")
    print(f"  noisy_train : {len(data['noisy_train'])} traces")
    print(f"  noisy_test  : {len(data['noisy_test'])} traces")
    print(f"  negatives   : {len(data['neg_traces'])} traces "
          f"across {len(set(data['neg_modes']))} modes")
    return data


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate all synthetic data for the TA experiments."
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Regenerate and overwrite existing data."
    )
    args = parser.parse_args()
    generate_all(force=args.force)