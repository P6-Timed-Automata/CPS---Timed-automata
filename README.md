# From Sensor Data to Timed Automata: Real-World Anomaly Detection Using TAG

Bachelor's thesis codebase, Department of Computer Science, Aalborg University, 2026.

**Authors:** Eto Kasia Kibanga, Jacob Kwasi Autzen, Li Thao Phung


This repository contains the code for the experiments reported in the thesis *From Sensor Data to Timed Automata: Real-World Anomaly Detection Using TAG*. The pipeline learns Deterministic Real-Time Automata (DRTAs) from discretized sensor time series using the TAG algorithm (Cornanguer, 2023) and evaluates them both as anomaly classifiers and as generative models for simulation in UPPAAL Stratego.

## What this project investigates

The thesis fixes TAG (k-future = 4) and treats the discretization method as the manipulated variable, comparing Equal-Width ("Naive"), SAX, and Persist along five axes (RQ1–RQ5 in §VII):

1. **Signal fidelity** — MAE between the discretized step function and the raw signal
2. **Structural cost** — TA state/edge count and training time
3. **Anomaly detection** — rejection rates against four synthetic fault modes (spikes, phase shift, stuck signal, calibration offset)
4. **Cross-domain behaviour** — synthetic vs. room-temperature vs. ECG data
5. **Behavioural fidelity** — UPPAAL verification of expected properties and statistical similarity of UPPAAL-simulated traces to real data

## Project layout

```
src/
├── TAG/                                 External TAG learner (Cornanguer, 2023)
│
├── Discretization/
│   ├── naive.py                         Equal-Width discretization
│   ├── sax.py                           SAX with global z-normalization + PAA
│   ├── persist.py                       Persist (Mörchen & Ultsch, 2005)
│   └── discretizationSetup.py           Symbol mapping, dwell-time formatting, out-of-range sentinel
│
├── DataProcessing/
│   ├── processData.py                   CSV cleaning, interval extraction, ECG R-peak detection
│   ├── negative_samples_production.py   Builds negative test traces from real data (perturbations)
│   ├── calibrate_ecg_parameter.py       Calibrates ECG spike/flat thresholds used by the UPPAAL observer
│   └── Generators.py                    Sinusoidal-trace generators (shared)
│
├── data_process_main.py                 Pre-processing driver (raw → cleaned → extracted traces)
├── naive_main.py                        §VII.C real-data pipeline — Naive
├── sax_main.py                          §VII.C real-data pipeline — SAX
├── persist_main.py                      §VII.C real-data pipeline — Persist
│
├── 1-Discretization_section/                                §VII.A — fidelity, structural cost, scaling
│   ├── benchmark_parameters_discretization_run.py           MAE / states / edges / time sweep
│   ├── Benchmarks_plot.py                                   Plots benchmark Figures
│   ├── Scaling_run.py                                       Trace-count scaling sweep 
│   ├── Scaling_plot.py                                      Plots Scaling_run.py figures
│   ├── exp_seq_characterization.py                          Symbol-distribution analysis 
│   ├── exp_seq_characterization_raw.py                      Raw-trace symbol characterization
│   ├── Generate_data.py                                     Local copy used by exp_seq_characterization
│   └── Generators.py                                        Local copy used by exp_seq_characterization
│
├── 2-Synthetic_section/                                     §VII.B — clean vs. noisy training
│   ├── Generate_data.py                                     Idempotent dataset generator (canonical)
│   ├── Generators.py                                        Sinusoidal traces + 4 anomaly modes
│   ├── Pipeline.py                                          Shared discretize → TAG → evaluate pipeline
│   ├── Metrics_clean_noisy__run.py                          Rejection-rate experiment (
│   └── Data_graphs.py                                       Plots 
│
└── 3-real_data_and_verification/                            §VII.E — UPPAAL simulation validation
    ├── statestik.py                                         KS test + ±2σ coverage vs. UPPAAL sims
    └── statestikPlot.py                                     Result table rendering 
```

## General Pipeline

See the end-to-end pipeline of Fig. 2 in the thesis:

1. **Format** raw sensor CSV → cleaned `time;value` CSV
2. **Extract intervals** — 24-hour windows for temperature data; R-peak-aligned single-beat windows for ECG
3. **Discretize** — Naive / SAX / Persist; bins, breakpoints, and normalization statistics are fitted on training data only and frozen for the test set (no leakage; §VI.C)
4. **Symbolize and format** — `[(letter, time), …]` is collapsed into TAG's dwell-time strings (`"a:300 b:120 …"`, each number being seconds the symbol persisted before changing)
5. **Learn TA** — TAG with k-future = 4 (initialize → merge → temporal refinement)
6. **Export to UPPAAL** — XML for Stratego; bin midpoints are multiplied by 100 so UPPAAL receives integer values (e.g. 22.41 °C → 2241)
7. **Evaluate** — PAR, NAR, precision, recall, F1 on held-out positive and negative traces

### Step 0 — Generate synthetic data (run once)

```bash
python src/2-Synthetic_section/Generate_data.py
# add --force to regenerate
```

Produces `Data/synthetic_data/{clean_train, clean_test, noisy_train, noisy_test, negative/{spikes, shifted, stuck, offset}}`. Generation parameters are fixed in `CONFIG` at the top of `Generate_data.py` and correspond to Table I in the thesis.

### §VII.A — Discretization benchmark 

```bash
python src/1-Discretization_section/benchmark_parameters_discretization_run.py
python src/1-Discretization_section/Benchmarks_plot.py
```

Sweeps each method over its parameter grid (Naive: bins=2..15; SAX: w ∈ {24, 48, 96, 144} × bins ∈ {5, 10, 15}; and reports per-trace MAE, state count, edge count, and training time across 20 randomly drawn traces. Runs are crash-resilient: each variant is wrapped in `try/except` and JSON logs are flushed after every cell so SLURM kills lose at most one in-progress experiment.

### §VII.A — Symbol distribution analysis

```bash
python src/1-Discretization_section/exp_seq_characterization.py
python src/1-Discretization_section/exo_seq_characterization_raw.py
```

Produces the symbol-frequency histograms used to compare alphabet usage across the three methods.

### §VII.A — Scaling 

```bash
python src/1-Discretization_section/Scaling_run.py
python src/1-Discretization_section/Scaling_plot.py
```

Measures learning time and TA complexity as a function of training-trace count (1..25, 5 replicates per cell, **disjoint** subsets so variance reflects subset selection rather than timing jitter) on both clean and noisy synthetic data. The missing-pool auto-generation falls back to `Generate_data.CONFIG` when the existing pool is too small.

### §VII.B — Clean vs. noisy synthetic experiments

```bash
python src/2-Synthetic_section/Metrics_clean_noisy__run.py
python src/2-Synthetic_section/Data_graphs.py
```

Trains TAs on clean and noisy synthetic data and computes per-mode rejection rates against the four anomaly types, plus aggregate F1, precision, and recall.

### §VII.C — Real-data pipeline 

Pre-process raw data (real data is not shipped — see the *Data* section below):

```bash
python src/data_process_main.py
```

This drives `processData.py` (cleaning + interval extraction) and `negative_samples_production.py` (building the negative test set by perturbing held-out positive traces with the four anomaly modes).

Then run the per-method real-data sweeps:

```bash
python src/naive_main.py
python src/sax_main.py
python src/persist_main.py
```

Each script sweeps trace count and bin size, writes TA PNGs to `Data/5-TaResults/`, UPPAAL XML to `Data/6-XMLOutput/`, and PAR/NAR metrics to `Data/8-LoggedData/metrics/`. Parameters at the top of each file:

- `data_type` — `"temp"` or `"ecg"`
- `room` — `"A"` (only room A is used in the thesis)
- `symbols`, `w` (SAX), `bins` (Persist) — bin and window parameters
- `k_min` / `k_max` — fixed at 4 throughout the thesis

### §VII.D — UPPAAL verification 

Verification is performed manually in UPPAAL Stratego on the XML files exported by the §VII.C scripts. The queries (`E<>`, `A[]`, `Pr[…]<>…`), the observer-model overlays used for temperature and ECG, and the TAG → UPPAAL conversion convention are documented in the thesis appendix and are not part of the Python pipeline.

The ECG observer thresholds (spike, flat) are calibrated from the training data by:

```bash
python src/DataProcessing/calibrate_ecg_parameter.py
```

### §VII.E — Statistical similarity of UPPAAL simulations 

After exporting simulation traces from UPPAAL into `Data/7-ExtractedUppaalGraphData/<method>/temp/`:

```bash
python src/3-real_data_and_verification/statestik.py
python src/3-real_data_and_verification/statestikPlot.py --method <naive|sax|persist>
```

Computes the Kolmogorov–Smirnov statistic, ±2σ coverage, and mean error between simulated and real temperature traces for every (method, bin count, trace count) configuration.

## Data

Data files are not shipped with this repository.

- **Real temperature data** — Aalborg University office monitoring dataset (Melgaard et al., Zenodo `10673763`). Only Room A is used. Place the raw CSV at `Data/1-Raw/dataset-2023-02-27_2023-12-31.csv`.
- **ECG data** — patient_100 from a public ECG corpus, provided by the supervisor. Place at `Data-marco/patient_100_ecg.csv`.
- **Synthetic data** — generated by `Generate_data.py` (Step 0). Deterministic given the seeds in `CONFIG`.

`data_process_main.py` produces the cleaned `Data/2-FormatedRawData/` and extracted-interval `Data/3-ExtractInterval/` folders that the main pipelines consume.

## Hardware

All experiments were run on the Aalborg DEIS-MCC cluster (AMD EPYC 7551, 16 GB RAM allocation).

## Requirements

Python ≥ 3.9 with `numpy`, `scipy`, `pandas`, `matplotlib`, `natsort`, `graphviz` (and the system `graphviz` binaries for TA rendering). UPPAAL Stratego ≥ 4.1 (external) is needed for verification (§VII.D) and simulation (§VII.E); the Python pipeline does not invoke it directly.

The TAG entry points raise the Python recursion limit to 50 000 (`sys.setrecursionlimit(50000)`) because TAG's recursive traversal exceeds the default 1000-frame limit at larger alphabets.

## Acknowledgments

- TAG algorithm: Lenaig Cornanguer, *Timed Automata Learning from Time Series* (PhD thesis, 2023). https://gitlab.inria.fr/lcornang/tag
- Persist implementation: adapted from Lenaig Cornanguer's reference code (GPL v2 or later). https://gitlab.inria.fr/x-LCorna/persist_discretization
- Example of how Lenaig Cornanguer tested in UPPAAL, https://gitlab.inria.fr/x-LCorna/aaai22_tag_supplementary_materials/-/tree/main/TV_logs_experiement?ref_type=heads
- Temperature dataset: Melgaard et al., Aalborg University. https://zenodo.org/records/10673763
- 