# CPS - Timed-automata
Learning behavioral models of heating systems from sensor time series using Timed Automata, with applications in anomaly detection and system understanding.

Pipeline overview:
raw CSV - format and clean - extract trraces - discretizise - Learn TA - XML / PNG

The pipeline has four stages:

Stage 1 - Data processing
Function:
format_temperature_data(input_file, output_file, col)

Reads the raw multi-room CSV, selects one room's temperature column, 
removes invalid entries (#I/T, #N/A, empty, NaN), sorts by timestamp,
and writes a cleaned semicolon-delimited CSV.
- Output columns: id;timestamp;temperature
- Run once per room; skipped automatically if the output already exists.

Function:
extract_time_intervals(input_file, output_folder, output_prefix, trace_days=1)

Splits a formatted room CSV into fixed-length traces of trace_days consecutive calendar days.
Days with a non-modal row count (incomplete readings) are dropped.
Consecutive-day runs are detected and windowed with a non-overlapping stride of trace_days.

- Output: one CSV per trace with columns time_seconds;temperature, 
where time_seconds is relative to the start of that trace.
- File naming: {prefix}-tid{n}.csv

Function: 
get_trace_files(folder_path, extension=".csv", max_files=None)

Returns a sorted list of file paths from a folder, optionally capped at max_files. 
Used to control how many traces are fed into each pipeline run.

Stage 2 - Shared I/O Utilities (Discretization/discretizationSetup.py)

Function:
csv_to_temp_time_list(input_files)

Loads a list of trace CSVs and returns a list of traces, 
where each trace is a list of (temperature, time_seconds) tuples.

Function:
map_bins_to_symbols(result, s, bins)

Maps integer bin indices to lowercase letters (a, b, c, …). Returns:

- symbolic_results: list of traces as (letter, time) tuples
- symbol_map: {letter: midpoint_temperature * 100} — used for TA XML export
- mapping: {bin_index: letter}

Function:
format_output(symbolic_res_list, output_path)

Writes symbolic traces to a .txt file in the format expected by TALearner.
Each line represents one trace: a:0 b:3600 c:7200 …

Stage 3 — Discretization
Three interchangeable methods are available. All produce the same output format: 
a list of traces as (bin_index, time_seconds) tuples, suitable for map_bins_to_symbols.


Equal-Width / Naive (Discretization/naive.py)

Function:
equal_width_discretization(traces, k)
Computes a global value range across all traces and partitions it into k equal-width 
bins using np.linspace. Each sample is assigned to a bin with np.digitize.

- Parameter: k — number of bins (symbols)
Pros: Simple, deterministic, no distributional assumptions.
Cons: Sensitive to outliers; bins may have very unequal occupancy.


SAX — Symbolic Aggregate approXimation (Discretization/sax.py and main_sax.py)

Function:
sax_discretization_multi(data_lists, w, k)
Applies full SAX to a list of traces:

- Z-normalise each trace independently.
- PAA (Piecewise Aggregate Approximation): reduce to w segment means.
- Quantise against Gaussian breakpoints (norm.ppf) into k symbols.
- Parameters: w — number of PAA segments (controls temporal resolution); k — alphabet size.
Pros: Normalization makes traces comparable regardless of absolute temperature level.
Cons: Z-normalization discards absolute temperature information, requires choosing w and k.

Helper function:
build_temp_symbol_map(symbol_map, bins, mean, std)
Back-projects z-score bin midpoints to Celsius for use in TA XML export.


Persist (Discretization/persist.py)

function:
Persist(x, break_min, break_max, divergence, skip, candidates)
Data-driven breakpoint selection that maximises a persistence score
— a measure of how much each state's self-transition probability deviates
from its marginal probability. Iteratively adds breakpoints from a candidate set
(equal-width "EW" or equal-frequency "EF") using either Wasserstein ("w") or KL divergence.

- Parameters: break_min/break_max — search range for number of breakpoints
- skip — exclude candidates near the data edges; divergence — distance measure.
Pros: Breakpoints are chosen to respect the temporal structure of the data (state persistence), not just value distribution.
Cons: Computationally heavier; requires tuning the search range.

helper functions (persist):
get_best_bins(persist_obj, ts)
Selects the breakpoint set with the highest persistence score and prepends/appends the global min/max as outer edges.

flatten_traces_to_ts(data_lists)
Concatenates all trace values into a single column vector for use as Persist input.

discretize_traces_with_bins(traces, bins)
Applies a fixed bin array (from Persist) to a list of traces using np.digitize.

plot_and_save_breakpoints(ts, bins, save_path, show)
Plots the raw time series with horizontal lines at each breakpoint and saves to disk.

Stage 4 — TA Learning (TAG/TALearner.py)
TALearner is an externally produced library. It reads the symbolic trace .txt file 
produced by format_output and learns a Timed Automaton.
Key parameters:

- tss_path — path to symbolic trace file
- k — lookahead horizon for merging TA states
- display — whether to show the TA interactively

Outputs per run:

learner.ta.show(...) — saves a PNG diagram of the TA
learner.ta.export_ta(...) — saves an XML representation with real temperature values 
via symbol_map


Visualisation (GraphGeneration/graphs.py)

Function:
plot_traces(trace_folder, output_folder)
Generates one PNG per trace (continuous temperature vs. time in hours)
and a single overlay plot of all traces in the folder.

Function:
plot_discretized_traces(discretized_traces, output_folder, bins, mapping)
Same structure as plot_traces but for discretized (step-function) traces.
Y-axis ticks are relabelled with symbols if mapping is provided.