import numpy as np
import os
import matplotlib.pyplot as plt
from TAG.TALearner import TALearner
from pathlib import Path

from Discretization.persist import (
    Persist,
    discretize_traces_with_bins,
    flatten_traces_to_ts,
    get_best_bins,
    plot_and_save_breakpoints

)

from Discretization.discretizationSetup import (
    csv_to_temp_time_list,
    format_output,
    map_bins_to_symbols
)

from DataProcessing.processData import (
    format_temperature_data,
    extract_time_intervals,
    get_trace_files
)

BASE_DIR = Path(__file__).resolve().parent.parent

# PARAMETERS SETTINGS
room = "A"
discretization_method = "persist"
period = "1day"

#Parameters for Persist
break_max = 15
break_min = 2
skip_min = 4
skip_max = 4

# Parameters for TAG
k_min = 4
k_max = 4
k_increment = 2

experiment_folder = BASE_DIR / "Data" / "3-ExtractInterval" / f"{period}-experiment"

all_traces = get_trace_files(folder_path = experiment_folder)
len_traces = len(all_traces)  + 1


#for trace_nr in range(1, len_traces):

trace_nr = 4

# Prepare input for Persist
rawTraces = all_traces[:trace_nr]
data_lists = csv_to_temp_time_list(input_files=rawTraces)
ts = flatten_traces_to_ts(data_lists)

# Discretenize with Persist
p = Persist(x = ts, break_min=break_min, break_max=break_max, divergence="w", candidates="EW", skip=np.array([skip_min, skip_max]))

# best breakpoints
bins = get_best_bins(p, ts)
symbols = len(bins) - 1

print("bins", bins)
print("symbols", symbols)


# Paths
discretinize_data_path = (BASE_DIR/ "Data"/ "4-DiscretizationData"/ discretization_method / period
                        / f"{room}-{trace_nr}trace-{period}-{discretization_method}-s{symbols}-trace.txt"
                        )
save_path = (BASE_DIR / "Data"/ "Graphs" /"persistGraph"/ f"{room}-{trace_nr}trace-{period}-s{symbols}-breakpoints.png")

# Combine delays together with bins
traces = discretize_traces_with_bins(data_lists, bins)

# Vizualize breakpoints on an instance of time series
plot_and_save_breakpoints(ts,bins,save_path,show=False)

symbolic_trace, symbol_map, mapping = map_bins_to_symbols(traces, symbols, bins)

format_output(symbolic_res_list=symbolic_trace, output_path=discretinize_data_path)

# Now vary k
for k in range(k_min, k_max + 1, k_increment):

    #Paths
    title = f"{room}-{trace_nr}trace-{period}-{discretization_method}-s{symbols}-k{k}-ta"
    TA_output_path = (BASE_DIR / "Data" / "5-TaResults" / discretization_method / period)
    xml_path = (BASE_DIR / "Data" / "6-XMLOutput" / discretization_method / period
                / f"{room}-{trace_nr}trace-{period}-{discretization_method}-s{symbols}-k{k}.xml")

    # Transform to TA
    learner = TALearner(tss_path=discretinize_data_path, display=False, k=k)
    learner.ta.show(title = title, savePng = True, output_path = TA_output_path)
    learner.ta.export_ta(path=xml_path, symbol_map=symbol_map)

    print(f"Done: trace={trace_nr}, k={k}, symbols={symbols}")

    print("-------------------------------------------------------------------------------------")
