import numpy as np
import os
import matplotlib.pyplot as plt
from TAG.TALearner import TALearner
from pathlib import Path

from Discretization.sax import (
    sax_discretization_multi,
    sax_discretization
)

from Discretization.discretizationSetup import (
    csv_to_temp_time_list,
    format_output,
    map_bins_to_symbols,
    preprocess_test_traces
)

from DataProcessing.processData import (
    format_temperature_data,
    extract_time_intervals,
    get_trace_files
)