import numpy as np
import string
import json



def csv_to_temp_time_list(input_file):

    all_results = []
    for input_file in input_files:
         # Load data
        data = np.genfromtxt(
            input_file,
            delimiter=';',
            dtype=str,
            skip_header=1
        )

        # Extract columns
        times = data[:, 0].astype(int)
        temps = data[:, 1].astype(float)

        # Build list of (temperature, time)
        result = [(temp, time) for temp, time in zip(temps, times)]

        all_results.append(results)


    print("tranformed data to a list")

    return result


def format_output(symbolic_res_list):
    lines = []

    for symbolic_res in symbolic_res_list:
        line = " ".join(f"{s}:{v}" for s, v in symbolic_res)
        lines.append(line)

    output = "\n".join(lines)

    with open("output.txt", "w") as f:
        f.write(output)

    print("File saved")

def map_bins_to_symbols(result, k):
    # Create symbols: a, b, c, ...
    symbols = list(string.ascii_lowercase)

    if k > len(symbols):
        raise ValueError("k too large (max 26 supported with simple letters)")

    # Create mapping: 0->'a', 1->'b', ...
    mapping = {i: symbols[i] for i in range(k)}

    # Apply mapping
    symbolic_result = [(mapping[int(label)], int(time)) for label, time in result]

    return symbolic_result, mapping


# Compute midpoints: symbol -> rounded midpoint value
symbol_map = {
    mapping[i]: round((bins[i] + bins[i + 1]) / 2*100)
    for i in range(k)
}

with open('symbol_map.json', 'w') as f:
    json.dump(symbol_map, f)


# Check

input_files = [
    '../dataProcessing/formated_data.csv',
    '../dataProcessing/formated_data2.csv'
]

data_lists = csv_to_temp_time_list(input_files)

print(data_lists)