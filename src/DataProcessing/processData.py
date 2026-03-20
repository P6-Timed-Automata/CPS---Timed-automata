import numpy as np
from datetime import datetime

def format_temperature_data(input_file, output_file):
    # Load data
    data = np.genfromtxt(
        input_file,
        delimiter=';',
        dtype=str,
        usecols=(0, 1, 2),
        encoding="utf-8-sig",
        skip_header=1
    )

    # Remove invalid temperature rows
    mask = (data[:, 2] != '#I/T') & (data[:, 2] != '')
    data = data[mask]

    # Extract columns
    ids = data[:, 0].astype(int)
    timestamps = data[:, 1]
    temperatures = data[:, 2].astype(float)

    # Parse timestamps
    def parse_ts(ts):
        return datetime.strptime(ts, "%Y-%m-%dT%H:%M:%S%z")

    parsed = np.array([parse_ts(ts) for ts in timestamps])

    # Compute delays relative to first timestamp
    t0 = parsed[0]
    delays = np.array([(t - t0).total_seconds() for t in parsed])

    # Combine into final structure
    result = np.column_stack((ids, delays, temperatures))

    # Remove any NaN rows (safety)
    result = result[~np.isnan(result.astype(float)).any(axis=1)]

    # Save to CSV
    np.savetxt(
        output_file,
        result,
        delimiter=';',
        header='id;time_seconds;temperature',
        fmt=['%d', '%.0f', '%.5f'],
        comments=''
    )

    print(f"Saved formatted data to {output_file}")

def extract_time_interval_seconds(input_file
                                  , output_file, start_sec, end_sec):
    # Load data (skip header)
    data = np.genfromtxt(
        input_file,
        delimiter=';',
        dtype=str,
        skip_header=1
    )

    # Extract columns
    times = data[:, 1].astype(int)
    temps = data[:, 2].astype(float)

    # Filter interval
    mask = (times >= start_sec) & (times <= end_sec)

    # Rebase time so start_sec becomes 0
    rebased_times = times[mask] - start_sec

    filtered = np.column_stack((
        rebased_times,
        temps[mask]
    ))

    # Save to CSV
    np.savetxt(
        output_file,
        filtered,
        delimiter=';',
        fmt=['%d', '%.5f'],
        header="time_seconds;temperature",
        comments=''
    )

    print(f"Saved {len(filtered)} rows to {output_file}")

input_raw_file ='../../data/extracted_data-this.csv'
output_raw_file = 'formated_raw_data.csv'

#format_temperature_data(input_raw_file,output_raw_file)


#second time sequence
input_file = "formated_raw_data.csv"
output_file ="formated_data2.csv"
start_sec = 86400
# 5 timer =
end_sec = 104400

extract_time_interval_seconds(input_file, output_file, start_sec, end_sec)

