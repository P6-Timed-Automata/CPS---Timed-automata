import numpy as np
from datetime import datetime

# Load all 3 columns
data = np.genfromtxt('../../data/extracted_data-this.csv', delimiter=';', dtype=str, usecols=(0, 1, 2), encoding="utf-8-sig", skip_header=1)

# Remove rows where temperature is '#I/T'
mask = data[:, 2] != '#I/T'
data = data[mask]

ids          = data[:, 0].astype(int)
timestamps   = data[:, 1]
temperatures = data[:, 2].astype(float)

# Parse and convert timestamps to delays in seconds
def parse_ts(ts):
    return datetime.strptime(ts, "%Y-%m-%dT%H:%M:%S%z")

parsed = np.array([parse_ts(ts) for ts in timestamps])
t0     = parsed[0]
delays = np.array([(t - t0).total_seconds() for t in parsed])

# Final result: [id, delay_seconds, temperature]
result = np.column_stack((ids, delays, temperatures))

print(f"{'ID':>5} {'Time (s)':>10} {'Temperature':>12}")
print("-" * 30)
for row in result:
    print(f"{int(row[0]):>5} {row[1]:>10.0f} {row[2]:>12.5f}")

# Remove rows with None/NaN values
mask   = ~np.isnan(result.astype(float))
result = result[mask.all(axis=1)]

# Save to CSV
np.savetxt(
    'output.csv',
    result,
    delimiter=';',
    header='id;time_seconds;temperature',
    fmt=['%d', '%.0f', '%.5f'],
    comments=''          # prevents '#' being added before the header
)