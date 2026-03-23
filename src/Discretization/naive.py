import numpy as np



def equal_width_discretization(data, k):
    # Split into separate arrays
    values = np.array([d[0] for d in data])
    times = np.array([d[1] for d in data])

    # Equal-width bins
    min_val = np.min(values)
    max_val = np.max(values)
    bins = np.linspace(min_val, max_val, k + 1)

    # Assign bins
    labels = np.digitize(values, bins) - 1
    labels[labels == k] = k - 1  # fix max edge


    # Print bin intervals
    print("Bin intervals:")
    for i in range(k):
        print(f"Bin {i}: [{bins[i]} → {bins[i+1]})")

    print("\nAssignments:")
    for v, l in zip(values, labels):
        print(f"Value {v} → Bin {l}")

    # Recombine (discretized value + original time)
    result = list(zip(labels, times))

    result = [(int(l), int(t)) for l, t in zip(labels, times)]

    return result, bins


data = [(10, 1), (15, 2), (8, 3), (20, 4)]

res, bins = equal_width_discretization(data, k=3)

print("Bins:", bins)
print("Result:", res)