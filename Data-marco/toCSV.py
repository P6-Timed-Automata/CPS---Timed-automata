# import json
# import pandas as pd
#
# with open("out_storm_full.json", "r", encoding="utf-8") as f:
#     data = json.load(f)["data"]
#
# df = pd.DataFrame(data)
#
# df["Datotid"] = pd.to_datetime(df["Datotid"])
# df["time_delta_seconds"] = (df["Datotid"] - df["Datotid"].iloc[0]).dt.total_seconds()
#
# df = df[["Datotid", "time_delta_seconds", "Havn Vandstand", "Fjord Vandstand"]]
#
# df.to_csv("storm.csv", index=False)
#
# # Load JSON
# with open("input.json", "r", encoding="utf-8") as f:
#     raw = json.load(f)
#
# data = raw["data"]
#
# # Parse timestamps (ISO 8601 with timezone)
# t0 = datetime.fromisoformat(data[0]["time"])
#
# rows = []
#
# for entry in data:
#     t = datetime.fromisoformat(entry["time"])
#
#     time_delta = (t - t0).total_seconds()
#
#     rows.append([
#         entry["time"],
#         time_delta,
#         entry["room_temp"]
#     ])
#
# # Write CSV
# with open("storm.csv", "w", newline="", encoding="utf-8") as f:
#     writer = csv.writer(f)
#
#     writer.writerow([
#         "time",
#         "time_delta_seconds",
#         "room_temp"
#     ])
#
#     writer.writerows(rows)
#
# import json
# import pandas as pd
#
# # Load JSON
# with open("data_tmv23.json", "r", encoding="utf-8") as f:
#     data = json.load(f)["data"]
#
# # Create DataFrame
# df = pd.DataFrame(data)
#
# # Parse time column (ISO 8601 with timezone)
# df["time"] = pd.to_datetime(df["time"])
#
# # Compute time delta from first measurement
# df["time_delta_seconds"] = (df["time"] - df["time"].iloc[0]).dt.total_seconds()
#
# # Select relevant columns
# df = df[["time", "time_delta_seconds", "room_temp"]]
#
# # Save to CSV
# df.to_csv("room_data_2.csv", index=False)

import json
import pandas as pd

# Load JSON
with open("data_patient_100.json", "r", encoding="utf-8") as f:
    data = json.load(f)["data"]

# Create DataFrame
df = pd.DataFrame(data)

# Select required columns
df = df[["ts", "MLII", "V5"]]

# Save to CSV
df.to_csv("patient_100_ecg.csv", index=False)
