import pandas as pd
import matplotlib.pyplot as plt
import os


output_dir = "output_plots"
os.makedirs(output_dir, exist_ok=True)



# =======================================================
# 1. ROOM DATA (first 1 hour)
# =========================================================
room = pd.read_csv("room_data_2.csv")

room_1day = room[room["time_delta_seconds"] <= 86400]

plt.figure()
plt.plot(room_1day["time_delta_seconds"], room_1day["room_temp"])
plt.xlabel("Delta time (s)")
plt.ylabel("Room temperature")
plt.title("Room temperature (first 1 hour)")
plt.grid()
plt.tight_layout()

plt.savefig(os.path.join(output_dir, "room_temp_1day.png"))
plt.close()


# =========================================================
# 2. STORM DATA (first 1 hour)
# =========================================================
storm = pd.read_csv("storm.csv")

storm_1day = storm[storm["time_delta_seconds"] <= 86400]

# Havn Vandstand
plt.figure()
plt.plot(storm_1day["time_delta_seconds"], storm_1day["Havn Vandstand"])
plt.xlabel("Delta time (s)")
plt.ylabel("Havn Vandstand")
plt.title("Havn Vandstand (first 1 hour)")
plt.grid()
plt.tight_layout()

plt.savefig(os.path.join(output_dir, "havn_vandstand_1day.png"))
plt.close()

# Fjord Vandstand
plt.figure()
plt.plot(storm_1day["time_delta_seconds"], storm_1day["Fjord Vandstand"])
plt.xlabel("Delta time (s)")
plt.ylabel("Fjord Vandstand")
plt.title("Fjord Vandstand (first 1 hour)")
plt.grid()
plt.tight_layout()

plt.savefig(os.path.join(output_dir, "fjord_vandstand_1day.png"))
plt.close()


# =========================================================
# 3. ECG DATA (first 1 hour)
# =========================================================
ecg = pd.read_csv("patient_100_ecg.csv")

# Prefer seconds if available
time_col = "t_seconds" if "t_seconds" in ecg.columns else "ts"

ecg_10m = ecg[ecg[time_col] <= 600000000 ]

# MLII
plt.figure()
plt.plot(ecg_10m[time_col], ecg_10m["MLII"])
plt.xlabel("Time (s)")
plt.ylabel("MLII")
plt.title("ECG MLII")
plt.grid()
plt.tight_layout()

plt.savefig(os.path.join(output_dir, "ecg_mlii.png"))
plt.close()

# V5
plt.figure()
plt.plot(ecg_10m[time_col], ecg_10m["V5"])
plt.xlabel("Time (s)")
plt.ylabel("V5")
plt.title("ECG V5 ")
plt.grid()
plt.tight_layout()

plt.savefig(os.path.join(output_dir, "ecg_v5.png"))
plt.close()