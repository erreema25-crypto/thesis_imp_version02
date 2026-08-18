import pandas as pd
import numpy as np

# =========================
# LOAD DATA
# =========================
data = pd.read_csv("uav_navigation_dataset.csv")

n = len(data)

# =========================
# DEFINE SPLITS (60-20-20)
# =========================
normal_end = int(0.6 * n)
spoof_end  = int(0.8 * n)
jam_end    = n

# Add label column
data['label'] = 0

# =========================
# SPOOFING (20%)
# =========================
spoof_data = data.iloc[normal_end:spoof_end].copy()
mid = len(spoof_data) // 2

# 🔴 Drift (first half)
for i in range(mid):
    drift = 0.00001 * i
    spoof_data.iloc[i, spoof_data.columns.get_loc('latitude')]  += drift
    spoof_data.iloc[i, spoof_data.columns.get_loc('longitude')] += drift

# 🔴 Jump (second half)
jump = 0.001
spoof_data.iloc[mid:, spoof_data.columns.get_loc('latitude')]  += jump
spoof_data.iloc[mid:, spoof_data.columns.get_loc('longitude')] += jump

spoof_data['label'] = 1

# =========================
# JAMMING (20%)
# =========================
jam_data = data.iloc[spoof_end:jam_end].copy()

size = len(jam_data)
part = size // 3

# 🟠 Noise (1/3)
noise_lat = np.random.normal(0, 0.0005, part)
noise_lon = np.random.normal(0, 0.0005, part)

jam_data.iloc[:part, jam_data.columns.get_loc('latitude')]  += noise_lat
jam_data.iloc[:part, jam_data.columns.get_loc('longitude')] += noise_lon

# 🟠 Loss (1/3)
jam_data.iloc[part:2*part, jam_data.columns.get_loc('latitude')]  = np.nan
jam_data.iloc[part:2*part, jam_data.columns.get_loc('longitude')] = np.nan
jam_data.iloc[part:2*part, jam_data.columns.get_loc('altitude')]  = np.nan

# 🟠 Freeze (1/3)
for i in range(2*part, size):
    jam_data.iloc[i, jam_data.columns.get_loc('latitude')]  = jam_data.iloc[2*part-1]['latitude']
    jam_data.iloc[i, jam_data.columns.get_loc('longitude')] = jam_data.iloc[2*part-1]['longitude']

jam_data['label'] = 2

# =========================
# NORMAL DATA (60%)
# =========================
normal_data = data.iloc[:normal_end].copy()
normal_data['label'] = 0

# =========================
# COMBINE ALL
# =========================
final_data = pd.concat([normal_data, spoof_data, jam_data])

# =========================
# SAVE
# =========================
final_data.to_csv("uav_final_balanced_dataset.csv", index=False)

# =========================
# CHECK DISTRIBUTION
# =========================
print(final_data['label'].value_counts())