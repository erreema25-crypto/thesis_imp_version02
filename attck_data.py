import pandas as pd
import numpy as np

# =========================
# FILE PATHS
# =========================
input_file = r"D:\d drive data\New Zealand\College notes\Thesis proposal\autopilot drones\Spoofing papers\Thesis file docs\uav_spoofing_Detection\Navigation_dataset_kaggle\uav_navigation_dataset.csv"

output_file = r"D:\d drive data\New Zealand\College notes\Thesis proposal\autopilot drones\Spoofing papers\Thesis file docs\uav_spoofing_Detection\Navigation_dataset_kaggle\uav_attacked_dataset.csv"

# =========================
# LOAD DATA
# =========================
data = pd.read_csv(input_file)

# =========================
# FIX COLUMN NAMES (AUTO)
# =========================
print("Columns in dataset:", data.columns)

# Handle typo automatically
if 'latitde' in data.columns:
    data.rename(columns={'latitde': 'latitude'}, inplace=True)

# Ensure required columns exist
required_cols = ['latitude', 'longitude', 'altitude']
for col in required_cols:
    if col not in data.columns:
        raise Exception(f"❌ Column '{col}' not found in dataset!")

# =========================
# ADD LABEL COLUMN
# =========================
data['label'] = 0   # 0=normal, 1=spoofing, 2=jamming

# =========================
# PARAMETERS
# =========================
n = len(data)

spoof_start = int(0.2 * n)
spoof_end   = int(0.4 * n)
jump_point  = int(0.5 * n)

jam_start   = int(0.6 * n)
jam_end     = int(0.8 * n)

freeze_start = int(0.85 * n)
freeze_end   = int(0.95 * n)

drift_rate = 0.00001
jump_value = 0.001
noise_std  = 0.0005

# =========================
# SPOOFING: GRADUAL DRIFT
# =========================
for i in range(spoof_start, spoof_end):
    drift = drift_rate * (i - spoof_start)
    data.loc[i, 'latitude']  += drift
    data.loc[i, 'longitude'] += drift
    data.loc[i, 'label'] = 1

# =========================
# SPOOFING: SUDDEN JUMP
# =========================
data.loc[jump_point:, 'latitude']  += jump_value
data.loc[jump_point:, 'longitude'] += jump_value
data.loc[jump_point:, 'label'] = 1

# =========================
# JAMMING: SIGNAL LOSS
# =========================
data.loc[jam_start:jam_end, ['latitude','longitude','altitude']] = np.nan
data.loc[jam_start:jam_end, 'label'] = 2

# =========================
# JAMMING: NOISE
# =========================
noise_lat = np.random.normal(0, noise_std, jam_end - jam_start)
noise_lon = np.random.normal(0, noise_std, jam_end - jam_start)

data.loc[jam_start:jam_end, 'latitude']  += noise_lat
data.loc[jam_start:jam_end, 'longitude'] += noise_lon

# =========================
# JAMMING: FREEZE
# =========================
data.loc[freeze_start:freeze_end, 'latitude']  = data.loc[freeze_start-1, 'latitude']
data.loc[freeze_start:freeze_end, 'longitude'] = data.loc[freeze_start-1, 'longitude']
data.loc[freeze_start:freeze_end, 'label'] = 2

# =========================
# SAVE FILE
# =========================
data.to_csv(output_file, index=False)

print("✅ SUCCESS: Attack dataset created!")
print("📁 Saved at:", output_file)