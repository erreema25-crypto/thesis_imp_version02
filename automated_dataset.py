import pandas as pd
import numpy as np
import random

data = pd.read_csv("uav_navigation_dataset.csv")
data['label'] = 0

n = len(data)

used_ranges = []

# =========================
# FUNCTION TO AVOID OVERLAP
# =========================
def is_overlap(start, end):
    for s, e in used_ranges:
        if start <= e and end >= s:
            return True
    return False

# =========================
# ATTACK FUNCTIONS
# =========================
def spoof_drift(start, end):
    drift_rate = random.uniform(0.000005, 0.00002)
    for i in range(start, end):
        drift = drift_rate * (i - start)
        data.loc[i, 'latitude'] += drift
        data.loc[i, 'longitude'] += drift
    data.loc[start:end, 'label'] = 1

def spoof_jump(start, end):
    jump = random.uniform(0.0005, 0.002)
    data.loc[start:end, 'latitude'] += jump
    data.loc[start:end, 'longitude'] += jump
    data.loc[start:end, 'label'] = 1

def jamming(start, end):
    size = end - start + 1
    noise = np.random.normal(0, 0.0005, size)
    data.loc[start:end, 'latitude'] += noise
    data.loc[start:end, 'longitude'] += noise
    data.loc[start:end, 'label'] = 2

# =========================
# FORCE ALL ATTACK TYPES
# =========================
attacks = ['spoof_drift', 'spoof_jump', 'jamming']

for attack in attacks:
    while True:
        start = random.randint(100, n-500)
        duration = random.randint(100, 300)
        end = start + duration
        
        if not is_overlap(start, end):
            used_ranges.append((start, end))
            break
    
    if attack == 'spoof_drift':
        spoof_drift(start, end)
    elif attack == 'spoof_jump':
        spoof_jump(start, end)
    elif attack == 'jamming':
        jamming(start, end)

# =========================
# SAVE
# =========================
data.to_csv("uav_auto_attacked_automated01.csv", index=False)

print("✅ Done!")

# Check labels
print(data['label'].value_counts())