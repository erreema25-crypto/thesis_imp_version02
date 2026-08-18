import pandas as pd
import numpy as np

# ============================================================
# 1. LOAD ATTACKED DATASET
# ============================================================

input_file = "uav_final_balanced_dataset.csv"
output_file = "uav_ekf_residual_dataset.csv"

data = pd.read_csv(input_file)

print("Dataset shape:", data.shape)
print(data["label"].value_counts())


# ============================================================
# 2. SELECT REQUIRED FEATURES
# ============================================================

# Measurement vector:
# GNSS position = latitude, longitude, altitude

measurement_cols = [
    "latitude",
    "longitude",
    "altitude"
]

# IMU acceleration
acc_cols = [
    "imu_acc_x",
    "imu_acc_y",
    "imu_acc_z"
]


# ============================================================
# 3. CONVERT DATA TO NUMPY
# ============================================================

lat = data["latitude"].to_numpy(dtype=float)
lon = data["longitude"].to_numpy(dtype=float)
alt = data["altitude"].to_numpy(dtype=float)

acc_x = data["imu_acc_x"].to_numpy(dtype=float)
acc_y = data["imu_acc_y"].to_numpy(dtype=float)
acc_z = data["imu_acc_z"].to_numpy(dtype=float)


# ============================================================
# 4. HANDLE MISSING SENSOR VALUES
# ============================================================

# Important because your jamming attack deliberately
# creates NaN values.

lat = pd.Series(lat).interpolate().bfill().ffill().to_numpy()
lon = pd.Series(lon).interpolate().bfill().ffill().to_numpy()
alt = pd.Series(alt).interpolate().bfill().ffill().to_numpy()

acc_x = pd.Series(acc_x).interpolate().bfill().ffill().to_numpy()
acc_y = pd.Series(acc_y).interpolate().bfill().ffill().to_numpy()
acc_z = pd.Series(acc_z).interpolate().bfill().ffill().to_numpy()


# ============================================================
# 5. EKF PARAMETERS
# ============================================================

dt = 1.0

# State:
# x = [latitude, longitude, altitude,
#      velocity_lat, velocity_lon, velocity_alt]

state_dim = 6

x = np.zeros(state_dim)

# Initial state
x[0] = lat[0]
x[1] = lon[0]
x[2] = alt[0]

# Initial velocity
x[3:] = 0.0


# State covariance
P = np.eye(state_dim) * 1e-3


# Process noise
Q = np.eye(state_dim) * 1e-5


# Measurement noise
R = np.diag([
    1e-8,    # latitude
    1e-8,    # longitude
    0.5      # altitude
])


# Measurement matrix
H = np.zeros((3, 6))

H[0, 0] = 1
H[1, 1] = 1
H[2, 2] = 1


# ============================================================
# 6. STORAGE FOR RESIDUALS
# ============================================================

residuals = []

predicted_measurements = []


# ============================================================
# 7. EKF LOOP
# ============================================================

for k in range(len(data)):

    # --------------------------------------------------------
    # Prediction
    # --------------------------------------------------------

    # State transition matrix
    F = np.array([
        [1, 0, 0, dt,  0,  0],
        [0, 1, 0,  0, dt,  0],
        [0, 0, 1,  0,  0, dt],
        [0, 0, 0,  1,  0,  0],
        [0, 0, 0,  0,  1,  0],
        [0, 0, 0,  0,  0,  1]
    ])

    # Acceleration input
    u = np.array([
        acc_x[k],
        acc_y[k],
        acc_z[k]
    ])

    # Control-input matrix
    B = np.array([
        [0.5 * dt**2, 0, 0],
        [0, 0.5 * dt**2, 0],
        [0, 0, 0.5 * dt**2],
        [dt, 0, 0],
        [0, dt, 0],
        [0, 0, dt]
    ])

    # Predict state
    x_pred = F @ x + B @ u

    # Predict covariance
    P_pred = F @ P @ F.T + Q


    # --------------------------------------------------------
    # Measurement
    # --------------------------------------------------------

    z = np.array([
        lat[k],
        lon[k],
        alt[k]
    ])


    # --------------------------------------------------------
    # Innovation / EKF residual
    # --------------------------------------------------------

    z_pred = H @ x_pred

    innovation = z - z_pred


    # --------------------------------------------------------
    # Innovation covariance
    # --------------------------------------------------------

    S = H @ P_pred @ H.T + R


    # --------------------------------------------------------
    # Kalman gain
    # --------------------------------------------------------

    K = P_pred @ H.T @ np.linalg.inv(S)


    # --------------------------------------------------------
    # EKF update
    # --------------------------------------------------------

    x = x_pred + K @ innovation

    P = (np.eye(state_dim) - K @ H) @ P_pred


    # --------------------------------------------------------
    # Store residuals
    # --------------------------------------------------------

    residuals.append(innovation)

    predicted_measurements.append(z_pred)


# ============================================================
# 8. CONVERT RESIDUALS TO DATAFRAME
# ============================================================

residuals = np.array(residuals)
predicted_measurements = np.array(predicted_measurements)

data["res_latitude"] = residuals[:, 0]
data["res_longitude"] = residuals[:, 1]
data["res_altitude"] = residuals[:, 2]

data["pred_latitude"] = predicted_measurements[:, 0]
data["pred_longitude"] = predicted_measurements[:, 1]
data["pred_altitude"] = predicted_measurements[:, 2]


# ============================================================
# 9. RESIDUAL MAGNITUDE
# ============================================================

data["residual_magnitude"] = np.sqrt(
    data["res_latitude"]**2 +
    data["res_longitude"]**2 +
    data["res_altitude"]**2
)


# ============================================================
# 10. SAVE DATASET
# ============================================================

data.to_csv(output_file, index=False)

print("\nEKF residual generation completed.")

print("Output file:", output_file)

print("\nResidual statistics:")
print(
    data[
        [
            "res_latitude",
            "res_longitude",
            "res_altitude",
            "residual_magnitude"
        ]
    ].describe()
)

print("\nFinal dataset shape:", data.shape)