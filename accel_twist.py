import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import linregress
import argparse

# -------------------------------
# 0. Parse command-line argument
# -------------------------------
parser = argparse.ArgumentParser(description="Compute rotation theta from accelerometer data.")
parser.add_argument("-v", "--velocity", type=float, default=20.0,
                    help="Linear velocity in mm/s (default: 20 mm/s)")
args = parser.parse_args()
v = args.velocity
print(f"Using velocity v = {v} mm/s")

# -------------------------------
# 1. Load and preprocess data
# -------------------------------
df = pd.read_csv("accelerometer.csv")
data = df.iloc[50:-50].copy()
time = data.iloc[:, 0].values
distance = (time - time[0]) * v  # zero first point
acc = data.iloc[:, 1:4].values

# -------------------------------
# 2. Compute theta from accelerometer data
# -------------------------------
# Use PCA to find rotation axis
from sklearn.decomposition import PCA
pca = PCA(n_components=3)
pca.fit(acc)
rotation_axis = pca.components_[-1]
rotation_axis /= np.linalg.norm(rotation_axis)

# Define orthogonal vectors for projection
def orthogonal_vectors(v):
    if abs(v[0]) < abs(v[1]):
        temp = np.array([1,0,0])
    else:
        temp = np.array([0,1,0])
    u1 = temp - np.dot(temp, v)*v
    u1 /= np.linalg.norm(u1)
    u2 = np.cross(v, u1)
    u2 /= np.linalg.norm(u2)
    return u1, u2

u1, u2 = orthogonal_vectors(rotation_axis)

# Compute theta
theta = []
for vec in acc:
    x = np.dot(vec, u1)
    y = np.dot(vec, u2)
    theta.append(np.arctan2(y, x))
theta = np.array(theta)
theta -= theta[0]
theta_deg = np.degrees(theta)

# -------------------------------
# 3. Linear regression
# -------------------------------
slope, intercept, r_value, p_value, slope_std_err = linregress(distance, theta_deg)
theta_fit_deg = slope * distance + intercept

print(f"Best-fit line: theta_deg = {slope:.4f} * distance + {intercept:.4f}")
print(f"Gradient (deg/mm): {slope:.4f} ± {slope_std_err:.4f}")

# -------------------------------
# 4. 2D plot
# -------------------------------
plt.figure(figsize=(10,5))
plt.plot(distance, theta_deg, label="Theta (deg)")
plt.plot(distance, theta_fit_deg, 'r--', label="Best-fit line")
plt.xlabel("Distance (mm)")
plt.ylabel("Theta (deg)")
plt.title("Rotation Angle vs Distance with Best-fit Line")
plt.grid(True)
plt.legend()

# Annotate gradient with error in top-right
plt.text(0.95, 0.95, f"Gradient = {slope:.4f} ± {slope_std_err:.4f} deg/mm",
         transform=plt.gca().transAxes, verticalalignment='top',
         horizontalalignment='right',
         bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))

plt.show()
