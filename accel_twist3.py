import numpy as np
import glob
import re
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# --- Step 1: Load CSVs and sort by position ---
file_pattern = "lis2dw-*.csv"
files = glob.glob(file_pattern)

def extract_position(filename):
    match = re.search(r"lis2dw-(\d+(\.\d+)?)\.csv", filename)
    return float(match.group(1)) if match else np.inf

files.sort(key=extract_position)
positions = [extract_position(f) for f in files]

# --- Step 2: Stack mean accelerometer vectors to define motion axis ---
mean_vectors = []
raw_data = []
for f in files:
    data = np.loadtxt(f, delimiter=',', skiprows=1)
    raw_data.append(data[:, 1:4])
    mean_vectors.append(np.mean(data[:, 1:4], axis=0))
mean_vectors = np.array(mean_vectors)

# --- Step 3: PCA to determine motion axis ---
pca = PCA(n_components=3)
pca.fit(mean_vectors)
motion_axis = pca.components_[0]  # principal motion axis
motion_axis /= np.linalg.norm(motion_axis)

# --- Step 4: Compute cumulative twist along path with per-reading deviations ---
cumulative_twist = 0.0
twist_distributions = []
cumulative_values = []

prev_mean_proj = None
mid_idx = len(files) // 2  # for color coding

for i, acc in enumerate(raw_data):
    # Project readings onto plane perpendicular to motion axis
    acc_proj = acc - np.outer(np.dot(acc, motion_axis), motion_axis)
    curr_mean_proj = np.mean(acc_proj, axis=0)

    if prev_mean_proj is None:
        # First file = zero cumulative twist
        angles_deg = np.zeros(len(acc))
        twist_distributions.append(angles_deg)
        cumulative_values.append(0.0)
        prev_mean_proj = curr_mean_proj
        continue

    # Incremental twist between consecutive means
    cross = np.cross(prev_mean_proj, curr_mean_proj)
    dot = np.dot(prev_mean_proj, curr_mean_proj)
    angle_rad = np.arctan2(np.dot(cross, motion_axis), dot)
    cumulative_twist += np.degrees(angle_rad)

    # --- Stable per-reading deviations relative to current mean ---
    # Normalize vectors
    acc_norm = acc_proj / np.linalg.norm(acc_proj, axis=1)[:, None]
    mean_norm = curr_mean_proj / np.linalg.norm(curr_mean_proj)

    # Signed angle of each reading relative to mean
    cross = np.cross(mean_norm, acc_norm)  # shape (n,3)
    dot = np.dot(acc_norm, mean_norm)      # shape (n,)
    dev_angles = np.degrees(np.arctan2(np.dot(cross, motion_axis), dot))

    # Total twist = cumulative + per-reading deviation
    angles_deg = cumulative_twist + dev_angles
    twist_distributions.append(angles_deg)
    cumulative_values.append(cumulative_twist)

    prev_mean_proj = curr_mean_proj

# --- Step 5: Plot boxplots + jittered points ---
plt.figure(figsize=(12, 6))
plt.boxplot(twist_distributions, positions=positions, widths=5, patch_artist=True)

# Add jittered points with color coding
for idx, (pos, angles) in enumerate(zip(positions, twist_distributions)):
    x_jitter = np.random.uniform(-2, 2, size=len(angles))
    color = 'blue' if idx <= mid_idx else 'red'
    plt.scatter(np.full(len(angles), pos) + x_jitter, angles, color=color, alpha=0.5, s=10)
    # Annotate cumulative twist value above boxplot
    plt.text(pos, np.max(angles)+0.5, f"{cumulative_values[idx]:.2f}°",
             ha='center', va='bottom', fontsize=8, color='black', rotation=0)

plt.xlabel('Position along motion axis')
plt.ylabel('Cumulative twist (degrees)')
plt.title('Total twist along axis of motion (PCA-based)')
plt.grid(True)
plt.show()

print(f"Total cumulative twist along the path: {cumulative_twist:.2f}°")
