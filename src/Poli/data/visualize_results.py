import numpy as np
import matplotlib.pyplot as plt
import os


DATA_PATH = "./2048-10f/" 

# Load the files you just saved
real_t0 = np.load("demo_real_t0.npy")
real_t1 = np.load("demo_real_t1.npy")
pred_base = np.load("demo_pred_baseline.npy")
pred_mp = np.load("demo_pred_mp.npy")
pred_grf = np.load("demo_pred_grf.npy")

# Setup 3D Plot
fig = plt.figure(figsize=(18, 6))

# Helper to plot one arm
def plot_arm(ax, points, title, color):
    ax.scatter(points[:, 0], points[:, 1], points[:, 2], c=color, s=5, alpha=0.6)
    ax.set_title(title)
    # Set fixed limits so we can compare movements
    ax.set_xlim([-1, 1]); ax.set_ylim([-0.5, 1.5]); ax.set_zlim([-1, 1])

# 1. Baseline Prediction
ax1 = fig.add_subplot(131, projection='3d')
plot_arm(ax1, real_t1, "Target (Black) vs Baseline (Red)", 'k')
plot_arm(ax1, pred_base, "", 'r')

# 2. MP Prediction
ax2 = fig.add_subplot(132, projection='3d')
plot_arm(ax2, real_t1, "Target (Black) vs MP (Blue)", 'k')
plot_arm(ax2, pred_mp, "", 'b')

# 3. GRF Prediction
ax3 = fig.add_subplot(133, projection='3d')
plot_arm(ax3, real_t1, "Target (Black) vs GRF (Green)", 'k')
plot_arm(ax3, pred_grf, "", 'g')

plt.tight_layout()
plt.show()

# --- Calculate Displacement Error ---
# This checks if models are actually moving points or just staying still
dist_real = np.linalg.norm(real_t1 - real_t0, axis=1).mean()
dist_base = np.linalg.norm(pred_base - real_t0, axis=1).mean()
dist_mp = np.linalg.norm(pred_mp - real_t0, axis=1).mean()
dist_grf = np.linalg.norm(pred_grf - real_t0, axis=1).mean()

print(f"Avg Real Movement amount: {dist_real:.5f}")
print(f"Baseline Predicted Move:  {dist_base:.5f}")
print(f"MP Predicted Move:        {dist_mp:.5f}")
print(f"GRF Predicted Move:       {dist_grf:.5f}")

if __name__ == "__main__":
    os.chdir(DATA_PATH)