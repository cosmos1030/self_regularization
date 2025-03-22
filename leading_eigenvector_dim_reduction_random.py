import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch
import torch.optim as optim

# Geomstats
import geomstats.visualization as visualization
from geomstats.geometry.hypersphere import Hypersphere
import geomstats.backend as gs

# Geoopt (Riemannian optimization library)
import geoopt
from geoopt import ManifoldParameter
from geoopt.manifolds import Stiefel

torch.manual_seed(42)
np.random.seed(42)

# Argument Parsing
parser = argparse.ArgumentParser(description="Project points onto S2 sphere and visualize")
parser.add_argument('--pk-file', type=str, required=True, help="Path to the .pk file containing data")
parser.add_argument('--output-dir', type=str, default="output", help="Directory to save results")
args = parser.parse_args()

# Create output directory if not exists
os.makedirs(args.output_dir, exist_ok=True)

# Load data from .pk file
import pickle
with open(args.pk_file, "rb") as f:
    data = pickle.load(f)

# Ensure data is a NumPy array
if not isinstance(data, np.ndarray):
    raise ValueError("Loaded data is not a numpy array")

# Shape information
n_points, dim = data.shape
print("Data shape:", data.shape)

# Normalize to project onto S2 sphere
def project_to_s2(points):
    norms = np.linalg.norm(points, axis=1, keepdims=True)
    norms = np.clip(norms, a_min=1e-14, a_max=None)  # Avoid division by zero
    return points / norms

sphere_data = project_to_s2(data)

# Riemannian Optimization (Stiefel Manifold)
manifold = Stiefel()
init_mat = torch.randn(dim, 3)
while torch.linalg.matrix_rank(init_mat) < 3:
    init_mat = torch.randn(dim, 3)
Q, _ = torch.linalg.qr(init_mat)
stiefel_param = ManifoldParameter(Q, manifold=manifold)

optimizer = geoopt.optim.RiemannianSGD([stiefel_param], lr=1e-5, momentum=0.9)
max_iter, tolerance, patience = 100000, 1e-20, 50
prev_loss, count = float('inf'), 0

# Loss function
def loss_fn(X, M):
    base_point, t1, t2 = M[:, 0], M[:, 1], M[:, 2]
    X_proj = (X - torch.sum(X * base_point, dim=1, keepdim=True) * base_point).T
    return 0.5 * torch.sum(torch.norm(X - X_proj.T, dim=1) ** 2)

# Optimization loop
print("Start Riemannian optimization...")
data_tensor = torch.tensor(sphere_data, dtype=torch.float32)
loss_values = []
with tqdm(range(max_iter), desc="Training", unit="step") as pbar:
    for step in pbar:
        optimizer.zero_grad()
        loss_value = loss_fn(data_tensor, stiefel_param)
        loss_value.backward()
        optimizer.step()

        with torch.no_grad():
            Q, _ = torch.linalg.qr(stiefel_param)
            stiefel_param.copy_(Q)
        
        current_loss = loss_value.item()
        loss_values.append(current_loss)
        pbar.set_postfix(loss=f"{current_loss:.6f}")
        
        if abs(prev_loss - current_loss) < tolerance:
            count += 1
            if count >= patience:
                print(f"✅ Converged at step {step} with loss={current_loss:.6f}")
                break
        prev_loss = current_loss

# Final projection
final_M = stiefel_param.detach().clone()
base_point, t1, t2 = final_M[:, 0], final_M[:, 1], final_M[:, 2]
sphere_data = (data_tensor - torch.sum(data_tensor * base_point, dim=1, keepdim=True) * base_point).cpu().numpy()

# Save results
np.savetxt(os.path.join(args.output_dir, "s2_projected_data.csv"), sphere_data)
print("Saved S2 projected data")

# Save loss values
np.savetxt(os.path.join(args.output_dir, "training_loss.csv"), np.array(loss_values))
print("Saved training loss values")

# Visualization
fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(sphere_data[:, 0], sphere_data[:, 1], sphere_data[:, 2], alpha=0.7)
ax.set_xlim([-1, 1])
ax.set_ylim([-1, 1])
ax.set_zlim([-1, 1])
plt.title("Projection onto S2 Sphere")
plt.savefig(os.path.join(args.output_dir, "s2_projection.png"))
plt.show()
print("Visualization saved")

# Plot loss curve
plt.figure()
plt.plot(loss_values, label='Loss')
plt.xlabel("Iteration")
plt.ylabel("Loss")
plt.title("Training Loss Curve")
plt.legend()
plt.savefig(os.path.join(args.output_dir, "training_loss_curve.png"))
plt.show()
print("Loss curve saved")