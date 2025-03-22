import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import geomstats.visualization as visualization
from geomstats.geometry.hypersphere import Hypersphere
from geomstats.learning.frechet_mean import FrechetMean
import jax.numpy as jnp
import jax
import optax
import geomstats.backend as gs
import os
from tqdm import tqdm

# Set up environment variables to prevent memory preallocation issues
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"

base_dir = 'run4'
print(jax.devices())

# Load eigenvectors from CSV file
df = pd.read_csv(os.path.join(base_dir, 'eigenvectors/fc1.csv'), sep='\s+', header=None)
print(df.shape)
eigv = df.to_numpy()
l, p = eigv.shape

# Define Hypersphere space
space = Hypersphere(dim=p - 1)
data_aug = eigv.copy()

# Ensure eigenvectors have consistent orientations
for _ in range(3):  # Iterate multiple times for correction
    for i in range(l - 1):
        if space.metric.dist(-1 * data_aug[i + 1, :], data_aug[i, :]) < space.metric.dist(data_aug[i + 1, :], data_aug[i, :]):
            data_aug[i + 1, :] = -1 * data_aug[i + 1, :]

# JAX-based geometric functions
def jexp(v, p):
    norm_v = jnp.linalg.norm(v)
    return jnp.cos(norm_v) * p + jnp.sin(norm_v) * v / norm_v

def jdist(A, B):
    norm_a = jnp.linalg.norm(A, axis=1)
    norm_b = jnp.linalg.norm(B, axis=1)
    inner_prod = jnp.sum(A * B, axis=1)
    cos_angle = inner_prod / (norm_a * norm_b)
    cos_angle = jnp.clip(cos_angle, -1, 1)
    return jnp.arccos(cos_angle)

def jto_tangent(v, p):
    coef = jnp.sum(v * p) / jnp.sum(p * p)
    return v - coef * p

def proj_to_2sphere(X, tangent_1, tangent_2, base_point):
    p1 = base_point
    p2 = jexp(tangent_1, base_point)
    p3 = jexp(tangent_2, base_point)
    
    A = jnp.hstack((p1.reshape(-1, 1), p2.reshape(-1, 1), p3.reshape(-1, 1)))
    proj = A @ jnp.linalg.pinv(A.T @ A) @ A.T  # Using pseudo-inverse
    projected_vec = (proj @ X.T).T
    row_norm = jnp.linalg.norm(projected_vec, axis=1)
    sphere_vec = (projected_vec.T / row_norm).T
    return sphere_vec

# Loss function definition
def loss(X, param):
    intercept, coef1, coef2 = jnp.split(param, 3)
    base_point = intercept / jnp.linalg.norm(intercept)
    penalty = jnp.sum(jnp.square(base_point - intercept))
    
    tangent_1 = jto_tangent(coef1, base_point)
    tangent_2 = jto_tangent(coef2, base_point)
    distances = jdist(X, proj_to_2sphere(X, tangent_1, tangent_2, base_point)) ** 2
    return jnp.sum(distances) / 2 + penalty

# Initialize parameters
intercept_init, coef1_init, coef2_init = np.random.normal(size=(3,) + space.shape)
params = jnp.hstack([
    intercept_init.flatten(),
    coef1_init.flatten(),
    coef2_init.flatten()
])

# Adam optimizer setup
learning_rate = 0.01
optimizer = optax.adam(learning_rate)
opt_state = optimizer.init(params)

# Gradient computation
loss_fn = lambda param: loss(data_aug, param)
grad_fn = jax.grad(loss_fn)

# Batch size setting
batch_size = 32
batched_loss_fn = jax.vmap(loss_fn, in_axes=(0, None))
batched_grad_fn = jax.vmap(jax.grad(loss_fn), in_axes=(0, None))

# Optimize with Adam
num_epochs = 1000
for step in tqdm(range(num_epochs), desc="Optimizing with Adam", unit="step"):
    grads = grad_fn(params)
    updates, opt_state = optimizer.update(grads, opt_state)
    params = optax.apply_updates(params, updates)

# Extract final optimized parameters
intercept_fin, coef1_fin, coef2_fin = np.split(params, 3)

# Convert parameters back to hypersphere space
intercept_ = space.projection(intercept_fin)
coef1_ = space.to_tangent(coef1_fin, intercept_)
coef2_ = space.to_tangent(coef2_fin, intercept_)

# 2-Sphere projection
sphere_vec = proj_to_2sphere(data_aug, coef1_, coef2_, intercept_)
rss = np.sum(space.metric.squared_dist(data_aug, sphere_vec))  # Residual Sum of Squares
print("RSS:", rss)

# Dimension reduction using QR decomposition
point1 = space.projection(coef1_)
point2 = space.projection(coef2_)
basis = np.vstack((intercept_, point1, point2)).T
Q, _ = np.linalg.qr(basis)  # QR Decomposition
sphere_data = sphere_vec @ Q  # Basis transformation
sphere_data = np.array(sphere_data, dtype=np.float64)
norms = gs.linalg.norm(sphere_data, axis=1, keepdims=True)
sphere_data /= norms

# Save processed data
np.savetxt(os.path.join(base_dir, "fc1_sphere.csv"), sphere_data)

# Compute Frechet mean
sphere = Hypersphere(dim=2)
sphere_mean = FrechetMean(sphere)
sphere_mean.fit(sphere_data)
sphere_mean_estimate = sphere_mean.estimate_

# Variance computation
sphere_variance = np.sum(sphere.metric.squared_dist(sphere_data, sphere_mean_estimate))
mixed_variance = rss + sphere_variance
fitting_score = 1 - rss / mixed_variance  # Variance unexplained
print("Fitting Score:", fitting_score)

# Visualization
fig = plt.figure(figsize=(8, 8))
ax = visualization.plot(sphere_data, space='S2', color='black', alpha=0.7)
ax.set_box_aspect([1, 1, 1])
ax.set_title('Leading eigenvectors of input layer')
fig.savefig(os.path.join(base_dir, "fc1_sphere.png"), dpi=300)
plt.show()
