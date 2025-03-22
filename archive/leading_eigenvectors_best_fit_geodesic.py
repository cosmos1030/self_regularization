import numpy as np
import jax
import jax.numpy as jnp
from jax.scipy.optimize import minimize
import matplotlib.pyplot as plt
from geomstats.geometry.hypersphere import Hypersphere
from geomstats.learning.frechet_mean import FrechetMean
import pandas as pd
import time

df = pd.read_csv('run5/fc1.csv', sep='\s+', header=None)
eigv = jnp.asarray(df.values, dtype=jnp.float32)
l, p = eigv.shape
space = Hypersphere(dim=p-1)

def jexp(v, p):
    return jnp.cos(jnp.linalg.norm(v)) * p + jnp.sin(jnp.linalg.norm(v)) * v / jnp.linalg.norm(v)

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

def proj_to_geodesic_jax(X, tangent_vec, base):
    factor = jnp.arctan(jnp.divide((jnp.dot(tangent_vec, X.T)), (jnp.dot(base, X.T))) / jnp.linalg.norm(tangent_vec))
    unit_tangent = tangent_vec / jnp.linalg.norm(tangent_vec)
    proj = (jnp.cos(factor).reshape(-1,1) @ base.reshape(1,-1)) + (jnp.sin(factor).reshape(-1,1) @ unit_tangent.reshape(1,-1))
    return proj

def geodesic_mean(gpca_param, base, utangent):
    N = jnp.size(gpca_param)
    t_star = jnp.sum(gpca_param) / N
    k = jnp.arange(N)
    t = t_star + 2 * jnp.pi * k / N
    array = jnp.sum(jnp.square(jnp.arccos(jnp.cos(t).reshape(-1,1) @ jnp.cos(gpca_param).reshape(1,-1) + jnp.sin(t).reshape(-1,1) @ jnp.sin(gpca_param).reshape(1,-1))), axis=1)
    idx = jnp.argmin(array)
    return space.metric.exp(t[idx] * utangent, base)

def loss_jax(X, param):
    intercept, coef = jnp.split(param, 2)
    intercept = jnp.reshape(intercept, (p,))
    coef = jnp.reshape(coef, (p,))
    
    base_point = intercept / jnp.linalg.norm(intercept)
    penalty = jnp.sum(jnp.square(base_point - intercept))
    
    tangent_vec = jto_tangent(coef, base_point)
    projected_vec = proj_to_geodesic_jax(X, tangent_vec, base_point)
    distances = jdist(X, projected_vec) ** 2
    
    return jnp.sum(distances) / 2 + penalty

tol = 1e-5
max_iter = 100
key = jax.random.PRNGKey(42)
intercept_init, coef_init = jax.random.normal(key, shape=(2, p))
intercept_hat = intercept_init / jnp.linalg.norm(intercept_init)
coef_hat = jto_tangent(coef_init, intercept_hat)
initial_guess = jnp.hstack([intercept_hat.flatten(), coef_hat.flatten()])

objective_with_grad = lambda param: loss_jax(eigv, param)
result = minimize(objective_with_grad, initial_guess, method="BFGS", tol=tol)

rss = jnp.sum(space.metric.squared_dist(eigv, proj_to_geodesic_jax(eigv, coef_hat, intercept_hat)))

gpcaprojpts = proj_to_geodesic_jax(eigv, coef_hat, intercept_hat)
gpcalog = space.metric.log(gpcaprojpts, intercept_hat)
parameters_ratio = gpcalog / coef_hat
gpca_param = parameters_ratio[:,0].flatten()

intrinsic_mean = geodesic_mean(gpca_param, intercept_hat, coef_hat)
mixed_variance = jnp.sum(space.metric.squared_dist(eigv, gpcaprojpts)) + jnp.sum(space.metric.squared_dist(gpcaprojpts, intrinsic_mean))
gpcarss = jnp.sum(space.metric.squared_dist(eigv, gpcaprojpts))
gpca_fitting_score = 1 - gpcarss / mixed_variance

print("Residual Sum of Squares:", gpcarss)
print("Fitting Score:", gpca_fitting_score)

xx = jnp.arange(1,101,1)

plt.figure()
plt.plot(xx, gpca_param)
plt.xlabel('Epochs')
plt.ylabel('Geodesic parameter')
plt.title('Leading eigenvectors of input layer')
plt.savefig('fc1_gpcaopt.png')
plt.show()

jnp.savetxt("rss.csv", jnp.array([gpcarss]))
jnp.savetxt("fitting_score.csv", jnp.array([gpca_fitting_score]))
