import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import geomstats.visualization as visualization
from geomstats.geometry.hypersphere import Hypersphere
from geomstats.learning.frechet_mean import FrechetMean
from geomstats.learning.pca import TangentPCA
from geomstats.numerics.optimization import ScipyMinimize

df = pd.read_csv('fc1.csv', sep='\s+', header=None)
eigv = pd.DataFrame.to_numpy(df.iloc[:,:])
l, p = np.shape(eigv)

space = Hypersphere(dim=p-1)
data = eigv
data_aug = data

for i in range(l-1):
    if space.metric.dist(-1 * data[i+1,:], data[i,:]) < space.metric.dist(data[i+1,:], data[i,:]):
        data_aug[i+1,:] = -1 * data[i+1,:]

def proj_to_2sphere(X, tangent_1, tangent_2, base_point): #projecting vectors to a sphere given by a basepoint and two directions
    p1 = base_point
    p2 = space.metric.exp(tangent_1, base_point)
    p3 = space.metric.exp(tangent_2, base_point)

    A = np.hstack((p1.reshape(-1,1),p2.reshape(-1,1),p3.reshape(-1,1)))
    proj = A @ (np.linalg.inv(A.T @ A)) @ (A.T)
    projected_vec = (proj @ (X.T)).T
    row_norm = np.linalg.norm(projected_vec, axis=1)
    D = np.diag(np.reciprocal(row_norm))
    sphere_vec = D @ projected_vec

    return sphere_vec

def loss(X, param):
    intercept, coef1, coef2 = np.split(param, 3)
    intercept = np.reshape(intercept, space.shape)
    coef1 = np.reshape(coef1, space.shape)
    coef2 = np.reshape(coef2, space.shape)

    base_point = space.projection(intercept)
    penalty = np.sum(np.square(base_point - intercept))

    tangent_1 = space.to_tangent(coef1, base_point)
    tangent_2 = space.to_tangent(coef2, base_point)
    distances = space.metric.squared_dist(X, proj_to_2sphere(X, tangent_1, tangent_2, base_point))

    return np.sum(distances) / 2 + penalty

tol = 1e-1
max_iter = 1000
optimizer = ScipyMinimize(
                method="CG",
                options={"disp": False, "maxiter": max_iter},
                tol=tol,
            )

intercept_init, coef1_init, coef2_init = np.random.normal(size=(3,) + space.shape)
intercept_hat = space.projection(intercept_init)
coef1_hat = space.to_tangent(coef1_init, intercept_hat)
coef2_hat = space.to_tangent(coef2_init, intercept_hat)
initial_guess = np.hstack([intercept_hat.flatten(), coef1_hat.flatten(), coef2_hat.flatten()])

objective_with_grad = lambda param: loss(data_aug, param)

result = optimizer.minimize(objective_with_grad, initial_guess)

intercept_fin, coef1_fin, coef2_fin = np.split(result.x, 3)
intercept_fin = np.reshape(intercept_fin, space.shape)
coef1_fin = np.reshape(coef1_fin, space.shape)
coef2_fin = np.reshape(coef2_fin, space.shape)

intercept_ = space.projection(intercept_fin)
coef1_ = space.to_tangent(coef1_fin, intercept_)
coef2_ = space.to_tangent(coef2_fin, intercept_)

sphere_vec = proj_to_2sphere(data_aug, coef1_, coef2_, intercept_)
rss = np.sum(space.metric.squared_dist(data_aug, sphere_vec)) #residual sum of squares
print(rss)

point1 = space.projection(coef1_)
point2 = space.projection(coef2_)
basis = np.vstack((intercept_, point1, point2)).T
Q, R = np.linalg.qr(basis) #QR decomposition, performs Gram--Schmidt

sphere_data = sphere_vec @ Q #dimension reduction, change of basis of the projectred vectors to an orthonormal basis
np.savetxt("fc1_sphere.csv",sphere_data)

sphere = Hypersphere(dim=2)
sphere_mean = FrechetMean(sphere)
sphere_mean.fit(sphere_data)
sphere_mean_estimate = sphere_mean.estimate_ #Frechet mean of points on 2 sphere

sphere_variance = np.sum(sphere.metric.squared_dist(sphere_data, sphere_mean_estimate)) #variance on the 2 sphere
mixed_variance = rss + sphere_variance
fitting_score = 1 - rss / mixed_variance #variance unexplained
print(fitting_score)

fig = plt.figure(figsize=(8,8))
ax = visualization.plot(sphere_data, space='S2', color='black', alpha=0.7) #2 sphere plot
ax.set_box_aspect([1,1,1])
ax.set_title('Leading eigenvectors of input layer')
plt.show()


