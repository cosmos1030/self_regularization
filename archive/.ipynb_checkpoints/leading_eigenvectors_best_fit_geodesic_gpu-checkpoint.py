import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
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

def proj_to_geodesic(X, tangent_vec, base): #project to geodesic formula from Chakraborty--Seo--Vemuri https://openaccess.thecvf.com/content_cvpr_2016/papers/Chakraborty_An_Efficient_Exact-PGA_CVPR_2016_paper.pdf
    factor = np.arctan(np.divide((tangent_vec @ (X.T)), (base @ (X.T))) / np.linalg.norm(tangent_vec))
    unit_tangent = tangent_vec / np.linalg.norm(tangent_vec)
    proj = ((np.cos(factor).reshape(-1,1)) @ base.reshape(1,-1)) + ((np.sin(factor).reshape(-1,1)) @ unit_tangent.reshape(1,-1))
    projdist1 = space.metric.dist(X, proj)
    negproj = -1 * proj
    projdist2 = space.metric.dist(X, negproj)
    msk = (projdist2 < projdist1)
    msk2 = np.argwhere(msk).reshape(-1)
    proj[msk2,:] = -1 * proj[msk2,:]
    return proj


def function_to_minimize(t, gpca_param):
    G = np.square(np.arccos((np.cos(t).reshape(-1,1) @ np.cos(gpca_param).reshape(1,-1)) + (np.sin(t).reshape(-1,1) @ np.sin(gpca_param).reshape(1,-1))))
    return np.sum(G,axis=1)


def geodesic_mean(gpca_param, base, utangent): #geodesic mean is the Frechet mean on the geodesic of the projected points, formula by Huckemann--Ziezold from https://www.cambridge.org/core/services/aop-cambridge-core/content/view/23C6D6ECE999EF8B2D77E8855EB02CD5/S0001867800000987a.pdf/div-class-title-principal-component-analysis-for-riemannian-manifolds-with-an-application-to-triangular-shape-spaces-div.pdf 
    N = np.size(gpca_param)
    t_star = np.sum(gpca_param) / N
    k = np.arange(N)
    t = t_star + 2 * np.pi * k / N
    array = function_to_minimize(t, gpca_param)
    idx = np.argmin(array)
    geodesic_mean = space.metric.exp(t[idx] * utangent, base)
    return geodesic_mean


def gpcaloss(X, parameters):
    intercept, coef = np.split(parameters, 2)
    intercept = np.reshape(intercept, space.shape)
    coef = np.reshape(coef, space.shape)

    p0 = space.projection(intercept)
    penalty = np.sum(np.square(p0 - intercept))

    tanvec = space.to_tangent(coef, p0)
    dist = space.metric.squared_dist(X, proj_to_geodesic(X, tanvec, p0))

    return np.sum(dist) / 2 + penalty

tol = 1e-5
max_iter = 100
optimizer2 = ScipyMinimize(
                method="CG",
                options={"disp": False, "maxiter": max_iter},
                tol=tol,
            )

intercept_init, coef_init = np.random.normal(size=(2,) + space.shape)
intercept_hat = space.projection(intercept_init)
coef_hat = space.to_tangent(coef_init, intercept_hat)
init_guess = np.hstack([intercept_hat.flatten(), coef_hat.flatten()])

obj_with_grad = lambda parameters: gpcaloss(data_aug, parameters)

result = optimizer2.minimize(obj_with_grad, init_guess)

intercept_0, coef_0 = np.split(result.x, 2)
intercept_0 = np.reshape(intercept_0, space.shape)
coef_0 = np.reshape(coef_0, space.shape)

intercept_fin = space.projection(intercept_0)
coef_fin = space.to_tangent(coef_0, intercept_fin)
unit_coef = space.projection(coef_fin)

gpcaprojpts = proj_to_geodesic(data_aug, unit_coef, intercept_fin) #projected points
gpcalog = space.metric.log(gpcaprojpts, intercept_fin) #taking log
parameters_ratio = gpcalog / unit_coef #dividing by the unit direction vector gives the geodesic parameter
geodesic_param = (parameters_ratio[:,0]).flatten()

intrinsic_mean = geodesic_mean(geodesic_param, intercept_fin, unit_coef)
mixed_var = np.sum(space.metric.squared_dist(data_aug, gpcaprojpts)) + np.sum(space.metric.squared_dist(gpcaprojpts, intrinsic_mean)) #mixed variance is the sum of RSS and the variance along the geodesic, given also by Huckemann--Ziezold
gpcarss = np.sum(space.metric.squared_dist(data_aug, gpcaprojpts))
print(gpcarss)
gpca_fitting_score = 1 - gpcarss / mixed_var #fitting score given by unexplained variances
print(gpca_fitting_score)

xx = np.arange(1,101,1)

gpcafig, gpcaax = plt.subplots()
gpcaax.plot(xx, geodesic_param)
gpcaax.set_xlabel('Epochs')
gpcaax.set_ylabel('Geodesic parameter')
gpcaax.set_title('Leading eigenvectors of input layer')
gpcafig.savefig('fc1_gpcaopt.png')
plt.show()

