from NSphere import NSphere
from PropossalLogLike import PropossalLogLike
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from dynesty import plotting as dyplot
import dynesty
import numpy as np
import multiprocessing as mp

#

# Set npoints and ndims for N sphere
npoints = 500
ndims = 3

# call NSphere class
sphere = NSphere(ndims)

# Generate npoints samples in the N-sphere
points = sphere.sampling(npoints)

# Obtain likes and vols for each sample
likes = sphere.logl_for_samples(points)

samples = np.concatenate([points, likes], axis=1)

# Define bounds for free parameters
sigmaLike = 0.3
logL_bounds = (-4*sigmaLike*ndims, 0)
alpha_bounds = (0, 1)
d_bounds = (1, 10)
logxMax_bounds = (-1e4, 0)
bounds_freeDim = [logL_bounds, logxMax_bounds, alpha_bounds, d_bounds]
bounds_noDim = [logL_bounds, logxMax_bounds, alpha_bounds]

# Define a priorTransform
bounds = bounds_noDim

def priorTransform(theta):
    points = []
    for c, bound in enumerate(bounds):
        points.append(theta[c]*(bound[1]-bound[0])+bound[0])
    return points

logl = PropossalLogLike(likes, ndims, sigma=sigmaLike)

loglike = logl.loglike_noDim
# loglike = logl.loglike_freeDim

sampler = dynesty.NestedSampler(loglike, priorTransform, ndims, bound='multi',
                                sample='unif', nlive=100)
sampler.run_nested(dlogz=0.01)
sampler.results.summary()

# the order is logL, logxMax, alpha, dim

fig, ax = dyplot.cornerplot(sampler.results, color='blue', show_titles=True,
                           quantiles=None)
plt.show()