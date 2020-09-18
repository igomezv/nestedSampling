from NSphere import NSphere
from PropossalLogLike import PropossalLogLike
import numpy as np
import scipy as sc
from scipy.optimize import minimize

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

# Call likefn
logl = PropossalLogLike(likes, ndims, sigma=sigmaLike)
loglike1 = logl.loglike_noDim

loglike2 = logl.loglike_freeDim

N = len(likes)
tmp_max_logl = np.max(likes)
print("Current max logl of the samples: {}".format(tmp_max_logl))

print("\nMinimize without dim as free parameter:")
op_noDim = minimize(loglike1,  [tmp_max_logl, -0.5, 0.5])
print("Max like: {}, logxMax: {}, "
      "max alpha: {}".format(op_noDim.x[0], op_noDim.x[1],
                             op_noDim.x[2]))

print("\nMinimize with dim as free parameter:")
op_freeD = minimize(loglike2,  [tmp_max_logl, -0.5, 0.5, 3])
print("Max like: {}, logxMax: {}, max alpha: {}, d:"\
      "{}".format(op_freeD.x[0], op_freeD.x[1], op_freeD.x[2], op_freeD.x[3]))