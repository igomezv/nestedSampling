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

# From here begins the volume calculation
print(np.shape(samples[:, :ndims]))
radius = np.sqrt(np.sum(samples[:, :ndims] ** 2, axis=1))
print(np.shape(radius), type(radius))

max_idx = np.argmax(radius)
print(max_idx)


max_vol = sphere.vol(radius[max_idx])

print("Max volume: {}, point: {}, loglike: {}, idx: {}".format(max_vol, samples[max_idx, :ndims],
                                                               samples[max_idx, ndims], max_idx))