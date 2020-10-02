from NSphere import NSphere
from PropossalLogLike import PropossalLogLike
import numpy as np
import scipy as sc
from scipy.optimize import minimize

# Set npoints and ndims for N sphere
npoints = 1000
ndims = 6
print("{} dimensions".format(ndims))

# call NSphere class
sphere = NSphere(ndims)

# Generate npoints samples in the N-sphere
points = sphere.sampling(npoints)

# Obtain likes and vols for each sample
likes = sphere.logl_for_samples(points)

samples = np.concatenate([points, likes], axis=1)

# From here begins the volume calculation
radius = np.sqrt(np.sum(samples[:, :ndims] ** 2, axis=1))

max_idx = np.argmax(radius)
max_vol = sphere.vol(radius[max_idx])

print("Max volume: {}, volr1: {}, ratio: {},"
      "loglike: {},"
      "radius: {}, ".format(max_vol, sphere.vol(1), max_vol/sphere.vol(1),
                            samples[max_idx, ndims], radius[max_idx]))