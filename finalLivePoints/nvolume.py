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
# if r=1 -> unit_vol = vol_n(r=1)
unit_vol = sphere.vol(1)
# X_0 = 1 -> 1 / unit_vol = X_i / vol_n(ri)
# ---------> X_i = cte vol_n(ri), where cte = 1/unit_vol
cte = 1. / unit_vol
# then X_max = cte * max_vol

print("---- \nMax volume: {} \nvolr1: {} \nratio: {} \n"
      "constant: {} \nX_max: {} \n"
      "loglike: {}\n"
      "radius: {}\n----".format(max_vol, unit_vol, max_vol/unit_vol,
                                cte, cte*max_vol,
                                samples[max_idx, ndims], radius[max_idx]))
