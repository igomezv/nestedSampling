from NSphere import NSphere
from PropossalLogLike import PropossalLogLike
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Set npoints and ndims for N sphere
npoints = 100
ndims = 10

# call NSphere class
sphere = NSphere(ndims)

# Generate npoints samples in the N-sphere
points = sphere.sampling(npoints)

# Obtain likes and vols for each sample
likes = sphere.logl_for_samples(points)
vols = sphere.vol_for_samples(points)

# plot
fig = plt.figure()
ax = fig.add_subplot(111)
ax.scatter(vols, likes, c='g')
ax.set_aspect('equal', 'box')

plt.xlabel("volume")
plt.ylabel("loglike")
plt.title("$log L vs volume for {}-ball$".format(ndims))
plt.show()