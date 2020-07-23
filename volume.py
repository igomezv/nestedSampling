from nested import nested
from saveDynesty import saveDynestyChain
import numpy as np
import random
from anesthetic import NestedSamples
import matplotlib.pyplot as plt

def priorTransform(theta):
    return theta

def logLike(theta):
    return -1/2/0.1**2 * ((theta-0.5)**2).sum()

ndims = 2
s = nested(logLike, priorTransform, nlive=200, ndims=ndims, outputname="outputs/gaussian")
s.sampling()
samples = NestedSamples(root='outputs/gaussian')

samples.plot_2d([0, 1])
plt.show()

last_samp = samples.iloc[-200, :ndims]
volume = ((last_samp - 0.5)**2).sum() * np.pi # volume is pi r^2
logX = np.log(volume)
print("logX vol", logX)

logXs = samples.logX(1000)
print("logXs", type(logXs))

for j, i in enumerate(reversed(range(0, len(logXs), 100))):
    logXs.iloc[i].hist(alpha=0.8)
    last_samp = samples.iloc[i, :ndims]
    volume = ((last_samp - 0.5)**2).sum() * np.pi
    logX = np.log(volume)
    plt.axvline(logX, color='C%i' % j)

plt.axvline(logX, color='k')
plt.show()

# Also should generalise to n-d balls
# https://en.wikipedia.org/wiki/N-sphere#Volume_and_surface_area
