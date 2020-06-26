from nested import nested
from saveDynesty import saveDynestyChain
import numpy as np
import random
from anesthetic import NestedSamples

def priorTransform(theta):
    return theta

def logLike(theta):
    return -1/2/0.1**2 * ((theta-0.5)**2).sum()

ndims = 2
s = nested(logLike, priorTransform, nlive=200, ndims=ndims, outputname="outputs/gaussian")
s.sampling()
samples = NestedSamples(root='outputs/gaussian')

# Problem: 
# - These do not look gaussian
# - increasing the number of steps makes them look worse
samples.plot_2d([0,1])


last_samp = samples.iloc[-1,:ndims]
volume = ((last_samp - 0.5)**2).sum() * np.pi # volume is pi r^2
logX = np.log(volume)

samples.logX() # Need to finish anesthetic PR https://github.com/williamjameshandley/anesthetic/pull/81
# Want to check these look the same

# Also should generalise to n-d balls
# https://en.wikipedia.org/wiki/N-sphere#Volume_and_surface_area

