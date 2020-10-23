from nestedSampling.NestedSampling import NestedSampling
from nestedSampling.saveDynesty import saveDynestyChain
import numpy as np
import random
from anesthetic import NestedSamples
import matplotlib.pyplot as plt
from scipy.special import gammaln
from scipy.special import gamma


def priorTransform(theta):
    return theta


def logLike(theta):
    return -1/2/0.1**2 * ((theta-0.5)**2).sum()


def volNsphere(r, n):
    m = n / 2 + 1
    return r**n * (np.power(np.pi, n/2) / gamma(m))


ndims = 3

s = NestedSampling(logLike, priorTransform, nlive=100, ndims=ndims, outputname="outputs/gaussian")
s.sampling()
samples = NestedSamples(root='outputs/gaussian')

samples.plot_2d([0, 1])
plt.show()

last_samp = samples.iloc[-100, :ndims]
radius = np.sqrt(((last_samp - 0.5)**2).sum())
# volume = ((last_samp - 0.5)**2).sum() * np.pi # volume is pi r^2
volume = volNsphere(radius, ndims)
logX = np.log(volume)

logXs = samples.logX(1000)

for j, i in enumerate(reversed(range(0, len(logXs), 10))):
    logXs.iloc[i].hist(alpha=0.8)
    last_samp = samples.iloc[i, :ndims]
    # volume = ((last_samp - 0.5)**2).sum() * np.pi
    radius = np.sqrt(((last_samp - 0.5)**2).sum())
    volume = volNsphere(radius, ndims)
    logX = np.log(volume)
    plt.axvline(logX, color='C%i' % j)

plt.axvline(logX, color='k')
plt.show()

# Also should generalise to n-d balls
# https://en.wikipedia.org/wiki/N-sphere#Volume_and_surface_area
