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

def gamma_fn(m): # 3
    if m == 0 or m == 1:
        return 1
    elif m == 0.5:
        return 0.5*np.sqrt(np.pi)
    else:
        return (m-1)*gamma_fn(m-2)

def volNsphere(r, n):
    m = n / 2 + 1  # m = 1.5 +1 = 2.5
    return (r**n * np.power(np.pi,n/2))/gamma_fn(m)


ndims = 3


s = nested(logLike, priorTransform, nlive=200, ndims=ndims, outputname="outputs/gaussian")
s.sampling()
samples = NestedSamples(root='outputs/gaussian')

samples.plot_2d([0, 1])
plt.show()

last_samp = samples.iloc[-200, :ndims]
radius = np.sqrt(((last_samp - 0.5)**2).sum())
print("radius", radius)
# volume = ((last_samp - 0.5)**2).sum() * np.pi # volume is pi r^2
volume = volNsphere(radius, ndims)
print("1 volume", volume)
logX = np.log(volume)
print("1 logX vol", logX)

logXs = samples.logX(1000)
print("logXs", type(logXs))

for j, i in enumerate(reversed(range(0, len(logXs), 100))):
    logXs.iloc[i].hist(alpha=0.8)
    last_samp = samples.iloc[i, :ndims]
    print("last_samp", last_samp)
    # volume = ((last_samp - 0.5)**2).sum() * np.pi
    radius = np.sqrt(((last_samp - 0.5)**2).sum())
    print("radius", radius)
    volume = volNsphere(radius, ndims)
    print("volume", volume)
    logX = np.log(volume)
    plt.axvline(logX, color='C%i' % j)

plt.axvline(logX, color='k')
plt.show()

# Also should generalise to n-d balls
# https://en.wikipedia.org/wiki/N-sphere#Volume_and_surface_area
