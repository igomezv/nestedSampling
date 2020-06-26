from nested import nested
import dynesty
import numpy as np
from scipy.special import logsumexp
import seaborn as sb
import matplotlib.pyplot as plt

names = ['r', 'x0']
LaTeXnames = ['r', 'x_0']
bounds = [[0, 5], [0, 6]]

def priorTransform(theta):
    priors = []
    # When theta 0-> append bound[0], if theta 1-> append bound[1]
    for c, bound in enumerate(bounds):
        priors.append(theta[c]*(bound[1]-bound[0])+bound[0])
    return np.array(priors)

# (x-a)**2+(y)**2 - r** = 0 - > y = s
def circle(x, x0, r):
  return r**2 - (x-x0)**2

rr = 2  # gradient of the line
xx0 = 3  # y-intercept of the line

M = 1000
xmin = 0.
xmax = 10.
sigma = 0.05

stepsize = (xmax-xmin)/M
xx = np.arange(xmin, xmax, stepsize)
# create the data - the model plus Gaussian noise
data = circle(xx, xx0, rr) + sigma*np.random.randn(M)

def logLike(vector):
    x0, r = vector  # unpack the parameters

    chisq = np.sum(((data - circle(xx, x0, r)) / sigma) ** 2)
    return -0.5 * chisq


s = nested(logLike, priorTransform, nlive=50, ndims=2, outputname="circle")

result = s.sampling(dlogz=0.01)
