from nestedSampling.NestedSampling import NestedSampling
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import random

nDims = 2
bounds = [[-5., 5.], [-5., 5.]]
nlive = 128

def plotterOutput(data, filename='toymodel plot'):
    data = np.array(data)
    x = data[:, 0]
    y = data[:, 1]
    plt.scatter(x, y, c='r')
    plt.xlabel("x")
    plt.ylabel("y")
    plt.axis('equal')
    plt.savefig(filename)
    plt.show()


def priorTransform(theta):
    """
    This is a common prior transform (flat priors).

    Parameters:
        theta  : is a random vector with de dimensionality of the model.
        bounds : list of lists of lower and higher bound for each parameter.
    """
    priors = []
    # When theta 0-> append bound[0], if theta 1-> append bound[1]
    for c, bound in enumerate(bounds):
        priors.append(theta[c]*(bound[1]-bound[0])+bound[0])

    # At this moment, np.array(priors) has shape (dims,)
    # print("Prior transform : {}".format(np.array(priors)))
    return np.array(priors)

def himmelLoglike(cube):
    return -(cube[0]**2+cube[1]-11)**2.0-(cube[0]+cube[1]**2-7)**2

def gaussLoglike(x):
    return -((x[0]) ** 2 + (x[1]) ** 2 / 2.0 - 1.0 * x[0] * x[1]) / 2.0

def ringLoglike(x):
    r2 = x[0] ** 2 + x[1] ** 2
    return -(r2 - 4.0) ** 2 / (2 * 0.5 ** 2)

names = ['x', 'y']

# logLike, priorTransform, nDims, bounds,  nlivepoints=50, names=None, LatexNames=None
sampler = NestedSampling(ringLoglike, priorTransform, nlive, nDims,
                         outputname='outputs/ring', maxiter=5000)
result = sampler.sampling(dlogz=0.1)
postsamples = result['samples']
plotterOutput(postsamples, 'outputs/ring')

sampler = NestedSampling(gaussLoglike, priorTransform, nlive, nDims,
                         outputname='outputs/gauss', maxiter=5000)

result = sampler.sampling(dlogz=0.01)
postsamples = result['samples']
plotterOutput(postsamples, 'outputs/gauss')

sampler = NestedSampling(himmelLoglike, priorTransform, nlive, nDims,
                         outputname='outputs/himmel', maxiter=5000)

result = sampler.sampling(dlogz=0.01)
postsamples = result['samples']
plotterOutput(postsamples, 'outputs/himmel')


