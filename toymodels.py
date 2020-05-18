from SkillingNS import SkillingNS
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import random

nDims = 2
bounds = [[-5., 5.], [-5., 5.]]
nlive = 128

def plotterOutput(df, filename='toymodel plot'):
    ax1 = df.plot.scatter(x='x', y='y')
    ax1.axis('equal')
    plt.savefig(filename)

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
sampler = SkillingNS(ringLoglike, priorTransform, nDims, bounds,
                     nlivepoints=nlive, names=names)
result = sampler.sampler(accuracy=0.01, maxiter=5000, outputname='ring')
postsamples = result['samples']
plotterOutput(postsamples, 'ring')

sampler = SkillingNS(gaussLoglike, priorTransform, nDims, bounds,
                     nlivepoints=nlive, names=names)
result = sampler.sampler(accuracy=0.001, maxiter=5000, outputname='gauss')
postsamples = result['samples']
plotterOutput(postsamples, 'gauss')

sampler = SkillingNS(himmelLoglike, priorTransform, nDims, bounds,
                     nlivepoints=nlive, names=names)
result = sampler.sampler(accuracy=0.01, maxiter=5000, outputname='himmel')
postsamples = result['samples']
plotterOutput(postsamples, 'himmel')


