from nestedSampling import nestedSampling
import numpy as np
import dynesty
from math import *
import random
# #### We need a prior Transform, logLike and Theory##########

def theory(x, m, c):
    """
    A straight line model: y = m*x + c

    Parameters:

        x (list): a set of abscissa points at which the model is defined
        m (float): the gradient of the line
        c (float): the y-intercept of the line
    """
    return m*x+c

names = ['m', 'c']
LaTeXnames = ['m_1', 'c_1']
bounds = [[0, 5], [-2, 6]]

m = 3.5  # gradient of the line
c = 1.2  # y-intercept of the line

M = 1000
xmin = 0.
xmax = 10.
sigma = 0.5

stepsize = (xmax-xmin)/M
np.random.seed(0)
x = np.arange(xmin, xmax, stepsize)

# create the data - the model plus Gaussian noise
data = theory(x, m, c) + sigma*np.random.randn(M)

def priorTransform(theta):
    """
    This is a common prior transform (flat priors).

    Parameters:
        theta  : is a random vector with de dimensionality of the model.
        bounds : list of lists of lower and higher bound for each parameter.
    """
    points = []
    # bounds = [[0, 10], [-2, 6]]
    # When theta 0-> append bound[0], if theta 1-> append bound[1]
    for c, bound in enumerate(bounds):
        points.append(theta[c]*(bound[1]-bound[0])+bound[0])

    # At this moment, np.array(priors) has shape (dims,)
    # print("Prior transform : {}".format(np.array(priors)))
    return points


def logLike(theta):
    """
    This function is the logLikelihood.
    Parameters:
        theta : is a random vector with lenght as the free parameters of the model.
    """
    m, c = theta  # unpack the parameters

    chisq = np.sum(((data-theory(x, m, c))/sigma)**2)

    return -0.5*chisq


def saveDynestyChain(result, outputname):
    f = open(outputname + '.txt', 'w+')

    weights = np.exp(result['logwt'] - result['logz'][-1])

    postsamples = result.samples

    print('\n Number of posterior samples is {}'.format(postsamples.shape[0]))

    for i, sample in enumerate(postsamples):
        strweights = str(weights[i])
        strlogl = str(result['logl'][i])
        strsamples = str(sample).lstrip('[').rstrip(']')
        row = strweights + ' ' + strlogl + ' ' + strsamples  # + strOLambda
        nrow = " ".join(row.split())
        f.write(nrow + '\n')

nlivepoints = 100

s = nestedSampling(logLike, priorTransform, nlive=200, ndims=2, maxiter=30000)
s.sampling()
# results = nested_sampling(n, max_iter, sample_from_prior, explore, priorTransform)
# process_results(results)

dysampler = dynesty.NestedSampler(logLike, priorTransform, 2,
                                  bound='single', sample='unif', nlive=100)
dysampler.run_nested(dlogz=0.01)
dyresults = dysampler.results
saveDynestyChain(dyresults, "dynestySamples")

dysampler = dynesty.NestedSampler(logLike, priorTransform, 2,
                                  bound='multi', sample='unif', nlive=100)
dysampler.run_nested(dlogz=0.01)
dyresults = dysampler.results
saveDynestyChain(dyresults, "dynestySamples2")


