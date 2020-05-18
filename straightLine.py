# from SkillingNS import SkillingNS
from SkillingNSv1 import SkillingNS
import numpy as np
import dynesty
# #### We need a prior Transform, logLike and Theory##########

names = ['m', 'c']
LaTeXnames = ['m_1', 'c_1']
bounds = [[0, 10], [-2, 6]]

def priorTransform(theta):
    """
    This is a common prior transform (flat priors).

    Parameters:
        theta  : is a random vector with de dimensionality of the model.
        bounds : list of lists of lower and higher bound for each parameter.
    """
    points = []
    bounds = [[0, 10], [-2, 6]]
    # When theta 0-> append bound[0], if theta 1-> append bound[1]
    for c, bound in enumerate(bounds):
        points.append(theta[c]*(bound[1]-bound[0])+bound[0])

    # At this moment, np.array(priors) has shape (dims,)
    # print("Prior transform : {}".format(np.array(priors)))
    return points

sigma = 0.5

def logLike(theta):
    """
    This function is the logLikelihood.
    Parameters:
        theta : is a random vector with lenght as the free parameters of the model.
    """
    m, c = theta  # unpack the parameters

    chisq = np.sum(((data-theory(x, m, c))/sigma)**2)

    return -0.5*chisq


def theory(x, m, c):
    """
    A straight line model: y = m*x + c

    Parameters:

        x (list): a set of abscissa points at which the model is defined
        m (float): the gradient of the line
        c (float): the y-intercept of the line
    """
    return m*x+c


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


# ##########create some data#######################################
# set the true values of the model parameters for creating the data
# m = 4.2  # gradient of the line
# c = 2.1  # y-intercept of the line
m = 3.5  # gradient of the line
c = 1.2  # y-intercept of the line

# set the "predictor variable"/abscissa
M = 1000
xmin = 0.
xmax = 10.

stepsize = (xmax-xmin)/M

x = np.arange(xmin, xmax, stepsize)
# create the data - the model plus Gaussian noise

data = theory(x, m, c) + sigma*np.random.randn(M)

# bounds = [[bound_inf_m, bound_sup_m], [bound_inf_c, bound_sup_c]]

# ##################Run Nested Sampling #########################
nDims = 2

#
# sampler = SkillingNS(logLike, priorTransform, nDims, bounds=bounds,
#                      nlivepoints=100, names=names)
# #
# sampler.sampler(accuracy=0.01, maxiter=1000, outputname="line")
# # bounds = [[0, 10], [-2, 6]]


#dynesty
dysampler = dynesty.NestedSampler(logLike, priorTransform, nDims,
                                  bound='single', sample='unif', nlive=100)
dysampler.run_nested(dlogz=0.01)
dyresults = dysampler.results
saveDynestyChain(dyresults, "dynestySamplessingle")
