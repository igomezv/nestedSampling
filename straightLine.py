from SkillingNS import SkillingNS
import numpy as np
# from scipy.special import ndtri
# #### We need a prior Transform, logLike and Theory##########

def priorTransform(theta, bounds):
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

# ==========================================================
# def gaussianPriorTransform(theta, bounds):
#     priors = []
#     for c, bound in enumerate(bounds):
#         mu = (bound[1]+ bound[0])/n
#         sigma = (bound[1]-bound[0])/n
#         priors.append(mu+sigma*(ndtri(theta[c])))
#     return np.array(priors)
#
# =========================================================


sigma = 0.5  # standard deviation of the noise
LN2PI = np.log(2.*np.pi)
LNSIGMA = np.log(sigma)


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

    return m * x + c


# ##########create some data#######################################
# set the true values of the model parameters for creating the data
#m = 4.2  # gradient of the line
#c = 2.1  # y-intercept of the line
m = 3.5  # gradient of the line
c = 1.2  # y-intercept of the line

# set the "predictor variable"/abscissa
M = 100
xmin = 0.
xmax = 10.

stepsize = (xmax-xmin)/M

x = np.arange(xmin, xmax, stepsize)
# create the data - the model plus Gaussian noise

data = theory(x, m, c) + sigma*np.random.randn(M)

# bounds = [[bound_inf_m, bound_sup_m], [bound_inf_c, bound_sup_c]]

bounds = [[0, 10], [-2, 6]]
# ##################Run Nested Sampling #########################
nDims = 2

sampler = SkillingNS(logLike, priorTransform, nDims, bounds, nlivepoints=100)

sampler.sampler()

