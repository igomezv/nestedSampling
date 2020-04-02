from SkillingNS import SkillingNS
import numpy as np

####################### We need a prior Transform, logLike and Theory##########
def priorTransform(theta, bounds):
    """
    This is a common prior transform (flat priors).
    
    Parameters:
        theta  : is a random vector with de dimensionality of the model.
        bounds : list lower and higher bound.
    """
    return  np.array(theta*(bounds[1]-bounds[0])+bounds[0])

sigma = 0.5 # standard deviation of the noise
LN2PI = np.log(2.*np.pi)
LNSIGMA = np.log(sigma)


def logLike(theta):
    """
    This function is the logLikelihood.
    Parameters:
        theta : is a random vector with lenght as the free parameters of the model.
    """
    m, c = theta # unpack the parameters

    # normalisation
    norm = -0.5*M*LN2PI - M*LNSIGMA

    # chi-squared (data, sigma and x are global variables defined early on in this notebook)
    chisq = np.sum(((data-theory(x, m, c))/sigma)**2)

    #return norm - 0.5*chisq
    return norm -0.5*chisq


def theory(x, m, c):
    """
    A straight line model: y = m*x + c
    
    Parameters:
        
        x (list): a set of abscissa points at which the model is defined
        m (float): the gradient of the line
        c (float): the y-intercept of the line
    """
    
    return m * x + c

#################create some data##############################################
# set the true values of the model parameters for creating the data
m = 3.5 # gradient of the line
c = 1.2 # y-intercept of the line

# set the "predictor variable"/abscissa
M = 100
xmin = 0.
xmax = 10.
stepsize = (xmax-xmin)/M
x = np.arange(xmin, xmax, stepsize)
# create the data - the model plus Gaussian noise


data = theory(x, m, c) + sigma*np.random.randn(M)


bounds = np.array([0, 10])
######################Run Nested Sampling #####################################

sampler = SkillingNS(logLike, priorTransform, 2, bounds, nlivepoints = 10)

sampler.run_sampler()

