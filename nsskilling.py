####Nested sampling based on Skilling (2009)
import numpy as np
import pandas as pd

####################### We need a prior Transform, logLike and Theory##########################3#

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

#################create data#################
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

####################### Here begin the nested sampling##########################

def replace(lpoint, llike, bounds):
    """
    This functions recieves the point with the lowest likelihood and generates one with higher likelihood.
    
    Parameters:
        lpoint       :   point with lower likelihood.
        llike       :   lower likelihood.
        priormass   :   Prior mass
        priorT      :   Prior transform
    """
    new_point = lpoint
    new_loglike = llike - 0.1

    while (new_loglike < llike):
        #print("finding best likeli")
        #print("new point  = llike", new_point)
        new_point = priorTransform(np.random.rand(dim,), bounds)  
        #print("new point 1", new_point)
        new_loglike = logLike(new_point)       
    print("better point")
    return new_point, new_loglike


    
#N is the number of live points    
N = 10
#dim is the dimension or the number of free parameters
dim = 2

bounds = np.array([0, 10])
vectors = []
test = np.random.rand(dim,)
for i in range(N):
    vectors.append(priorTransform(np.random.rand(dim,), bounds))

#evidence is z
z = 0

#initial prior mass is x0 = 1
xi = 1

#j iterations
j = 100

#vector of loglikes
loglikes = []
for vector in vectors:
    print(type(vector), np.shape(vector), vector)
    loglikes.append(logLike(vector))
#join loglikes and points
df = pd.DataFrame()
df['points'] = vectors
df['loglikes'] = loglikes

for i in range(j):
    print("iteration {}".format(i))
    #sort points by loglikes
    df.sort_values('loglikes', inplace = True)
    df.reset_index(drop=True, inplace=True)
    
    #record the lower loglike,  L_i in the skilling paper
    lowerpoint, lowerlike = df.iloc[0]
    #new prior mass X
    xf = np.exp(-i/N)
    #wi simple
    wi = xi - xf
    z += lowerlike * wi
    #replace lower like point
    print("prior mass {}".format(xf))
    bounds = xf*bounds
    print("bounds remain {}".format(bounds))
    newpoint, newlike = replace(lowerpoint, lowerlike, bounds)
    print("new point {} \n".format(newpoint))
    print("Z : {} ".format(z))
    
    print("logZ : {} ".format(np.log(z)))

    df.iloc[0] = newpoint, newlike


print(df['loglikes'].values)
z += (1/N)*xf*np.sum(df['loglikes'].values)
highestpoint, highestlike = df.iloc[N-1]
print("Parameter estimation : \n {}  \n ".format(df['points'].values))
print("Bayesian Evidence : {}".format(z))
print("log Z : {}".format(np.log(z)))

