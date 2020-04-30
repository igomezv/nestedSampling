from SkillingNS import SkillingNS
import numpy as np
import matplotlib.pyplot as plt

def plotterOutput(filename):
    data = np.loadtxt(filename+'.txt')
    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    ax1.set_title(filename)
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    ax1.scatter(data[:, 2], data[:, 3], c=np.random.rand(3))
    plt.savefig(filename)

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

def himmelLoglike(cube):
    return -(cube[0]**2+cube[1]-11)**2.0-(cube[0]+cube[1]**2-7)**2

def gaussLoglike(x):
    return -((x[0]) ** 2 + (x[1]) ** 2 / 2.0 - 1.0 * x[0] * x[1]) / 2.0

def ringLoglike(x):
    r2 = x[0] ** 2 + x[1] ** 2
    return -(r2 - 4.0) ** 2 / (2 * 0.5 ** 2)


nDims = 2
bounds = [[-5., 5.], [-5., 5.]]
nlive = 128

sampler = SkillingNS(ringLoglike, priorTransform, nDims, bounds,
                     nlivepoints=nlive)
sampler.sampler(accuracy=0.01, maxiter=5000, outputname='ring')
plotterOutput('ring')

sampler = SkillingNS(gaussLoglike, priorTransform, nDims, bounds,
                     nlivepoints=nlive)
sampler.sampler(accuracy=0.001, maxiter=5000, outputname='gauss')
plotterOutput('gauss')


sampler = SkillingNS(himmelLoglike, priorTransform, nDims, bounds,
                     nlivepoints=nlive)
sampler.sampler(accuracy=0.01, maxiter=5000, outputname='himmel')
plotterOutput('himmel')

