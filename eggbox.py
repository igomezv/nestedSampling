from SkillingNS import SkillingNS
import numpy as np
import matplotlib.pyplot as plt

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

tmax = 5.0 * np.pi
constant = np.log(1.0 / tmax**2)
def eggLoglike(cube):
    t = 2.0 * tmax * cube - tmax
    # a = (2.0 + np.cos(t[0]/2.0)*np.cos(t[1]/2.0))**5.0
    return (2.0 + np.cos(t[0]/2.0)*np.cos(t[1]/2.0))**5.0

def plotterOutput(filename):
    data = np.loadtxt(filename+'.txt')
    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    ax1.set_title(filename)
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    ax1.scatter(data[:, 2], data[:, 3], c='blue')
    plt.savefig(filename)

nDims = 2
# bounds = [[0., 1.], [0., 1.], [-100., 300.]]
bounds = [[0., 1.], [0., 1.]]

sampler = SkillingNS(eggLoglike, priorTransform, nDims, bounds,
                     nlivepoints=100)
sampler.sampler(accuracy=0.01, maxiter=5000, outputname='egg')
plotterOutput('egg')