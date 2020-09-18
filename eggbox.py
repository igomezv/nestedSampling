from nestedSampling.NestedSampling import NestedSampling
import numpy as np
import matplotlib.pyplot as plt


bounds = [[0., 1.], [0., 1.]]


def priorTransform(theta):
    priors = []
    # When theta 0-> append bound[0], if theta 1-> append bound[1]
    for c, bound in enumerate(bounds):
        priors.append(theta[c]*(bound[1]-bound[0])+bound[0])
    return np.array(priors)


tmax = 5.0 * np.pi
constant = np.log(1.0 / tmax**2)


def eggLoglike(cube):
    t = 2.0 * tmax * cube - tmax
    return (2.0 + np.cos(t[0]/2.0)*np.cos(t[1]/2.0))**5.0


def plotterOutput(data, filename='toymodel plot'):
    data = np.array(data)
    x = data[:, 0]
    y = data[:, 1]
    plt.scatter(x, y, c='g')
    plt.xlabel("x")
    plt.ylabel("y")
    plt.axis('equal')
    plt.savefig(filename)
    plt.show()


nDims = 2
sampler = NestedSampling(eggLoglike, priorTransform, 100, nDims,
                         outputname='outputs/egg', maxiter=5000)

sampler.sampling(dlogz=0.5)
plotterOutput('outputs/egg')
