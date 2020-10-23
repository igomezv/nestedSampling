from NSphere import NSphere
from PropossalLogLikeDet import PropossalLogLikeDet
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from dynesty import plotting as dyplot
import dynesty
import numpy as np
import multiprocessing as mp
from saveDynesty import saveDynestyChain
from getdist import plots, MCSamples, chains
from getdist import *

# Set npoints and ndims for N sphere
npoints = 10
ndims = 3
# if dim == True -> dim is free paramaterer
dim = False

# call NSphere class
sphere = NSphere(ndims)

# Generate npoints samples in the N-sphere
points = sphere.sampling(npoints)

# Obtain likes and vols for each sample
likes = sphere.logl_for_samples(points)
samples = np.concatenate([points, likes], axis=1)
np.savetxt('samples.txt', samples)

# Define bounds for free parameters
sigmaLike = 0.5
logL_bounds = [-ndims/2*(sigmaLike**2), 0]
# alpha_bounds = [0, 1]
d_bounds = [1, 10]
logxMax_bounds = [0, -2]
# bounds_freeDim = [logL_bounds, logxMax_bounds, alpha_bounds, d_bounds]
# bounds_noDim = [logL_bounds, logxMax_bounds, alpha_bounds]
bounds_freeDim = [logL_bounds, logxMax_bounds, d_bounds]
bounds_noDim = [logL_bounds, logxMax_bounds]
outputname = 'test_sampling_{}D_sphere_DIM_{}_sigma{}'.format(ndims, dim, sigmaLike)

logl = PropossalLogLikeDet(samples, ndims, sigma=sigmaLike)
# Define a priorTransform
if dim:
    bounds = bounds_freeDim
    loglike = logl.loglike_freeDim
else:
    bounds = bounds_noDim
    loglike = logl.loglike_noDim

nfreepars = len(bounds)


def priorTransform(theta):
    points = []
    for c, bound in enumerate(bounds):
        points.append(theta[c]*(bound[1]-bound[0])+bound[0])
    return points


sampler = dynesty.NestedSampler(loglike, priorTransform, nfreepars, bound='multi',
                                sample='unif', nlive=100)
sampler.run_nested(dlogz=0.01)
sampler.results.summary()

results = sampler.results

# pars = [['logL', 'logL_{max}'], ['logX', 'logX_{max}'], ['alpha', '\\alpha']]
pars = [['logL', 'logL_{max}'], ['logX', 'logX_{max}']]

if dim:
    pars.append(['dim', 'dim'])

saveDynestyChain(results, "../outputs/{}".format(outputname), pars)
filename = '../outputs/{}'.format(outputname)
mcsamplefile = mcsamples.loadMCSamples(filename, ini=None, jobItem=None, no_cache=False)
g = plots.getSubplotPlotter(width_inch=10,
                            analysis_settings={'smooth_scale_2D': 0.8,
                                              'smooth_scale_1D': 0.8})
g.settings.lab_fontsize = 10
g.settings.legend_fontsize = 9
g.settings.axes_fontsize = 9
g.triangle_plot(mcsamplefile, [par[0] for par in pars],
                filled=False,
                shaded=True)

plt.savefig(filename+"_getdist.png")
# the order is logL, logxMax, alpha, dim
fig, ax = dyplot.cornerplot(results, color='blue', show_titles=True,
                            quantiles=None)

plt.savefig(filename+"_dyplot.png")