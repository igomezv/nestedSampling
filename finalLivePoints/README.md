Content:

- NSphere : class that allows generate samples within a N-ball, calculates its spherical likelihood and volume.

- PropossalLogLike : class that contains the propossal 
geometric function to fit the samples (final live points)
and a chi-square as likelihood (with and without dims as free
parameter).

- testBayesian : script that implements sampling via dynesty
to the proposal function.

- optimizeFn : script that uses scipy.minimize to find the minimum 
of the chi-square that contains the propossal function.

- testPlotSamplingNSphere : this script uses the NSphere class
to generate points into an unitary N-ball and plot it with 
matplotlib.scatter.