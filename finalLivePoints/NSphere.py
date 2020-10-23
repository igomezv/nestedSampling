import numpy as np
from scipy.special import loggamma
# Use instead loggamma


class NSphere:
    def __init__(self, ndims, sigma=0.5):
        self.ndims = ndims
        self.sigma = sigma

    def sample_point(self):
        u = np.random.normal(0, 1, self.ndims)
        norm = np.sum(u ** 2) ** 0.5
        r = np.random.random() ** (1.0 / self.ndims)
        x = r * u / norm
        return x

    def sampling(self, npoints):
        return np.array([self.sample_point() for _ in range(npoints)])

    def vol(self, r):
        # r radius
        m = self.ndims / 2 + 1
        return r ** self.ndims * (np.power(np.pi, self.ndims / 2) / gamma(m))

    def logvol(self, r):
        # r radius
        m = self.ndims / 2 + 1
        return self.ndims*np.log(r) + (self.ndims/2)*np.log(np.pi) - loggamma(m)


    def loglike(self, x):
        return -np.sum(x ** 2) / 2 / self.sigma ** 2

    def logl_for_samples(self, array):
        """
        params:
        -------
        array : numpy array with points/samples

        returns:
        loglikes:
        numpy array with the loglikes for each sample/point
        """
        npoints = len(array)
        loglikes = np.zeros((npoints, 1))
        for i, p in enumerate(array):
            loglikes[i] = self.loglike(p)

        return loglikes

    def vol_for_samples(self, array):
        """
        params:
        -------
        array : numpy array with points/samples

        returns:
        loglikes:
        numpy array with the vol for each sample/point
        """
        npoints = len(array)
        volumes = np.zeros((npoints, 1))
        for i, p in enumerate(array):
            # radius r
            r = np.sqrt(np.sum(p ** 2))
            volumes[i] = self.vol(r)

        return volumes
