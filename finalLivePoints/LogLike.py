class LogLike:
    def __init__(self, data):
        self.N, self.ndims = np.shape(data)
        self.data = np.sort(data, axis=0)

    def propossal_fn(self, logLmax, logxMax, i, alpha, d):
        xMax = np.exp(logxMax)
        return logLmax - alpha * np.sign(xMax) * (i * np.abs(xMax) / self.N) ** (2 / d)

    def loglike(self, theta):
        loglmax, logxMax, alpha, d = theta
        sigma = 0.5
        i = np.arange(1, self.N + 1)
        chisq = np.sum(((data -
                         propossalfn(loglmax, logxMax, i, alpha, d)) /
                        sigma) ** 2)

        return 0.5 * chisq

    def loglike_noDim(self, theta, data):
        loglmax, logxMax, alpha = theta
        sigma = 0.5
        #     chisq = np.sum(((likes - propossalfn(points, loglmax, alpha, d))/sigma)**2)
        i = np.arange(1, self.N + 1)
        chisq = np.sum(((data -
                         propossalfn(loglmax, logxMax, i, alpha, self.ndims)) /
                        sigma) ** 2)

        return 0.5 * chisq
