import numpy as np

class NSphere:
    def __init__(self, ndims):
        self.ndims = ndims

    def sample_point(self):
        u = np.random.normal(0, 1, self.ndims)
        norm = np.sum(u ** 2) ** 0.5
        r = np.random.random() ** (1.0 / self.ndims)
        x = r * u / norm
        return x

    def sampling(self, npoints):
        return np.array([self.samplePoint(self.ndims) for _ in range(npoints)])
