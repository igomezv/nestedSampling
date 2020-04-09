# Nested sampling based on Skilling (2009)
import numpy as np
import pandas as pd
from time import sleep

# ###################### Here begin the nested sampling


class SkillingNS:
    def __init__(self, logLike, priorTransform, nDims, bounds, nlivepoints=10,
                 accuracy=0.5, **kwargs):
        """
            Parameters
            -----------
            logLike : log-likelihood function
            priorTransform :
                                        priorTransform function that maps(0, 1)
                        -> (bound_inf, bound_sup)
            nDims : int # dimensions
            accuracy : float
            #dim is the dimension or the number of free parameters

            **kwargs:

        """
        self.logLike = logLike
        self.priorTransform = priorTransform
        self.nDims = nDims
        self.bounds = bounds
        self.nlivepoints = nlivepoints
        self.accuracy = accuracy
        # in order to save live and death points
        self.livepoints = []
        self.deathpoints = []

    def sampler(self):
        vectors = []
        # test = np.random.rand(self.nDims,)
        for i in range(self.nDims):
            vectors.append(self.priorTransform(np.random.rand(self.nDims,),
                                               self.bounds))
        # 2) initialise S = 0, X_0 = 1
        # evidence is z
        print("bounds", self.bounds)
        logz = 0
        # initial prior mass is x0 = 1
        xp = 1
        h = 0.0
        # vector of loglikes
        loglikes = []
        for vector in vectors:
            loglikes.append(self.logLike(vector))
        # join loglikes and points
        df = pd.DataFrame()
        df['points'] = vectors
        df['loglikes'] = loglikes

        print("data frame", df.head())
        # j iterations
        j = 200

        for i in range(1, j + 1):
            print("Iteration {}".format(i))
            # sort points by loglikes
            df.sort_values('loglikes', inplace=True)
            df.reset_index(drop=True, inplace=True)
            # sleep(0.5)
            # record the lower loglike,  L_i in the skilling paper
            lowpoint, lowloglike = df.iloc[0]

            # new prior mass X_i in the paper [crude]
            xi = np.exp(-i / self.nlivepoints)
            # xi = -i / self.nlivepoints
            print("X_i : {}".format(xi))
            # wi simple (no trapezoidal)
            wi = xp - xi
            print("w_i : {}".format(wi))
            # z increment
            logz += lowloglike * wi
            print("log(Z) : {}".format(logz))
            # replace lower like point

            # for i, _ in enumerate(self.bounds):
            #     self.bounds[i][0] = self.bounds[i][0] * xi
            #     self.bounds[i][1] = self.bounds[i][1] * xi

            # print("bounds out", self.bounds)

            newpoint, newlike, accepted, rechazed = self.new_point(
                lowpoint, lowloglike, self.bounds)

            print("new point {} \n".format(newpoint))
            df.iloc[0] = newpoint, newlike
            self.print_func(df, logz, xi)
            xp = xi

    def new_point(self, lpoint, llike, bounds):
        """
            This functions recieves the point 
            with the lowest likelihood and generates one with higher 
                    likelihood.

            Parameters:
                ------------

                    lowpoint      :   point with lower likelihood.
                    lowlike       :   lower likelihood.
                    priormass   :   Prior mass
                    priorT      :   Prior transform
        """
        accepted = 0
        rechazed = 0
        ncall = 0
        new_point = lpoint
        new_loglike = llike
        # while (new_loglike < llike or accepted == 0):
        while (ncall < 100 or accepted == 0):
        #print("new point  = llike", new_point)
        	new_point = self.priorTransform(np.random.rand(self.nDims,), bounds)
        	#print("new point 1", new_point)
        	new_loglike = self.logLike(new_point)
        	if new_loglike > llike:
        		accepted += 1
        	else: 
        		rechazed += 1
        	
        	ncall += 1
        return new_point, new_loglike, accepted, rechazed

    def print_func(self, df, logz, xf):
        # print(df['loglikes'].values)
        logz += 1 / self.nDims * xf * np.sum(df['loglikes'].values)

        highestpoint, highestlike = df.iloc[self.nDims - 1]
        # print("Parameter estimation : \n {}  \n ".format(
        #     df['points'].values))
        print("Parameter estimation : {}".format(highestpoint))
        print("Likelihood : \n {} ".format(highestlike))
        print("Bayesian Evidence : {}".format(logz))
        # print("log Z : {}".format(np.log(-z)))
