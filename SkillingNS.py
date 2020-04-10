# Nested sampling based on Skilling (2009)
import numpy as np
import pandas as pd

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
        # in order to save live and dead points
        # self.livepoints = []
        # self.deadpoints = []

    def sampler(self):
        vectors = []
        # vector of loglikes
        loglikes = []
        # list for dead values
        deadpoints = []
        deadloglikes = []

        for i in range(self.nlivepoints):
            vectors.append(self.priorTransform(np.random.rand(self.nDims,),
                                               self.bounds))
            loglikes.append(self.logLike(vectors[i]))

        # join loglikes and points
        df_live = pd.DataFrame()
        df_live['points'] = vectors
        df_live['loglikes'] = loglikes

        print("data frame", df_live.head())
        # 2) initialise S = 0, X_0 = 1
        # evidence is z
        # print("bounds", self.bounds)
        logz = 0

        # initial prior mass is x0 = 1
        logx_prev = 1
        h = 0
        # j iterations
        j = 10000

        for i in range(j):
            print("Iteration {}".format(i + 1))
            # sort points by loglikes
            df_live.sort_values('loglikes', inplace=True)
            df_live.reset_index(drop=True, inplace=True)
            # sleep(0.5)
            # record the lower loglike,  L_i in the skilling paper
            lowpoint, lowloglike = df_live.iloc[0]
            deadloglikes.append(lowloglike)
            deadpoints.append(lowpoint)
            # new prior mass X_i in the paper [crude]
            logx_current = np.exp(-(i + 1) / self.nlivepoints)
            # xi = -i / self.nlivepoints
            print("X_i : {}".format(logx_current))
            # wi simple (no trapezoidal)
            wi = logx_prev - logx_current
            print("w_i : {}".format(wi))
            # z increment
            logz += lowloglike * wi

            # THIS IS BAD, RIGHT?
            # for i, _ in enumerate(self.bounds):
            #     self.bounds[i][0] = self.bounds[i][0] * xi
            #     self.bounds[i][1] = self.bounds[i][1] * xi

            # print("bounds out", self.bounds)

            newpoint, newlike = self.generate_point(lowpoint, lowloglike)

            print("new point {} \n".format(newpoint))
            df_live.iloc[0] = newpoint, newlike
            self.print_func(df_live, logz)
            logx_prev = logx_current

        logz += (1 / self.nlivepoints) * \
            np.sum(df_live['loglikes'].values) * logx_current

            


        df_dead = pd.DataFrame()
        df_dead['points'] = deadpoints
        df_dead['loglikes'] = deadloglikes
        samples = pd.concat([df_dead, df_live], ignore_index=True)
        # Only in order to visualize the total samples:
        for row in samples.values:
        	print(row)

    def generate_point(self, lpoint, llike):
        """
            This functions recieves the point 
            with the lowest likelihood and generates one with higher 
            Likelihood.

            Parameters:
		    ------------

		    lowpoint      :   point with lower likelihood.
		    lowlike       :   lower likelihood.
		    riormass   :   Prior mass
		    priorT      :   Prior transform
	    """

        self.accepted = 0
        self.rejected = 0
        self.ncall = 0
        new_point = lpoint
        new_loglike = llike
        # while (ncalls < 1000 or accepted == 0):
        while (new_loglike < llike or self.accepted == 0):
            # print("new point  = llike", new_point)
            proposal_point = np.random.rand(self.nDims,)
            # new_point = self.priorTransform(proposal_point, bounds)
            new_point = self.priorTransform(proposal_point, self.bounds)
            new_loglike = self.logLike(new_point)
            if new_loglike > llike:
                self.accepted += 1
            else:
                self.rejected += 1

            self.ncall += 1
        return new_point, new_loglike

    def print_func(self, df, logz):
        print("Accepted: {} || Rejected: {} || ncalls: {}".format(
            					self.accepted, self.rejected, self.ncall))

        highestpoint, highestlike = df.iloc[self.nlivepoints - 1]
        print("Parameter estimation : {}".format(highestpoint))
        print("logLikelihood : {}".format(highestlike))
        print("log(Z) : {}".format(logz))
        # print("log Z : {}".format(np.log(-z)))
