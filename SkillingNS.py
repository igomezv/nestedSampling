# Nested sampling based on Skilling (2009)
import numpy as np
import pandas as pd
import os.path
from scipy.special import logsumexp
import sys

class SkillingNS:
    def __init__(self, logLike, priorTransform, nDims, bounds, nlivepoints=50,
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
        self.outputname = "outputSamples"
        self.accepted = 0
        self.rejected = 0
        # self.ncall = 0

    def sampler(self):
        vectors = []
        # vector of loglikes
        loglikes = []
        # list for dead values
        deadpoints = []
        deadloglikes = []
        if os.path.isfile(self.outputname+'.txt'):
    		print("File exist! Please choose another name or move the existing file.")
    		sys.exit(1)
    	else:
        	f = open(self.outputname + '.txt', 'a+')

        for i in range(self.nlivepoints):
            vectors.append(self.priorTransform(np.random.rand(self.nDims, ),
                                               self.bounds))
            loglikes.append(self.logLike(vectors[i]))
            strvector = str(vectors[i]).lstrip('[').rstrip(']')
            f.write("{} {} {}\n".format(0, loglikes[i], strvector))

        # join loglikes and points
        df_live = pd.DataFrame()
        df_live['points'] = vectors
        df_live['loglikes'] = loglikes
        print(loglikes)
        print("data frame", df_live.head())
        # 2) initialise S = 0, X_0 = 1
        # evidence is z
        # print("bounds", self.bounds)
        # What is a good initial value for logz?
        # logz = -np.inf
        # logz = 1e-50

        z = 0
        # initial prior mass is x0 = 1
        # x_prev = 1 -> logx =0
        logx = 0
        # x_prev = 1
        #x_current = 1
        clogx = 0

        # h = 0
        # j iterations
        j = 10000
        for i in range(j):
            print("\nIteration {}".format(i + 1))
            # sort points by loglikes
            df_live.sort_values('loglikes', inplace=True)
            df_live.reset_index(drop=True, inplace=True)
            # sleep(0.5)
            # record the lower loglike,  L_i in the skilling paper
            lowpoint, lowloglike = df_live.iloc[0]
            deadloglikes.append(lowloglike)
            deadpoints.append(lowpoint)
            # new prior mass X_i in the paper [crude]
            
            # x_current = np.exp(-(i + 1) / self.nlivepoints)
            clogx = -(i + 1) / self.nlivepoints
            # xi = -i / self.nlivepoints
            print("logX_i : {}".format(clogx))
            # wi simple (no trapezoidal)
            #wi = x_prev - x_current
            # Error! use scipy
            # logwi = np.logaddexp(logx, clogx)
            logwi = logsumexp([logx, clogx], b = [1, -1])
            
            print("logw_i : {}".format(logwi))
            logLwi = lowloglike - logwi
            # z increment
            z += np.exp(lowloglike) * np.exp(logwi)
            # logz = np.logaddexp(logz, logLwi)
            print("lowpoint {}".format(lowpoint))
            # newpoint, newlike = self.generate_point(lowpoint, lowloglike)
            newpoint, newlike = self.metropolis(lowpoint, lowloglike, 500)

            print("new point {}".format(newpoint))
            df_live.iloc[0] = newpoint, newlike
            self.print_func(df_live, np.log(z))
            # x_prev = x_current
            logx = clogx
            strnewpoint = str(newpoint).lstrip('[').rstrip(']')
            f.write("{} {} {}".format(logwi, newlike, strnewpoint))

        f.close()

        # logz = np.logaddexp(logz, np.logaddexp(df_live['loglikes'].values) + logx - np.log(self.nlivepoints))
        # logz = np.logaddexp(logz, np.sum(df_live['loglikes'].values + logx - np.log(self.nlivepoints)))
        z += np.sum(np.exp(df_live['loglikes'].values)) * np.exp(clogx) / self.nlivepoints

        df_dead = pd.DataFrame()
        df_dead['points'] = deadpoints
        df_dead['loglikes'] = deadloglikes
        samples = pd.concat([df_dead, df_live], ignore_index=True)
        # Only in order to visualize the total samples:
        # for row in samples.values:
        #     print(row)

    def rejection_sampling(self, lpoint, llike):
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

        new_point = lpoint
        new_loglike = llike
        # while (ncalls < 1000 or accepted == 0):
        while (new_loglike < llike or self.accepted == 0):
            # print("new point  = llike", new_point)
            proposal_point = np.random.rand(self.nDims, )
            # new_point = self.priorTransform(proposal_point, bounds)
            new_point = self.priorTransform(proposal_point, self.bounds)
            new_loglike = self.logLike(new_point)
            if new_loglike == llike:
                epsilon = 1e-10 * np.random.rand()
                # See equation 6 of Skilling's paper
                # What is a good value for epsilon?
                self.accepted += 1
            elif new_loglike > llike:
                self.accepted += 1
            else:
                self.rejected += 1

        return new_point, new_loglike

    def print_func(self, df, logz):
        print("Accepted: {} || Rejected: {} ".format(
            self.accepted, self.rejected))

        highestpoint, highestlike = df.iloc[self.nlivepoints - 1]
        print("Parameter estimation : {}".format(highestpoint))
        print("logLikelihood : {}".format(highestlike))
        print("log(Z) : {}".format(logz))
	
    def metropolis(self, ctheta, cloglike, iter):
        logf = lambda x : self.logLike(x) + self.logPrior(x)       
        samples = np.zeros((iter, 2))
        self.accepted = 0
        self.rejected = 0
        
        for i in range(iter):
            vstar = np.array(ctheta) + np.random.normal(size=len(ctheta))
	    	# logLikeStar = self.logLike(vstar)
            r = np.random.rand()
            if logf(vstar) - logf(ctheta) > np.log(r):
                ctheta = vstar
                cloglike = self.logLike(ctheta)
                self.accepted+=1
            else:
                self.rejected+=1
            
        samples[i] = np.array(ctheta)
        return np.array(ctheta), cloglike

    def logPrior(self, theta):
        for i, bound in enumerate(self.bounds):
            if bound[0] < theta[i] < bound[1]:
                flag = True
            else:
                flag = False
                break  
        
        if flag == True:
            return 0.0
        else:
            return -np.inf