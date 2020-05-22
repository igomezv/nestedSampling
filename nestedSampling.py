# Nested sampling based on Skilling (2009)
import numpy as np
import pandas as pd
import os.path
from scipy.special import logsumexp
import sys


class SkillingNS:
    def __init__(self, logLike, priorTransform, nDims, bounds, nlivepoints=50):
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
        self.accepted = 0
        self.rejected = 0
        # self.ncall = 0

    def sampler(self, accuracy=0.01, maxiter=10000, outputname=None):
        vectors = []
        # vector of loglikes
        loglikes = []
        # list for dead values
        deadpoints = []
        deadloglikes = []
        if outputname is None:
            pass
        elif os.path.isfile(outputname + '.txt'):
            print("Output file exists! Please choose another"
                  " name or move the existing file.")
            sys.exit(1)
        else:
            f = open(outputname + '.txt', 'w+')

        for i in range(self.nlivepoints):
            vectors.append(self.priorTransform(np.random.rand(self.nDims, )))
            loglikes.append(self.logLike(vectors[i]))
            if outputname:
                strvector = str(vectors[i]).lstrip('[').rstrip(']')
                f.write("{} {} {}\n".format(0, loglikes[i], strvector))

        # join loglikes and points
        df_live = pd.DataFrame()
        df_live['points'] = vectors
        df_live['loglikes'] = loglikes
        print(loglikes)
        print("data frame", df_live.head())
        logz = -np.inf
        # initial prior mass is x0 = 1, z= 0
        plogx = 0
        # h = 0
        for i in range(maxiter):
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
            clogx = -(i + 1) / float(self.nlivepoints)
            print("logX_i-1: {}, clogXi: {}".format(plogx, clogx))
            logwi = logsumexp([plogx, clogx], b=[1, -1])
            print("logw_i: {}".format(logwi))
            logLwi = lowloglike + logwi
            # z increment
            logz = logsumexp([logz, logLwi])
            print("lowpoint: {}".format(lowpoint))
            # newpoint, newlike = self.generate_point(lowpoint, lowloglike)
            newpoint, newlike = self.metropolis(lowpoint, lowloglike, 500)
            print("new point: {}".format(newpoint))
            df_live.iloc[0] = newpoint, newlike
            self.print_func(df_live, logz)
            plogx = clogx
            if outputname:
                strnewpoint = str(newpoint).lstrip('[').rstrip(']')
                f.write("{} {} {}\n".format(logwi, newlike, strnewpoint))
            # What is a good value for f?
            stop = self.stoppingCriteria(df_live['loglikes'].values, clogx, logz, f=accuracy)
            if stop:
                break
        if outputname:
            f.close()
        lgsumexplglikes = logsumexp(df_live['loglikes'].values)
        logzsum = lgsumexplglikes + clogx - np.log(float(self.nlivepoints))
        logz = logsumexp([logz, logzsum])
        print("logz : {}".format(logz))

        df_dead = pd.DataFrame()
        df_dead['points'] = deadpoints
        df_dead['loglikes'] = deadloglikes
        samples = pd.concat([df_dead, df_live], ignore_index=True)

        return ({'nlive': self.nlivepoints, 'niter': i, 'samples': samples,
                 'logwi': logwi, 'logz': logz})


    def metropolis(self, ctheta, cloglike, iter):
        logf = lambda x: self.logLike(x) + self.logPrior(x)
        # samples = np.zeros((iter, 2))
        self.accepted = 0
        self.rejected = 0

        for i in range(iter):
            #propossal dist
            vstar = ctheta + np.random.normal(size=len(ctheta))
            r = np.random.rand()
            #q:
            if logf(vstar) - logf(ctheta) > np.log(r):
                ctheta = vstar
                cloglike = self.logLike(ctheta)
                self.accepted += 1
            else:
                self.rejected += 1

        # samples[i] = ctheta
        return ctheta, cloglike

    def logPrior(self, theta):
        flag = True
        for i, bound in enumerate(self.bounds):
            if bound[0] < theta[i] < bound[1]:
                pass
            else:
                flag = False
                break
        if flag:
            return 0.0
        else:
            return -np.inf

    def stoppingCriteria(self, loglikes, logx, logz, f=0.01):
        maxloglike = np.max(loglikes)
        if maxloglike + logx < logz + np.log(f):
            print("Stopping Criteria reached!")
            return True
        else:
            print("maxloglike+logx: {}, logz+logf: {}".format(
                maxloglike + logx, logz + np.log(f)))
            return False

    def print_func(self, df, logz):
        print("Accepted: {} || Rejected: {} ".format(
            self.accepted, self.rejected))

        highestpoint, highestlike = df.iloc[self.nlivepoints - 1]
        print("Parameter estimation : {}".format(highestpoint))
        print("logLikelihood : {}".format(highestlike))
        print("log(Z) : {}".format(logz))

class Parameter:
    def __init__(self, inivalue, bounds, name, LatexName):
        self.inivalue = inivalue
        self.bounds = bounds
        self.name = name
        self.LatexName = LatexName

    def paramFile(selfs, outputname):
        pass
