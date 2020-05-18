# Nested sampling based on Skilling (2009)
import numpy as np
import pandas as pd
import os.path
from scipy.special import logsumexp
import sys
from tqdm import tqdm


class SkillingNS:
    def __init__(self, logLike, priorTransform, nDims, bounds, nlivepoints=50, names = None):
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
        livepoints = []
        # vector of loglikes
        liveloglikes = []
        # list for dead values
        deadpoints = []
        deadloglikes = []
        livebirth_contours = []
        deadbirth_contours = []
        dead_contours = []
        liveweights = []
        deadweights = []

        # if outputname:
        #     if os.path.isfile(outputname + '.txt'):
        #         print("Output file exists! Please choose another"
        #               " name or move the existing file.")
        #     sys.exit(1)
        # else:
        #     f = open(outputname + '.txt', 'w+')

        pbar = tqdm()

        for i in range(self.nlivepoints):
            pbar.update()
            livepoints.append(self.priorTransform(np.random.rand(self.nDims, )))
            liveloglikes.append(self.logLike(livepoints[i]))
            livebirth_contours.append(-np.inf)
            liveweights.append(1)

        # join loglikes and points
        df_live = pd.DataFrame()
        df_live['points'] = livepoints
        df_live['loglikes'] = liveloglikes
        df_live['birth_contours'] = livebirth_contours
        df_live['weights'] = liveweights

        logz = -np.inf
        # initial prior mass is x0 = 1, z= 0
        plogx = 0
        # h = 0
        for i in range(maxiter):
            # sort points by loglikes
            df_live.sort_values('loglikes', inplace=True)
            df_live.reset_index(drop=True, inplace=True)
            # sleep(0.5)
            # record the lower loglike,  L_i in the skilling paper
            worstpoint, worstloglike, birthcontour, worstweight = df_live.iloc[0]

            deadloglikes.append(worstloglike)
            deadpoints.append(worstpoint)
            deadbirth_contours.append(birthcontour)
            # print("worst weight {}".format(worstweight))
            deadweights.append(worstweight)
            # new prior mass X_i in the paper [crude]
            # x_current = np.exp(-(i + 1) / self.nlivepoints)
            clogx = -(i + 1) / float(self.nlivepoints)
            logwi = logsumexp([plogx, clogx], b=[1, -1])
            logLwi = worstloglike + logwi
            # z increment
            logz = logsumexp([logz, logLwi])
            newpoint, newlike = self.metropolis(worstpoint, worstloglike, 100)
            # the worst log like converts into the birth contour, right?
            df_live.iloc[0] = newpoint, newlike, worstloglike, logwi
            plogx = clogx

            pbar.set_description("accepted {} | rejected {} | loglike: {:.3f} | "
                                 "logz: {:.3f} | logw: {:.3f} "
                                 "| logX: {} | new point: {} ".format(
                                    self.accepted, self.rejected, newlike,
                                    logz, logwi, clogx, newpoint))
            pbar.update()
            self.saveDFtotxt(df_live, outputname)
            stop = self.stoppingCriteria(df_live['loglikes'].values, clogx, logz, f=accuracy)
            if stop:
                break

        pbar.close()
        lgsumexplglikes = logsumexp(df_live['loglikes'].values)
        logzsum = lgsumexplglikes + clogx - np.log(float(self.nlivepoints))
        logz = logsumexp([logz, logzsum])
        # print("logz : {}".format(logz))

        df_dead = pd.DataFrame()
        df_dead['points'] = deadpoints
        df_dead['loglikes'] = deadloglikes
        df_dead['birth_contours'] = deadbirth_contours
        df_dead['weights'] = deadweights
        self.saveDFtotxt(df_dead, outputname, ext='dead-birth')

        samples = pd.concat([df_dead, df_live], ignore_index=True)
        self.saveDFtotxt(samples, outputname, ext='1')

        return ({'nlive': self.nlivepoints, 'niter': i, 'samples': samples,
                 'logwi': logwi, 'logz': logz, 'deadbirth' : df_dead, 'livebirth': df_live})


    def metropolis(self, ctheta, cloglike, iter):
        logf = lambda x: self.logLike(x) + self.logPrior(x)
        # samples = np.zeros((iter, 2))
        self.accepted = 0
        self.rejected = 0

        for i in range(iter):
            #propossal dist
            vstar = ctheta + np.random.normal(size=len(ctheta))
            r = np.random.rand()
            # q:
            if logf(vstar) - logf(ctheta) > np.log(r) or i == iter-1:
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
            print("\nStopping Criteria reached!")
            return True
        else:
            # print("maxloglike+logx: {}, logz+logf: {}".format(
            #    maxloglike + logx, logz + np.log(f)))
            return False


    def saveDFtotxt(self, df, outputname, ext='live-birth'):
        f = open("{}_{}".format(outputname, ext), 'w+')
        if ext == '1':
            normws = logsumexp(df['weights'])
           # print("logaexp weights {}".format(normws))
        for _, row in df.iterrows():
            strpoint = "{}".format(row['points']).lstrip('[').rstrip(']').strip(',')
            strpoint = strpoint.replace(',', '')
            if ext == '1':
                strrow = "{} {} {}".format(np.exp(row['weights'] - normws), row['loglikes'], strpoint)
            else:
                strrow = "{} {} {}".format(strpoint, row['loglikes'], row['birth_contours'])
            strrow = strrow.replace('  ', ' ')
            f.write("{}\n".format(strrow))
        f.close()
