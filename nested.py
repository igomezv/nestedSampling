import sys
import numpy as np
from scipy.special import logsumexp
import scipy as sc

class nested:
    def __init__(self, loglike, priorTransform, nlive, ndims, maxiter=100000,
                 outputname="outputs/test", sampletype='diff',
                 mc_scale=0.0001, ajust_mcstep=False):
        # mcmc fixed step_size = (0.001-0.01), mcmc_cov step_size = (0.0001-0.001)
        # self.sampletype = {'mcmc', 'mcmc_cov', 'diff'}
        self.loglike = loglike
        self.priorTransform = priorTransform
        self.ndims = ndims
        self.nlive = nlive
        self.maxiter = maxiter
        self.outputname = outputname
        # Following is for mcmc_explore
        self.mcaccept = 0
        self.mcreject = 0
        self.list_accept = []
        self.ajust_mcstep = ajust_mcstep
        # self.scale = 1 # if ajust_scale = True
        self.scale = mc_scale
        self.sampletype = sampletype

    def sampling(self, dlogz=0.01):
        """
        dlogz = 0.01 -> f = exp(0.01) = 1.01
        The stopping criteria is when the volume
        maxloglike*prior_mass increase by 0.01 the current Z
        So:
            maxLoglikes * Xj < Z * 1.01 = Z * (1  + 0.01)
        """
        lupoints = np.empty((self.nlive, self.ndims), dtype=np.float64)  # unit cube params
        lvpoints = np.empty((self.nlive, self.ndims), dtype=np.float64)  # physical params
        lloglikes = np.empty((self.nlive,), dtype=np.float64)
        lloglw = np.empty((self.nlive,), dtype=np.float64)
        llogLstar = np.empty((self.nlive,), dtype=np.float64)
        # Start nlive points
        print("Creating first nlive points")
        for i in range(self.nlive):
            lupoints[i, :] = np.random.rand(self.ndims)
            lvpoints[i, :] = self.priorTransform(lupoints[i, :])
            lloglikes[i] = self.loglike(lvpoints[i, :])
            llogLstar[i] = -np.inf
            lloglw[i] = 0.0
            print("{} live point created: {}, logl: {}".format(i+1, lvpoints[i, :], lloglikes[i]))
            # print("{} {}".format(lvpoints[i, :], lloglikes[i]))
        print("first live points created")
        # Begin the nested sampling loop
        # s -> saved
        svpoints = []
        slogL = []
        slogLw = []
        slogw = []
        slogLstar = []
        # initial values
        clogz = -1e300
        # previous x, where x is the prior mass point X_i -> X_0 = 1
        px = 1.
        # current x -> cx
        cx = np.exp(-1.0 / self.nlive)
        clogw = np.log(px-cx)
        for i in range(self.maxiter):
            self.saveFile([lvpoints, lloglikes, llogLstar])
            # Find worst index (min loglike)
            worst = np.argmin(lloglikes)
            # print("worst:", worst)
            # log(L*w) = logL + logw
            clogLw = clogw + lloglikes[worst]
            # Update Z-> Increment z += Lw
            clogz = logsumexp([clogz, clogLw])
            # Add worst objects to samples: v, logLw, logw, logL
            svpoints.append(np.array(lvpoints[worst]))
            slogL.append(lloglikes[worst])
            slogLw.append(clogLw)
            slogw.append(clogw)
            slogLstar.append(llogLstar[worst])

            loglstar = lloglikes[worst]

            if self.sampletype == 'diff':
                nu, nv, nlogl = self.diff_evol(lupoints, worst, loglstar)
            else:
                idx = np.random.randint(self.nlive) # choose another point to sample from it
                if self.sampletype == 'mcmc_cov':
                    livecov = np.cov(lupoints[:, 0], lupoints[:, 1])
                    nu, nv, nlogl = self.mcmc_explore(lupoints[idx], lvpoints[idx], loglstar, cov=livecov)
                elif self.sampletype == 'mcmc':
                    nu, nv, nlogl = self.mcmc_explore(lupoints[idx], lvpoints[idx], loglstar)

            lupoints[worst] = nu
            lvpoints[worst] = nv
            lloglikes[worst] = nlogl
            lloglw[worst] = clogw
            llogLstar[worst] = loglstar

            # Shrink interval of prior volume.
            clogw -= 1.0 / self.nlive
            # rlogz -> logz remain in livepoints
            rlogz = np.max(lloglikes) + clogw
            cdlogz = np.logaddexp(clogz, rlogz) - clogz
            if cdlogz < dlogz:
                print("Stopping criteria!")
                break
            print("{}/{} | worst {} | logz: {:.3f} | dlogz: {:.3f} | logw: {:.3f} | "
                  "logLw: {:.3f}| logLstar: {:.3f} | v: {}".format(i + 1, self.maxiter, worst, clogz,
                                                      cdlogz, clogw, clogLw, loglstar, lvpoints[worst]))

        # Last increment in Z
        sumloglx_over_n = np.sum(lloglikes) + np.log(cx) - np.log(self.nlive)
        clogz = logsumexp([clogz, sumloglx_over_n])
        print("Final logZ: {}".format(clogz))

        self.saveFile([svpoints, slogL, slogLstar], type="dead")
        # # Adding last live points to posterior samples
        for p in range(self.nlive):
             slogLw.append(lloglw[p])
             slogL.append(lloglikes[p])
             svpoints.append(lvpoints[p])
        # # Postprocessing
        #normws = logsumexp(np.array(slogLw))
        # nLw = np.exp(np.array(slogLw) - normws)
        pi = np.exp(slogL + slogw - clogz)
        self.saveFile([svpoints, slogL, pi], type="post")

        return {'it': i+1, 'logz': clogz, 'dlogz': dlogz, 'loglw': slogLw, 'logw' : slogw,
                'logl': slogL, 'loglstar' : slogLstar, 'samples' : svpoints}


    def diff_evol(self, upoints, worstidx, logLstar):
        while True:
            idx1 = np.random.randint(self.nlive)
            while True:
                idx2 = np.random.randint(self.nlive)
                if idx2 != idx1:
                    break
            # Generate a candidate point
            tryu = upoints[worstidx] + upoints[idx2] - upoints[idx1]
            tryv = self.priorTransform(tryu)
            tryloglike = self.loglike(tryv)
            if tryloglike >= logLstar:
                break

        return tryu, tryv, tryloglike

    def mcmc_explore(self, uworst, vworst, logLstar, cov=None, nsteps=100):
        pu = uworst
        pv = vworst
        ploglike = logLstar
        if cov is not None:
            # take the diagonal of covariance matrix
            for j, item in enumerate(cov):
                for k, c in enumerate(item):
                    if j != k:
                        cov[j, k] = 0
            #normalize values of covariance matrix
            sumcov = np.sum(cov)
            cov = cov/sumcov

        while(True):
            if len(self.list_accept) > 0:
                if self.ajust_mcstep:
                    m = np.mean(self.list_accept)
                    if m > 0.5:
                        self.scale *= np.exp(1. / np.sum(self.list_accept))
                    else:
                        self.scale /= np.exp(1. / (len(self.list_accept) - np.sum(self.list_accept)))
                    self.list_accept = []
            naccepts = 0
            for it in range(nsteps):
                while True:
                    # Generate a candidate point
                    if cov is not None:
                        tryu = np.random.multivariate_normal(uworst, self.scale*cov)
                    # as in nested_sampling Buchner implementation from github
                    else:
                        tryu = uworst + np.random.normal(0, self.scale, size=self.ndims)
                    # Force that the point lies in [0, 1]
                    if np.all(tryu >= 0.) and np.all(tryu <= 1.):
                        break
                # Obtain the respective physical point
                # Evaluate loglike in the try point
                tryv = self.priorTransform(tryu)
                tryloglike = self.loglike(tryv)
                #print("current like: {}, try like: {}".format(logLstar, tryloglike))
                # acceptance ratio r
                #if min(tryloglike - ploglike, 0) > np.log(np.random.uniform(0, 1)):
                # or hard like constrain
                # accept = L > Li or numpy.random.uniform() < exp(L - Li)
                accept = tryloglike >= ploglike
                if accept:
                    ploglike = tryloglike
                    pu = tryu
                    pv = tryv
                    naccepts+=1
                if self.ajust_mcstep is False:
                    if tryloglike >= ploglike:
                        break
                else:
                    self.list_accept.append(accept)
                    if it > 100:
                        if tryloglike >= ploglike:
                            break
            if naccepts > 0:
                break

        return pu, pv, ploglike


    def saveFile(self, result, type='live', fname=None):
        points, logls, star = result
        if type == 'post':
            fname = self.outputname+"_posteriors.txt"
        elif type == 'live':
            fname = self.outputname+"_phys_live-birth.txt"
        elif type == 'dead':
            fname = self.outputname + "_dead-birth.txt"
        f = open(fname, "+w")
        nsamp, _ = np.shape(points)
        for i in range(nsamp):
            strp = ""
            for el, p in enumerate(points[i]):
                if el == 0:
                    strp = "{}{}".format(strp, p)
                else:
                    strp = "{} {}".format(strp, p)
            if type == "post":
                f.write("{} {} {}\n".format(star[i], logls[i], strp))
            else:
                f.write("{} {} {}\n".format(strp, logls[i], star[i]))
        f.close()
