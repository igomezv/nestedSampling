import sys
import numpy as np
from scipy.special import logsumexp

class nested:
    def __init__(self, loglike, priorTransform, nlive, ndims, maxiter=1000000):
        self.loglike = loglike
        self.priorTransform = priorTransform
        self.ndims = ndims
        self.nlive = nlive
        self.maxiter = maxiter
        self.mcaccept = 0
        self.mcreject = 0
        self.mcsigma = 0.1

    def sampling(self, dlogz=0.01):
        """
        dlogz = 0.01 -> f = exp(0.01) = 1.01
        The stopping criteria is when the volume
        maxloglike*prior_mass increase by 0.01 the current Z
        So:
            maxLoglikes * Xj < Z * 1.01 = Z * (1  + 0.01)
        """
        # l -> live lloglikes-> live loglikes
        lupoints = np.random.rand(self.nlive, self.ndims)  # position in unit cube
        lvpoints = np.empty((self.nlive, self.ndims), dtype=np.float64)  # real params
        lloglikes = np.empty((self.nlive,), dtype=np.float64)
        lloglw = np.empty((self.nlive,), dtype=np.float64)
        # Start nlive points
        print("Creating first nlive points")
        for i in range(self.nlive):
            lvpoints[i, :] = self.priorTransform(lupoints[i, :])
            lloglikes[i] = self.loglike(lvpoints[i, :])
            lloglw[i] = 0.0
            print("{} live point created: {}, logl: {}".format(i+1, lvpoints[i, :], lloglikes[i]))
            # print("{} {}".format(lvpoints[i, :], lloglikes[i]))
        print("live points created")
        # Begin the nested sampling loop
        # s -> saved
        svpoints = []
        slogL = []
        slogLw = []
        slogw = []
        # initial values
        clogz = -1e300
        # previous x, where x is the prior mass point X_i -> X_0 = 1
        px = 1.

        for i in range(self.maxiter):
            # current x -> cx
            cx = np.exp(-(1.0+i) / self.nlive)
            clogw = np.log(px - cx)
            px = cx
            # Find worst index (min loglike)
            worst = np.argmin(lloglikes)
            # print("worst:", worst)
            # log(L*w) = logL + logw
            clogLw = clogw + lloglikes[worst]
            # Increment z += Lw
            clogz = np.logaddexp(clogz, clogLw)
            # Add worst objects to samples: v, logLw, logw, logL
            svpoints.append(np.array(lvpoints[worst]))
            slogL.append(lloglikes[worst])
            slogLw.append(clogLw)
            slogw.append(clogw)

            loglstar = lloglikes[worst]
            # #Kill worst object in favour of copy of different survivor
            while True:
                copy = np.random.randint(self.nlive)
                if copy != worst:
                    u = lupoints[worst, :]
                    break
            nu, nv, nlogl = self.explore(u, loglstar)
            lupoints[worst] = nu
            lvpoints[worst] = nv
            lloglikes[worst] = nlogl
            lloglw[worst] = clogw

            # Update Evidence Z
            clogz = logsumexp([clogz, clogLw])

            # Shrink interval of prior volume.
            clogw -= 1.0 / self.nlive
            # rlogz -> logz remain in livepoints
            rlogz = np.max(lloglikes) + clogw
            cdlogz = np.logaddexp(clogz, rlogz) - clogz
            if cdlogz < dlogz:
                print("Stopping criteria!")
                break
            print("{}/{} | worst {} | logz: {:.3f} | dlogz: {:.3f} | logw: {:.3f} "
                  "| logLstar: {:.3f} | v: {}".format(i + 1, self.maxiter, worst, clogz,
                                                      cdlogz, clogw, loglstar, lvpoints[worst]))

        # Last increment in Z
        sumloglx_over_n = np.sum(lloglikes) + np.log(cx) - np.log(self.nlive)
        clogz = logsumexp([clogz, sumloglx_over_n])
        print("Final logZ: {}".format(clogz))
        finalx = np.exp(clogLw)/self.nlive
        # # Adding last live points to posterior samples
        for i in range(self.nlive):
             slogLw.append(finalx)
             slogL.append(lloglikes[i])
             svpoints.append(lvpoints[i])
        # Postprocessing
        normws = logsumexp(np.array(slogLw))
        nLw = np.exp(np.array(slogLw) - normws)
        nsamp, _ = np.shape(svpoints)
        f = open("posteriors.txt", "+w")
        for i in range(nsamp):
            strv=""
            for el, p in enumerate(svpoints[i]):
                if el==0:
                    strv = "{}{}".format(strv, p)
                else:
                    strv = "{} {}".format(strv, p)
            f.write("{} {} {}\n".format(nLw[i], slogL[i], strv))
        f.close()

    def explore(self, uworst, logLstar, nsteps=100):
        # each nsteps * 2 callbacks check if the step size can be ajusted.
        if self.mcreject + self.mcreject > 2*nsteps:
            if self.mcaccept < self.mcreject:
                self.mcsigma /= np.exp(1.0 / self.mcreject)
            self.mcaccept = 0
            self.mcreject = 0

        for _ in range(nsteps):
            diag = np.array([self.mcsigma**2]*self.ndims)
            cov = np.diag(diag)
            while True:
                # Generate a candidate point
                # pu = np.random.multivariate_normal(uworst, cov)
                # pu = uworst + self.mcsigma * (2. * np.random.random(self.ndims) - 1.)
                pu = np.random.random(self.ndims)
                # Force that the point lies in [0, 1]
                if np.all(pu > 0.) and np.all(pu < 1.):
                     break
            # Obtain the respective physical point
            pv = self.priorTransform(pu)
            # Evaluate loglike in the proposal point
            ploglike = self.loglike(pv)
            # acceptance ratio r
            if min(ploglike-logLstar, 0) > np.log(np.random.uniform(0, 1)):
                self.mcaccept += 1
                return pu, pv, ploglike
            else:
                self.mcreject += 1

        return uworst, self.priorTransform(uworst), logLstar


