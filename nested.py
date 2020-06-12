import sys
import numpy as np
from scipy.special import logsumexp

class nested:
    def __init__(self, loglike, priorTransform, nlive, ndims, maxiter=10000):
        self.loglike = loglike
        self.priorTransform = priorTransform
        self.ndims = ndims
        self.nlive = nlive
        self.maxiter = maxiter

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
        # Start nlive points
        print("Creating first nlive points")
        for i in range(self.nlive):
            lvpoints[i, :] = self.priorTransform(lupoints[i, :])
            lloglikes[i] = self.loglike(lvpoints[i, :])
            print("{} live point created".format(i+1))
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
        # previous x, where x is the prior mass point X_i
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
            #Kill worst object in favour of copy of different survivor
            while True:
                copy = np.random.randint(self.nlive)
                if copy != worst:
                    u = lupoints[copy, :]
                    # print("u", u)
                    break
            nu, nv, nlogl = self.explore(u, loglstar)
            lupoints[worst] = nu
            lvpoints[worst] = nv
            lloglikes[worst] = nlogl

            # Update Evidence Z
            clogz = logsumexp([clogz, clogLw])

            # Shrink interval of prior volume.
            clogw -= 1.0 / self.nlive
            #plist.append(prow)
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

    def explore(self, uworst, logLstar):
        step = 0.1
        accept = 0
        reject = 0
        ncall = 0
        n = 20
        for _ in range(n):
            while True:
                pu = uworst + step * np.random.randn(self.ndims)
                if np.all(pu > 0.) and np.all(pu < 1.):
                     break
            pv = self.priorTransform(pu)
            ploglike = self.loglike(pv)
            if ploglike >= logLstar:
                accept += 1
                break
            else:
                reject += 1
            # Refine step-size
            if accept > reject:
                step *= np.exp(1.0 / accept)
            elif accept < reject:
                step /= np.exp(1.0 / reject)
        return pu, pv, ploglike

