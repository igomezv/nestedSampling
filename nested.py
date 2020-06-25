import sys
import numpy as np
from scipy.special import logsumexp

np.random.seed(0)

class nested:
    def __init__(self, loglike, priorTransform, nlive, ndims, maxiter=100000):
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
        # current x -> cx
        cx = np.exp(-1.0/ self.nlive)
        clogw = np.log(px-cx)
        for i in range(self.maxiter):
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

            # while True:
            #     idx = np.random.randint(self.nlive)
            #     if idx != worst:
            #         u = lupoints[idx, :]
            #         break

            loglstar = lloglikes[worst]
            nu, nv, nlogl = self.explore(lupoints[worst, :], loglstar)
            lupoints[worst] = nu
            lvpoints[worst] = nv
            lloglikes[worst] = nlogl
            lloglw[worst] = clogw

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
        # sumloglx_over_n = np.sum(lloglikes) + np.log(cx) - np.log(self.nlive)
        # clogz = logsumexp([clogz, sumloglx_over_n])
        # print("Final logZ: {}".format(clogz))
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

    def explore(self, uworst, logLstar, nsteps=30):
        step = 0.1
        accept = 0
        reject = 0
        pu = uworst
        pv = self.priorTransform(pu)
        ploglike = logLstar

        for _ in range(nsteps):
            diag = np.array([step**2]*self.ndims)
            cov = np.diag(diag)
            while True:
                # Generate a candidate point
                tryu = np.random.multivariate_normal(pu, cov)
                # tryu = pu + step * (2. * np.random.random(self.ndims) - 1.)
                # Force that the point lies in [0, 1]
                if np.all(tryu > 0.) and np.all(tryu < 1.):
                     break
            # Obtain the respective physical point
            tryv = self.priorTransform(tryu)
            # Evaluate loglike in the try point
            tryloglike = self.loglike(tryv)
            # acceptance ratio r
            # if min(tryloglike - ploglike, 0) > np.log(np.random.uniform(0, 1)):
            # or hard like constrain
            if tryloglike > ploglike:
                accept += 1
                ploglike = tryloglike
                pu = tryu
                pv = tryv
            else:
                reject += 1
            # Refine step-size
            if reject > accept:
                step /= np.exp(1.0 / reject)
            elif accept > reject:
                step *= np.exp(1.0 / accept)

        return pu, pv, ploglike


