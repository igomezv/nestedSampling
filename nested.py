import numpy as np
from scipy.special import logsumexp

class nested:
    def __init__(self, loglike, priorTransform, nlive, ndims, maxiter=5000):
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
        #u, x, logx, logw, logz
        upoints = []
        vpoints = []
        logL = []
        logLstar = []
        logx = []
        logw = []
        logLw = []
        logz = []
        dead = []
        # Start nlive points
        for i in range(self.nlive):
            upoints.append(np.random.rand(self.ndims))
            vpoints.append(self.priorTransform(upoints[i]))
            logL.append(self.loglike(vpoints[i]))
            logLw.append(1e-300)
            logw.append(1e-300)
            logz.append(1e-300)
        # Begin the nested sampling loop
        clogz = -1e300
        # previous x, where x is the prior mass point X_i
        px = 1.

        for i in range(self.maxiter):
            # current x -> cx
            cx = np.exp(-(1.0+i) / self.nlive)
            clogw = np.log(px - cx)
            px = cx
            # Find worst index (min loglike)
            worst = np.argmin(logL)
            # log(L*w) = logL + logw
            clogLw = clogw + logL[worst]
            # Increment z += Lw
            clogz = np.logaddexp(clogz, clogLw)
            # Add worst objects to samples: v, logLw, logw, logL

            # Kill worst object in favour of copy of different survivor
            if self.nlive > 1:
                while True:
                    copy = np.random.randint(0, self.nlive)
                    if copy != worst:
                        break

            upoints[worst], vpoints[worst], logL[worst] = upoints[copy], vpoints[copy], logL[copy]
            upoints[worst], vpoints[worst], logL[worst] = self.explore(upoints[worst], logL[worst])
            # dead.append([upoints[worst], vpoints[worst], logL[worst],
            #             logw[worst], logLw[worst], logz[worst]])

            # rlogz -> logz remain in livepoints
            rlogz = np.max(np.array(logL)) + clogw
            cdlogz = np.logaddexp(clogz, rlogz) - clogz
            if cdlogz < dlogz:
                print("Stopping criteria!")
                break

            print("{}/{} | logz: {:.3f} | dlogz: {:.3f} | logw: {:.3f} "
                  "| logLstar: {:.3f} | v: {}".format(i + 1, self.maxiter, clogz,
                                                      cdlogz, clogw, logL[worst], vpoints[worst]))

            # Update Evidence Z
            clogz = logsumexp([clogz, clogLw])
            # Shrink interval of prior volume.
            clogw -= 1.0 / self.nlive

        # Last increment in Z
        sumloglx_over_n = np.sum(logL) + np.log(cx) - np.log(self.nlive)
        clogz = logsumexp([clogz, sumloglx_over_n])
        print("Final logZ: {}".format(clogz))

    def explore(self, uworst, logLstar):
        step = 0.1
        accept = 0
        reject = 0
        for _ in range(50):
            pu = uworst + step * (2.*np.random.random(self.ndims) - 1.)
            pu -= np.floor(pu)
            pv = self.priorTransform(pu)
            ploglike = self.loglike(pv)
            # Hard likelihood constraint
            if ploglike > logLstar:
                accept += 1
            else:
                reject += 1
            # Refine step-size
            if accept > reject:
                step *= np.exp(1.0 / accept)
            elif accept < reject:
                step /= np.exp(1.0 / reject)
        return pu, pv, ploglike

